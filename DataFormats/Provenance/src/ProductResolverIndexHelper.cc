
#include "DataFormats/Provenance/interface/ProductResolverIndexHelper.h"
#include "DataFormats/Provenance/interface/ViewTypeChecker.h"
#include "FWCore/Reflection/interface/DictionaryTools.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Reflection/interface/TypeWithDict.h"
#include "FWCore/Utilities/interface/WrappedClassName.h"
#include "FWCore/Utilities/interface/getAnyPtr.h"

#include <TClass.h>

#include <cassert>
#include <iostream>
#include <limits>

namespace edm {

  namespace productholderindexhelper {
    TypeID getContainedTypeFromWrapper(TypeID const& wrappedTypeID, std::string const& className) {
      static int const vtcOffset =
          TClass::GetClass("edm::WrapperBase")->GetBaseClassOffset(TClass::GetClass("edm::ViewTypeChecker"));
      static TClass const* const wbClass = TClass::GetClass("edm::WrapperBase");
      static std::string const refVector("edm::RefVector<");
      static std::string const refToBaseVector("edm::RefToBaseVector<");
      static std::string const ptrVector("edm::PtrVector<");
      static std::string const vectorPtr("std::vector<edm::Ptr<");
      static std::string const vectorUniquePtr("std::vector<std::unique_ptr<");
      static std::string const associationMap("edm::AssociationMap<");
      static std::string const newDetSetVector("edmNew::DetSetVector<");
      static size_t const rvsize = refVector.size();
      static size_t const rtbvsize = refToBaseVector.size();
      static size_t const pvsize = ptrVector.size();
      static size_t const vpsize = vectorPtr.size();
      static size_t const vupsize = vectorUniquePtr.size();
      static size_t const amsize = associationMap.size();
      static size_t const ndsize = newDetSetVector.size();
      bool mayBeRefVector = (className.substr(0, rvsize) == refVector) ||
                            (className.substr(0, rtbvsize) == refToBaseVector) ||
                            (className.substr(0, pvsize) == ptrVector) || (className.substr(0, vpsize) == vectorPtr) ||
                            (className.substr(0, vupsize) == vectorUniquePtr);
      // AssociationMap and edmNew::DetSetVector do not support View and
      // this function is used to get a contained type that can be accessed
      // using a View. So return the void type in these cases.
      // In practice, they were the only types causing a problem, but any
      // type with a typedef named value_type that does not support
      // View might also cause problems and might need to be added here in
      // the future.
      if (className.substr(0, amsize) == associationMap || className.substr(0, ndsize) == newDetSetVector) {
        return TypeID(typeid(void));
      }
      TClass* cl = TClass::GetClass(wrappedTypeID.className().c_str());
      if (cl == nullptr) {
        return TypeID(typeid(void));
      }
      void* p = cl->New();
      int offset = cl->GetBaseClassOffset(wbClass) + vtcOffset;
      std::unique_ptr<ViewTypeChecker> checker = getAnyPtr<ViewTypeChecker>(p, offset);
      if (mayBeRefVector) {
        std::type_info const& ti = checker->memberTypeInfo();
        if (ti != typeid(void)) {
          return TypeID(ti);
        }
      }
      return TypeID(checker->valueTypeInfo());
    }

    TypeID getContainedType(TypeID const& typeID) {
      const std::string& className = typeID.className();
      TypeWithDict const wrappedType = TypeWithDict::byName(wrappedClassName(className));
      TypeID const wrappedTypeID = TypeID(wrappedType.typeInfo());
      return getContainedTypeFromWrapper(wrappedTypeID, className);
    }
  }  // namespace productholderindexhelper

  ProductResolverIndexHelper::ProductResolverIndexHelper()
      : nextIndexValue_(0),
        beginElements_(0),
        items_(new std::set<ProductResolverIndexHelper::Item>),
        processItems_(new std::set<std::string>) {}

  ProductResolverIndex ProductResolverIndexHelper::index(KindOfType kindOfType,
                                                         TypeID const& typeID,
                                                         char const* moduleLabel,
                                                         char const* instance,
                                                         char const* process) const {
    unsigned int iToIndexAndNames = indexToIndexAndNames(kindOfType, typeID, moduleLabel, instance, process);

    if (iToIndexAndNames == std::numeric_limits<unsigned int>::max()) {
      return ProductResolverIndexInvalid;
    }
    return indexAndNames_[iToIndexAndNames].index();
  }

  ProductResolverIndexHelper::Matches::Matches(ProductResolverIndexHelper const* productResolverIndexHelper,
                                               unsigned int startInIndexAndNames,
                                               unsigned int numberOfMatches)
      : productResolverIndexHelper_(productResolverIndexHelper),
        startInIndexAndNames_(startInIndexAndNames),
        numberOfMatches_(numberOfMatches) {
    if (numberOfMatches != 0 &&
        startInIndexAndNames_ + numberOfMatches_ > productResolverIndexHelper_->indexAndNames_.size()) {
      throw Exception(errors::LogicError)
          << "ProductResolverIndexHelper::Matches::Matches - Arguments exceed vector bounds.\n";
    }
  }

  ProductResolverIndex ProductResolverIndexHelper::Matches::index(unsigned int i) const {
    if (i >= numberOfMatches_) {
      throw Exception(errors::LogicError) << "ProductResolverIndexHelper::Matches::index - Argument is out of range.\n";
    }
    return productResolverIndexHelper_->indexAndNames_[startInIndexAndNames_ + i].index();
  }

  bool ProductResolverIndexHelper::Matches::isFullyResolved(unsigned int i) const {
    if (i >= numberOfMatches_) {
      throw Exception(errors::LogicError)
          << "ProductResolverIndexHelper::Matches::isFullyResolved - Argument is out of range.\n";
    }
    return (productResolverIndexHelper_->indexAndNames_[startInIndexAndNames_ + i].startInProcessNames() != 0U);
  }

  char const* ProductResolverIndexHelper::Matches::processName(unsigned int i) const {
    if (i >= numberOfMatches_) {
      throw Exception(errors::LogicError)
          << "ProductResolverIndexHelper::Matches::processName - Argument is out of range.\n";
    }
    unsigned int startInProcessNames =
        productResolverIndexHelper_->indexAndNames_[startInIndexAndNames_ + i].startInProcessNames();
    return &productResolverIndexHelper_->processNames_[startInProcessNames];
  }

  char const* ProductResolverIndexHelper::Matches::moduleLabel(unsigned int i) const {
    if (i >= numberOfMatches_) {
      throw Exception(errors::LogicError)
          << "ProductResolverIndexHelper::Matches::moduleLabel - Argument is out of range.\n";
    }
    unsigned int start =
        productResolverIndexHelper_->indexAndNames_[startInIndexAndNames_ + i].startInBigNamesContainer();
    return &productResolverIndexHelper_->bigNamesContainer_[start];
  }

  ProductResolverIndexHelper::Matches ProductResolverIndexHelper::relatedIndexes(KindOfType kindOfType,
                                                                                 TypeID const& typeID,
                                                                                 char const* moduleLabel,
                                                                                 char const* instance) const {
    unsigned int startInIndexAndNames = indexToIndexAndNames(kindOfType, typeID, moduleLabel, instance, nullptr);
    unsigned int numberOfMatches = 1;

    if (startInIndexAndNames == std::numeric_limits<unsigned int>::max()) {
      numberOfMatches = 0;
    } else {
      auto vSize = indexAndNames_.size();
      for (unsigned int j = startInIndexAndNames + 1U; j < vSize && (indexAndNames_[j].startInProcessNames() != 0U);
           ++j) {
        ++numberOfMatches;
      }
    }
    return Matches(this, startInIndexAndNames, numberOfMatches);
  }

  ProductResolverIndexHelper::Matches ProductResolverIndexHelper::relatedIndexes(KindOfType kindOfType,
                                                                                 TypeID const& typeID) const {
    unsigned int startInIndexAndNames = std::numeric_limits<unsigned int>::max();
    unsigned int numberOfMatches = 0;

    // Look for the type and check to see if it found it
    unsigned iType = indexToType(kindOfType, typeID);
    if (iType != std::numeric_limits<unsigned int>::max()) {
      // Get the range of entries with a matching TypeID
      Range const& range = ranges_[iType];

      startInIndexAndNames = range.begin();
      numberOfMatches = range.end() - range.begin();
    }
    return Matches(this, startInIndexAndNames, numberOfMatches);
  }

  ProductResolverIndex ProductResolverIndexHelper::insert(TypeID const& typeID,
                                                          char const* moduleLabel,
                                                          char const* instance,
                                                          char const* process,
                                                          TypeID const& containedTypeID,
                                                          std::vector<TypeWithDict>* baseTypesOfContainedType) {
    if (!items_) {
      throw Exception(errors::LogicError)
          << "ProductResolverIndexHelper::insert - Attempt to insert more elements after frozen.\n";
    }

    if (process == nullptr || *process == '\0') {
      throw Exception(errors::LogicError) << "ProductResolverIndexHelper::insert - Empty process.\n";
    }

    // Throw if this has already been inserted
    Item item(PRODUCT_TYPE, typeID, moduleLabel, instance, process, 0);
    std::set<Item>::iterator iter = items_->find(item);
    if (iter != items_->end()) {
      throw Exception(errors::LogicError)
          << "ProductResolverIndexHelper::insert - Attempt to insert duplicate entry.\n";
    }

    // Put in an entry for the product
    item.setIndex(nextIndexValue_);
    unsigned int savedProductIndex = nextIndexValue_;
    ++nextIndexValue_;
    items_->insert(item);

    // Put in an entry for the product with an empty process name
    // if it is not already there
    item.clearProcess();
    iter = items_->find(item);
    if (iter == items_->end()) {
      item.setIndex(nextIndexValue_);
      ++nextIndexValue_;
      items_->insert(item);
    }

    // Now put in entries for a contained class if this is a
    // recognized container.
    if (containedTypeID != TypeID(typeid(void)) && containedTypeID != TypeID()) {
      TypeWithDict containedType(containedTypeID.typeInfo());

      Item containedItem(ELEMENT_TYPE, containedTypeID, moduleLabel, instance, process, savedProductIndex);
      iter = items_->find(containedItem);
      if (iter != items_->end()) {
        containedItem.setIndex(ProductResolverIndexAmbiguous);
        items_->erase(iter);
      }
      items_->insert(containedItem);

      containedItem.clearProcess();
      iter = items_->find(containedItem);
      if (iter == items_->end()) {
        containedItem.setIndex(nextIndexValue_);
        ++nextIndexValue_;
        items_->insert(containedItem);
      }

      // Repeat this for all public base classes of the contained type
      if (baseTypesOfContainedType) {
        for (TypeWithDict const& baseType : *baseTypesOfContainedType) {
          TypeID baseTypeID(baseType.typeInfo());
          Item baseItem(ELEMENT_TYPE, baseTypeID, moduleLabel, instance, process, savedProductIndex);
          iter = items_->find(baseItem);
          if (iter != items_->end()) {
            baseItem.setIndex(ProductResolverIndexAmbiguous);
            items_->erase(iter);
          }
          items_->insert(baseItem);

          baseItem.clearProcess();
          iter = items_->find(baseItem);
          if (iter == items_->end()) {
            baseItem.setIndex(nextIndexValue_);
            ++nextIndexValue_;
            items_->insert(baseItem);
          }
        }
      }
    }
    return savedProductIndex;
  }

  ProductResolverIndex ProductResolverIndexHelper::insert(TypeID const& typeID,
                                                          char const* moduleLabel,
                                                          char const* instance,
                                                          char const* process) {
    TypeID containedTypeID = productholderindexhelper::getContainedType(typeID);
    bool hasContainedType = (containedTypeID != TypeID(typeid(void)) && containedTypeID != TypeID());
    std::vector<TypeWithDict> baseTypes;
    std::vector<TypeWithDict>* baseTypesOfContainedType = &baseTypes;
    if (hasContainedType) {
      std::vector<std::string> missingDictionaries;
      public_base_classes(missingDictionaries, containedTypeID, baseTypes);
    }
    return insert(typeID, moduleLabel, instance, process, containedTypeID, baseTypesOfContainedType);
  }

  void ProductResolverIndexHelper::setFrozen() {
    if (!items_)
      return;

    // Make a first pass and count things so we
    // can reserve memory in the vectors. Also
    // fill processItems_ on the first pass.
    bool iFirstThisType = true;
    bool beginElementsWasSet = false;
    TypeID previousTypeID;
    KindOfType previousKindOfType = PRODUCT_TYPE;
    std::string previousModuleLabel;
    std::string previousInstance;
    unsigned int iCountTypes = 0;
    unsigned int iCountCharacters = 0;
    for (auto const& item : *items_) {
      if (iFirstThisType || item.typeID() != previousTypeID || item.kindOfType() != previousKindOfType) {
        ++iCountTypes;
        iFirstThisType = true;

        if (!beginElementsWasSet) {
          if (item.kindOfType() == ELEMENT_TYPE) {
            beginElementsWasSet = true;
          } else {
            beginElements_ = iCountTypes;
          }
        }
      }

      processItems_->insert(item.process());

      if (iFirstThisType || item.moduleLabel() != previousModuleLabel || item.instance() != previousInstance) {
        iCountCharacters += item.moduleLabel().size();
        iCountCharacters += item.instance().size();
        iCountCharacters += 2;
      }

      iFirstThisType = false;
      previousTypeID = item.typeID();
      previousKindOfType = item.kindOfType();
      previousModuleLabel = item.moduleLabel();
      previousInstance = item.instance();
    }

    // Size and fill the process name vector
    unsigned int processNamesSize = 0;
    for (auto const& processItem : *processItems_) {
      processNamesSize += processItem.size();
      ++processNamesSize;
    }
    processNames_.reserve(processNamesSize);
    for (auto const& processItem : *processItems_) {
      for (auto const& c : processItem) {
        processNames_.push_back(c);
      }
      processNames_.push_back('\0');
      lookupProcessNames_.push_back(processItem);
    }

    // Reserve memory in the vectors
    sortedTypeIDs_.reserve(iCountTypes);
    ranges_.reserve(iCountTypes);
    indexAndNames_.reserve(items_->size());
    bigNamesContainer_.reserve(iCountCharacters);

    // Second pass. Really fill the vectors this time.
    bool iFirstType = true;
    iFirstThisType = true;
    previousTypeID = TypeID();
    unsigned int iCount = 0;
    unsigned int iBeginning = 0;
    iCountCharacters = 0;
    unsigned int previousCharacterCount = 0;
    if (!items_->empty()) {
      for (auto const& item : *items_) {
        if (iFirstThisType || item.typeID() != previousTypeID || item.kindOfType() != previousKindOfType) {
          iFirstThisType = true;
          sortedTypeIDs_.push_back(item.typeID());
          if (iFirstType) {
            iFirstType = false;
          } else {
            ranges_.push_back(Range(iBeginning, iCount));
          }
          iBeginning = iCount;
        }
        ++iCount;

        if (iFirstThisType || item.moduleLabel() != previousModuleLabel || item.instance() != previousInstance) {
          unsigned int labelSize = item.moduleLabel().size();
          for (unsigned int j = 0; j < labelSize; ++j) {
            bigNamesContainer_.push_back(item.moduleLabel()[j]);
          }
          bigNamesContainer_.push_back('\0');

          unsigned int instanceSize = item.instance().size();
          for (unsigned int j = 0; j < instanceSize; ++j) {
            bigNamesContainer_.push_back(item.instance()[j]);
          }
          bigNamesContainer_.push_back('\0');

          previousCharacterCount = iCountCharacters;

          iCountCharacters += labelSize;
          iCountCharacters += instanceSize;
          iCountCharacters += 2;
        }

        unsigned int processStart = processIndex(item.process().c_str());
        if (processStart == std::numeric_limits<unsigned int>::max()) {
          throw Exception(errors::LogicError)
              << "ProductResolverIndexHelper::setFrozen - Process not found in processNames_.\n";
        }
        indexAndNames_.emplace_back(item.index(), previousCharacterCount, processStart);

        iFirstThisType = false;
        previousTypeID = item.typeID();
        previousKindOfType = item.kindOfType();
        previousModuleLabel = item.moduleLabel();
        previousInstance = item.instance();
      }
      ranges_.push_back(Range(iBeginning, iCount));
    }

    // Some sanity checks to protect against out of bounds vector accesses
    // These should only fail if there is a bug. If profiling ever shows
    // them to be expensive one might delete them.
    sanityCheck();

    // Cleanup, do not need the temporary containers anymore
    // propagate_const<T> has no reset() function
    items_ = nullptr;
    processItems_ = nullptr;
  }

  std::vector<std::string> const& ProductResolverIndexHelper::lookupProcessNames() const {
    if (items_) {
      throw Exception(errors::LogicError)
          << "ProductResolverIndexHelper::lookupProcessNames - Attempt to access names before frozen.\n";
    }
    return lookupProcessNames_;
  }

  unsigned int ProductResolverIndexHelper::indexToIndexAndNames(KindOfType kindOfType,
                                                                TypeID const& typeID,
                                                                char const* moduleLabel,
                                                                char const* instance,
                                                                char const* process) const {
    // Look for the type and check to see if it found it
    unsigned iType = indexToType(kindOfType, typeID);
    if (iType != std::numeric_limits<unsigned int>::max()) {
      unsigned startProcess = 0;
      if (process) {
        startProcess = processIndex(process);
        if (startProcess == std::numeric_limits<unsigned int>::max()) {
          return std::numeric_limits<unsigned int>::max();
        }
      }

      ProductResolverIndexHelper::Range const& range = ranges_[iType];
      unsigned int begin = range.begin();
      unsigned int end = range.end();

      while (begin < end) {
        unsigned int midpoint = begin + ((end - begin) / 2);
        char const* namePtr = &bigNamesContainer_[indexAndNames_[midpoint].startInBigNamesContainer()];

        // Compare the module label
        char const* label = moduleLabel;
        while (*namePtr && (*namePtr == *label)) {
          ++namePtr;
          ++label;
        }
        if (*namePtr == *label) {  // true only if both are at the '\0' at the end of the C string
          ++namePtr;               // move to the next C string

          // Compare the instance name
          char const* instanceName = instance;
          while (*namePtr && (*namePtr == *instanceName)) {
            ++namePtr;
            ++instanceName;
          }
          if (*namePtr == *instanceName) {
            // Compare the process name
            if (startProcess == indexAndNames_[midpoint].startInProcessNames()) {
              return midpoint;
            } else {
              if (indexAndNames_[midpoint].startInProcessNames() > startProcess) {
                while (true) {
                  --midpoint;
                  if (indexAndNames_[midpoint].startInProcessNames() == startProcess) {
                    return midpoint;
                  }
                  if (indexAndNames_[midpoint].startInProcessNames() == 0)
                    break;
                }
              } else {
                while (true) {
                  ++midpoint;
                  if (midpoint == indexAndNames_.size())
                    break;
                  if (indexAndNames_[midpoint].startInProcessNames() == 0)
                    break;
                  if (indexAndNames_[midpoint].startInProcessNames() == startProcess) {
                    return midpoint;
                  }
                }
              }
              break;
            }
          } else if (*namePtr < *instanceName) {
            if (begin == midpoint)
              break;
            begin = midpoint;
          } else {
            end = midpoint;
          }
        } else if (*namePtr < *label) {
          if (begin == midpoint)
            break;
          begin = midpoint;
        } else {
          end = midpoint;
        }
      }  // end while (begin < end)
    }
    return std::numeric_limits<unsigned int>::max();
  }

  unsigned int ProductResolverIndexHelper::indexToType(KindOfType kindOfType, TypeID const& typeID) const {
    unsigned int beginType = 0;
    unsigned int endType = beginElements_;
    if (kindOfType == ELEMENT_TYPE) {
      beginType = beginElements_;
      endType = sortedTypeIDs_.size();
    }

    while (beginType < endType) {
      unsigned int midpointType = beginType + ((endType - beginType) / 2);
      if (sortedTypeIDs_[midpointType] == typeID) {
        return midpointType;  // Found it
      } else if (sortedTypeIDs_[midpointType] < typeID) {
        if (beginType == midpointType)
          break;
        beginType = midpointType;
      } else {
        endType = midpointType;
      }
    }
    return std::numeric_limits<unsigned int>::max();  // Failed to find it
  }

  unsigned int ProductResolverIndexHelper::processIndex(char const* process) const {
    char const* ptr = &processNames_[0];
    char const* begin = ptr;
    while (true) {
      char const* p = process;
      char const* beginName = ptr;
      while (*ptr && (*ptr == *p)) {
        ++ptr;
        ++p;
      }
      if (*ptr == *p) {
        return beginName - begin;
      }
      while (*ptr) {
        ++ptr;
      }
      ++ptr;
      if (static_cast<unsigned>(ptr - begin) >= processNames_.size()) {
        return std::numeric_limits<unsigned int>::max();
      }
    }
    return 0;
  }

  ProductResolverIndexHelper::ModulesToIndiciesMap ProductResolverIndexHelper::indiciesForModulesInProcess(
      const std::string& iProcessName) const {
    ModulesToIndiciesMap result;
    for (unsigned int i = 0; i < beginElements_; ++i) {
      auto const& range = ranges_[i];
      for (unsigned int j = range.begin(); j < range.end(); ++j) {
        auto const& indexAndNames = indexAndNames_[j];
        if (0 == strcmp(&processNames_[indexAndNames.startInProcessNames()], iProcessName.c_str())) {
          //The first null terminated string is the module label
          auto pModLabel = &bigNamesContainer_[indexAndNames.startInBigNamesContainer()];
          auto l = strlen(pModLabel);
          auto pInstance = pModLabel + l + 1;
          result.emplace(pModLabel, std::make_tuple(&sortedTypeIDs_[i], pInstance, indexAndNames.index()));
        }
      }
    }
    return result;
  }

  void ProductResolverIndexHelper::sanityCheck() const {
    bool sanityChecksPass = true;
    if (sortedTypeIDs_.size() != ranges_.size())
      sanityChecksPass = false;

    unsigned int previousEnd = 0;
    for (auto const& range : ranges_) {
      if (range.begin() != previousEnd)
        sanityChecksPass = false;
      if (range.begin() >= range.end())
        sanityChecksPass = false;
      previousEnd = range.end();
    }
    if (previousEnd != indexAndNames_.size())
      sanityChecksPass = false;

    unsigned maxStart = 0;
    unsigned maxStartProcess = 0;
    for (auto const& indexAndName : indexAndNames_) {
      if (indexAndName.index() >= nextIndexValue_ && indexAndName.index() != ProductResolverIndexAmbiguous)
        sanityChecksPass = false;

      if (indexAndName.startInBigNamesContainer() >= bigNamesContainer_.size())
        sanityChecksPass = false;
      if (indexAndName.startInProcessNames() >= processNames_.size())
        sanityChecksPass = false;

      if (indexAndName.startInBigNamesContainer() > maxStart)
        maxStart = indexAndName.startInBigNamesContainer();
      if (indexAndName.startInProcessNames() > maxStartProcess)
        maxStartProcess = indexAndName.startInProcessNames();
    }

    if (!indexAndNames_.empty()) {
      if (bigNamesContainer_.back() != '\0')
        sanityChecksPass = false;
      if (processNames_.back() != '\0')
        sanityChecksPass = false;
      if (maxStart >= bigNamesContainer_.size())
        sanityChecksPass = false;
      unsigned int countZeroes = 0;
      for (unsigned j = maxStart; j < bigNamesContainer_.size(); ++j) {
        if (bigNamesContainer_[j] == '\0') {
          ++countZeroes;
        }
      }
      if (countZeroes != 2)
        sanityChecksPass = false;
      if (maxStartProcess >= processNames_.size())
        sanityChecksPass = false;
      countZeroes = 0;
      for (unsigned j = maxStartProcess; j < processNames_.size(); ++j) {
        if (processNames_[j] == '\0') {
          ++countZeroes;
        }
      }
      if (countZeroes != 1)
        sanityChecksPass = false;
    }

    if (!sanityChecksPass) {
      throw Exception(errors::LogicError) << "ProductResolverIndexHelper::setFrozen - Detected illegal state.\n";
    }
  }

  ProductResolverIndexHelper::Item::Item(KindOfType kindOfType,
                                         TypeID const& typeID,
                                         std::string const& moduleLabel,
                                         std::string const& instance,
                                         std::string const& process,
                                         ProductResolverIndex index)
      : kindOfType_(kindOfType),
        typeID_(typeID),
        moduleLabel_(moduleLabel),
        instance_(instance),
        process_(process),
        index_(index) {}

  bool ProductResolverIndexHelper::Item::operator<(Item const& right) const {
    if (kindOfType_ < right.kindOfType_)
      return true;
    if (kindOfType_ > right.kindOfType_)
      return false;
    if (typeID_ < right.typeID_)
      return true;
    if (typeID_ > right.typeID_)
      return false;
    if (moduleLabel_ < right.moduleLabel_)
      return true;
    if (moduleLabel_ > right.moduleLabel_)
      return false;
    if (instance_ < right.instance_)
      return true;
    if (instance_ > right.instance_)
      return false;
    return process_ < right.process_;
  }

  void ProductResolverIndexHelper::print(std::ostream& os) const {
    os << "\n******* Dump ProductResolverIndexHelper *************************\n";

    os << "\nnextIndexValue_ = " << nextIndexValue_ << "\n";
    os << "beginElements_ = " << beginElements_ << "\n";

    os << "\n******* sortedTypeIDs_ \n";
    for (auto const& i : sortedTypeIDs_) {
      os << i << "\n";
    }
    os << "******* ranges_ \n";
    for (auto const& i : ranges_) {
      os << i.begin() << " " << i.end() << "\n";
    }
    os << "******* indexAndNames_ \n";
    for (auto const& i : indexAndNames_) {
      os << i.index() << " " << i.startInBigNamesContainer() << " ";
      char const* ptr = &bigNamesContainer_[i.startInBigNamesContainer()];
      while (*ptr) {
        os << *ptr;
        ++ptr;
      }
      ++ptr;
      os << " ";
      while (*ptr) {
        os << *ptr;
        ++ptr;
      }
      os << " " << i.startInProcessNames() << " ";
      ptr = &processNames_[i.startInProcessNames()];
      while (*ptr) {
        os << *ptr;
        ++ptr;
      }
      os << "\n";
    }
    os << "******* bigNamesContainer_ \n";
    for (auto i : bigNamesContainer_) {
      if (i == '\0')
        os << '\\' << '0';
      else
        os << i;
    }
    if (!bigNamesContainer_.empty())
      os << "\n";
    os << "******* processNames_ \n";
    for (auto i : processNames_) {
      if (i == '\0')
        os << '\\' << '0';
      else
        os << i;
    }
    if (!processNames_.empty())
      os << "\n";
    if (items_) {
      os << "******* items_ \n";
      for (auto const& item : *items_) {
        std::cout << item.kindOfType() << " " << item.moduleLabel() << " " << item.instance() << " " << item.process()
                  << " " << item.index() << " " << item.typeID() << "\n";
      }
    }
    if (processItems_) {
      os << "******* processItems_ \n";
      for (auto const& item : *processItems_) {
        os << item << "\n";
      }
    }
    os << "sortedTypeIDs_.size() = " << sortedTypeIDs_.size() << "\n";
    os << "indexAndNames_.size() = " << indexAndNames_.size() << "\n";
    os << "bigNamesContainer_.size() = " << bigNamesContainer_.size() << "\n";
    os << "processNames_.size() = " << processNames_.size() << "\n";
    os << "\n";
  }
}  // namespace edm
