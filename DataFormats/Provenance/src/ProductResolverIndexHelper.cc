
#include "DataFormats/Provenance/interface/ProductResolverIndexHelper.h"
#include "DataFormats/Provenance/interface/ViewTypeChecker.h"
#include "FWCore/Reflection/interface/DictionaryTools.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Reflection/interface/TypeWithDict.h"
#include "FWCore/Utilities/interface/WrappedClassName.h"
#include "FWCore/Utilities/interface/getAnyPtr.h"

#include <TClass.h>

#include <cassert>
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

    bool typeIsViewCompatible(TypeID const& requestedViewType,
                              TypeID const& wrappedtypeID,
                              std::string const& className) {
      auto elementType = getContainedTypeFromWrapper(wrappedtypeID, className);
      if (elementType == TypeID(typeid(void)) or elementType == TypeID()) {
        //the wrapped type is not a container
        return false;
      }
      if (elementType == requestedViewType) {
        return true;
      }
      //need to check for inheritance match
      std::vector<std::string> missingDictionaries;
      std::vector<TypeID> baseTypes;
      if (!public_base_classes(missingDictionaries, elementType, baseTypes)) {
        return false;
      }
      for (auto const& base : baseTypes) {
        if (TypeID(base.typeInfo()) == requestedViewType) {
          return true;
        }
      }
      return false;
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

    auto checkForSingleProcess = [this](unsigned int index) {
      //0 is for blank process name. If not zero, we have a match
      if (indexAndNames_[index].startInProcessNames() != 0U) {
        return index;
      }
      //Now check to see if only one process has this type/module/instance name
      // we need to skip the skipCurrentProcess entry
      auto nextIndex = index + 2;
      while (indexAndNames_.size() > nextIndex && indexAndNames_[nextIndex].startInProcessNames() != 0U) {
        ++nextIndex;
      }
      return (nextIndex == index + 3) ? index + 2 : index;
    };

    return indexAndNames_[checkForSingleProcess(iToIndexAndNames)].index();
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

  char const* ProductResolverIndexHelper::Matches::processName(unsigned int i) const {
    if (i >= numberOfMatches_) {
      throw Exception(errors::LogicError)
          << "ProductResolverIndexHelper::Matches::processName - Argument is out of range.\n";
    }
    unsigned int startInProcessNames =
        productResolverIndexHelper_->indexAndNames_[startInIndexAndNames_ + i].startInProcessNames();
    return &productResolverIndexHelper_->processNames_[startInProcessNames];
  }

  char const* ProductResolverIndexHelper::Matches::productInstanceName(unsigned int i) const {
    if (i >= numberOfMatches_) {
      throw Exception(errors::LogicError)
          << "ProductResolverIndexHelper::Matches::productInstanceName - Argument is out of range.\n";
    }
    unsigned int start =
        productResolverIndexHelper_->indexAndNames_[startInIndexAndNames_ + i].startInBigNamesContainer();
    auto moduleLabelSize = strlen(&productResolverIndexHelper_->bigNamesContainer_[start]);
    return &productResolverIndexHelper_->bigNamesContainer_[start + moduleLabelSize + 1];
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

  ProductResolverIndex ProductResolverIndexHelper::insert(TypeID const& typeID,
                                                          char const* moduleLabel,
                                                          char const* instance,
                                                          char const* process,
                                                          TypeID const& containedTypeID,
                                                          std::vector<TypeID>* baseTypesOfContainedType) {
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
    auto insertNoProcessCase = [](Item& item, std::set<Item>& container) {
      item.clearProcess();
      auto iter = container.find(item);
      if (iter == container.end()) {
        item.setIndex(ProductResolverIndexInitializing);
        container.insert(item);
        //add entry for skipCurrentProcess
        item.setSkipCurrentProcess();
        item.setIndex(ProductResolverIndexInitializing);
        container.insert(item);
      }
    };
    insertNoProcessCase(item, *items_);

    // Now put in entries for a contained class if this is a
    // recognized container.
    if (containedTypeID != TypeID(typeid(void)) && containedTypeID != TypeID()) {
      TypeWithDict containedType(containedTypeID.typeInfo());

      auto insertAndCheckForAmbiguous = [](Item& item, std::set<Item>& container) {
        auto iter = container.find(item);
        if (iter != container.end()) {
          item.setIndex(ProductResolverIndexAmbiguous);
          container.erase(iter);
        }
        container.insert(item);
      };

      Item containedItem(ELEMENT_TYPE, containedTypeID, moduleLabel, instance, process, savedProductIndex);
      insertAndCheckForAmbiguous(containedItem, *items_);
      insertNoProcessCase(containedItem, *items_);

      // Repeat this for all public base classes of the contained type
      if (baseTypesOfContainedType) {
        for (TypeID const& baseTypeID : *baseTypesOfContainedType) {
          Item baseItem(ELEMENT_TYPE, baseTypeID, moduleLabel, instance, process, savedProductIndex);
          insertAndCheckForAmbiguous(baseItem, *items_);
          insertNoProcessCase(baseItem, *items_);
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
    std::vector<TypeID> baseTypes;
    std::vector<TypeID>* baseTypesOfContainedType = &baseTypes;
    if (hasContainedType) {
      std::vector<std::string> missingDictionaries;
      public_base_classes(missingDictionaries, containedTypeID, baseTypes);
    }
    return insert(typeID, moduleLabel, instance, process, containedTypeID, baseTypesOfContainedType);
  }

  namespace {
    void setNoProcessIndices(std::vector<edm::ProductResolverIndexHelper::IndexAndNames>::iterator itBegin,
                             std::vector<edm::ProductResolverIndexHelper::IndexAndNames>::iterator itEnd,
                             std::vector<std::string> const& orderedProcessNames,
                             std::vector<char> const& processNames) {
      /*The order of the iterators should be
      0 : the empty process name case -> ''
      1 : the skip current process case -> '#'
      2+: the cases with a specific process name*/
      using IndexAndNames = edm::ProductResolverIndexHelper::IndexAndNames;
      assert(itEnd - itBegin > 2);
      const auto itNoProcess = itBegin;
      const auto itSkipCurrentProcess = itBegin + 1;
      const auto itFirstWithSetProcess = itBegin + 2;
      assert(itNoProcess->startInProcessNames() == 0U);
      assert(itSkipCurrentProcess->startInProcessNames() == 1U);
      assert(processNames[itSkipCurrentProcess->startInProcessNames()] == '#');
      assert(itNoProcess->index() == edm::ProductResolverIndexInitializing);
      assert(itSkipCurrentProcess->index() == edm::ProductResolverIndexInitializing);
      if (itEnd - itBegin == 3) {
        //only have one actual process
        *itNoProcess = IndexAndNames(itFirstWithSetProcess->index(),
                                     itNoProcess->startInBigNamesContainer(),
                                     itNoProcess->startInProcessNames());
        //Now handle skipCurrentProcess
        if (orderedProcessNames[0] == &processNames[itFirstWithSetProcess->startInProcessNames()]) {
          //the one process is the current process
          *itSkipCurrentProcess = IndexAndNames(ProductResolverIndexInvalid,
                                                itSkipCurrentProcess->startInBigNamesContainer(),
                                                itSkipCurrentProcess->startInProcessNames());
        } else {
          *itSkipCurrentProcess = IndexAndNames(itFirstWithSetProcess->index(),
                                                itSkipCurrentProcess->startInBigNamesContainer(),
                                                itSkipCurrentProcess->startInProcessNames());
        }
      } else {
        bool foundFirstMatch = false;
        for (auto const& proc : orderedProcessNames) {
          auto it = itFirstWithSetProcess;
          while (it != itEnd && proc != &processNames[it->startInProcessNames()]) {
            ++it;
          }
          if (it != itEnd) {
            if (not foundFirstMatch) {
              foundFirstMatch = true;
              //found a process that matches
              *itNoProcess = IndexAndNames(
                  it->index(), itNoProcess->startInBigNamesContainer(), itNoProcess->startInProcessNames());
              //Now handle skipCurrentProcess
              if (proc != orderedProcessNames[0]) {
                *itSkipCurrentProcess = IndexAndNames(it->index(),
                                                      itSkipCurrentProcess->startInBigNamesContainer(),
                                                      itSkipCurrentProcess->startInProcessNames());
                break;
              } else {
                //this process is the current process
                *itSkipCurrentProcess = IndexAndNames(ProductResolverIndexInvalid,
                                                      itSkipCurrentProcess->startInBigNamesContainer(),
                                                      itSkipCurrentProcess->startInProcessNames());
              }
            } else {
              *itSkipCurrentProcess = IndexAndNames(it->index(),
                                                    itSkipCurrentProcess->startInBigNamesContainer(),
                                                    itSkipCurrentProcess->startInProcessNames());
              break;
            }
          }
        }
      }
    }
  }  // namespace
  void ProductResolverIndexHelper::setFrozen(std::vector<std::string> const& orderedProcessNames) {
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

    //sanity check
    for (auto const& p : *processItems_) {
      if (p.empty() or p == skipCurrentProcessLabel())
        continue;
      if (orderedProcessNames.end() == std::find(orderedProcessNames.begin(), orderedProcessNames.end(), p)) {
        throw Exception(errors::LogicError)
            << "ProductResolverIndexHelper::setFrozen process not in ordered list " << p << std::endl;
      }
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
    unsigned int iNoProcessBegin = 0;
    if (!items_->empty()) {
      for (auto const& item : *items_) {
        if (iFirstType || item.typeID() != previousTypeID || item.kindOfType() != previousKindOfType) {
          iFirstThisType = true;
          sortedTypeIDs_.push_back(item.typeID());
          if (iFirstType) {
            iFirstType = false;
          } else {
            ranges_.push_back(Range(iBeginning, iCount));
          }
          iBeginning = iCount;
        }

        if (iFirstThisType || item.moduleLabel() != previousModuleLabel || item.instance() != previousInstance) {
          if (iNoProcessBegin != iCount) {
            setNoProcessIndices(indexAndNames_.begin() + iNoProcessBegin,
                                indexAndNames_.begin() + iCount,
                                orderedProcessNames,
                                processNames_);
            iNoProcessBegin = iCount;
          }
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
        ++iCount;
      }
      ranges_.push_back(Range(iBeginning, iCount));
      setNoProcessIndices(
          indexAndNames_.begin() + iNoProcessBegin, indexAndNames_.end(), orderedProcessNames, processNames_);
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
    std::string errorMessage;
    if (sortedTypeIDs_.size() != ranges_.size()) {
      sanityChecksPass = false;
      errorMessage += "sortedTypeIDs_.size() != ranges_.size()\n";
    }

    unsigned int previousEnd = 0;
    for (auto const& range : ranges_) {
      if (range.begin() != previousEnd) {
        sanityChecksPass = false;
        errorMessage += "ranges_ are not contiguous\n";
      }
      if (range.begin() >= range.end()) {
        sanityChecksPass = false;
        errorMessage += "ranges_ are not valid\n";
      }
      previousEnd = range.end();
    }
    if (previousEnd != indexAndNames_.size()) {
      sanityChecksPass = false;
      errorMessage += "ranges_ do not cover all of indexAndNames_\n";
    }

    unsigned maxStart = 0;
    unsigned maxStartProcess = 0;
    for (auto const& indexAndName : indexAndNames_) {
      if (indexAndName.index() >= nextIndexValue_ && (indexAndName.index() != ProductResolverIndexAmbiguous and
                                                      indexAndName.index() != ProductResolverIndexInvalid)) {
        sanityChecksPass = false;
        auto startOfModule = indexAndName.startInBigNamesContainer();
        auto startOfInstance = strlen(&bigNamesContainer_[startOfModule]) + 1 + startOfModule;
        errorMessage += "indexAndNames_ has invalid index" + std::to_string(indexAndName.index()) + " " +
                        &bigNamesContainer_[startOfModule] + " " + &bigNamesContainer_[startOfInstance] + " " +
                        &processNames_[indexAndName.startInProcessNames()] + "\n";
      }

      if (indexAndName.startInBigNamesContainer() >= bigNamesContainer_.size()) {
        sanityChecksPass = false;
        errorMessage += "indexAndNames_ has invalid startInBigNamesContainer\n";
      }
      if (indexAndName.startInProcessNames() >= processNames_.size()) {
        sanityChecksPass = false;
        errorMessage += "indexAndNames_ has invalid startInProcessNames\n";
      }

      if (indexAndName.startInBigNamesContainer() > maxStart)
        maxStart = indexAndName.startInBigNamesContainer();
      if (indexAndName.startInProcessNames() > maxStartProcess)
        maxStartProcess = indexAndName.startInProcessNames();
    }

    if (!indexAndNames_.empty()) {
      if (bigNamesContainer_.back() != '\0') {
        sanityChecksPass = false;
        errorMessage += "bigNamesContainer_ does not end with null char\n";
      }
      if (processNames_.back() != '\0') {
        sanityChecksPass = false;
        errorMessage += "processNames_ does not end with null char\n";
      }
      if (maxStart >= bigNamesContainer_.size()) {
        sanityChecksPass = false;
        errorMessage += "maxStart >= bigNamesContainer_.size()\n";
      }
      unsigned int countZeroes = 0;
      for (unsigned j = maxStart; j < bigNamesContainer_.size(); ++j) {
        if (bigNamesContainer_[j] == '\0') {
          ++countZeroes;
        }
      }
      if (countZeroes != 2) {
        sanityChecksPass = false;
        errorMessage += "bigNamesContainer_ does not have two null chars\n";
      }
      if (maxStartProcess >= processNames_.size()) {
        sanityChecksPass = false;
        errorMessage += "maxStartProcess >= processNames_.size()\n";
      }
      countZeroes = 0;
      for (unsigned j = maxStartProcess; j < processNames_.size(); ++j) {
        if (processNames_[j] == '\0') {
          ++countZeroes;
        }
      }
      if (countZeroes != 1) {
        sanityChecksPass = false;
        errorMessage += "processNames_ does not have one null char\n";
      }
    }

    if (!sanityChecksPass) {
      throw Exception(errors::LogicError) << "ProductResolverIndexHelper::setFrozen - Detected illegal state.\n"
                                          << errorMessage;
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
        os << item.kindOfType() << " " << item.moduleLabel() << " " << item.instance() << " " << item.process() << " "
           << item.index() << " " << item.typeID() << "\n";
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
