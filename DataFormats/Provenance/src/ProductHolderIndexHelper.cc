
#include "DataFormats/Provenance/interface/ProductHolderIndexHelper.h"
#include "FWCore/Utilities/interface/DictionaryTools.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"

#include <iostream>
#include <limits>

namespace edm {

  ProductHolderIndexHelper::ProductHolderIndexHelper() :
    nextIndexValue_(0),
    beginElements_(0),
    items_(new std::set<ProductHolderIndexHelper::Item>),
    processItems_(new std::set<std::string>) {
  }

  ProductHolderIndex
  ProductHolderIndexHelper::index(KindOfType kindOfType,
                                  TypeID const& typeID,
                                  char const* moduleLabel,
                                  char const* instance,
                                  char const* process) const {

    unsigned int iToIndexAndNames = indexToIndexAndNames(kindOfType,
                                                         typeID,
                                                         moduleLabel,
                                                         instance,
                                                         process);

    if (iToIndexAndNames == std::numeric_limits<unsigned int>::max()) {
      return ProductHolderIndexInvalid;
    }
    return indexAndNames_[iToIndexAndNames].index();
  }

  ProductHolderIndexHelper::Matches::Matches(ProductHolderIndexHelper const* productHolderIndexHelper,
                                             unsigned int startInIndexAndNames,
                                             unsigned int numberOfMatches) :
    productHolderIndexHelper_(productHolderIndexHelper),
    startInIndexAndNames_(startInIndexAndNames),
    numberOfMatches_(numberOfMatches) {
    if (numberOfMatches != 0 && startInIndexAndNames_ + numberOfMatches_ > productHolderIndexHelper_->indexAndNames_.size()) {
      throw Exception(errors::LogicError)
        << "ProductHolderIndexHelper::Matches::Matches - Arguments exceed vector bounds.\n";
    }
  }

  ProductHolderIndex
  ProductHolderIndexHelper::Matches::index(unsigned int i) const {
    if (i >= numberOfMatches_) {
      throw Exception(errors::LogicError)
        << "ProductHolderIndexHelper::Matches::index - Argument is out of range.\n";
    }
    return productHolderIndexHelper_->indexAndNames_[startInIndexAndNames_ + i].index();
  }

  bool
  ProductHolderIndexHelper::Matches::isFullyResolved(unsigned int i) const {
    if (i >= numberOfMatches_) {
      throw Exception(errors::LogicError)
        << "ProductHolderIndexHelper::Matches::isFullyResolved - Argument is out of range.\n";
    }
    return (productHolderIndexHelper_->indexAndNames_[startInIndexAndNames_ + i].startInProcessNames() != 0U);
  }

  ProductHolderIndexHelper::Matches
  ProductHolderIndexHelper::relatedIndexes(KindOfType kindOfType,
                                           TypeID const& typeID,
                                           char const* moduleLabel,
                                           char const* instance) const {

    unsigned int startInIndexAndNames = indexToIndexAndNames(kindOfType,
                                                             typeID,
                                                             moduleLabel,
                                                             instance,
                                                             0);
    unsigned int numberOfMatches = 1;

    if (startInIndexAndNames == std::numeric_limits<unsigned int>::max()) {
      numberOfMatches = 0;
    } else {
      auto vSize = indexAndNames_.size();
      for (unsigned int j = startInIndexAndNames + 1U;
           j < vSize && (indexAndNames_[j].startInProcessNames() != 0U);
           ++j) {
        ++numberOfMatches;
      }
    }
    return Matches(this, startInIndexAndNames, numberOfMatches);
  }

  ProductHolderIndexHelper::Matches
  ProductHolderIndexHelper::relatedIndexes(KindOfType kindOfType,
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

  ProductHolderIndex
  ProductHolderIndexHelper::insert(TypeWithDict const& typeWithDict,
                                   char const* moduleLabel,
                                   char const* instance,
                                   char const* process) {
    if (!items_) {
      throw Exception(errors::LogicError)
        << "ProductHolderIndexHelper::insert - Attempt to insert more elements after frozen.\n";
    }

    if (process == 0 || *process == '\0') {
      throw Exception(errors::LogicError)
        << "ProductHolderIndexHelper::insert - Empty process.\n";
    }

    TypeID typeID(typeWithDict.typeInfo());

    // Throw if this has already been inserted
    Item item(PRODUCT_TYPE, typeID, moduleLabel, instance, process, 0);
    std::set<Item>::iterator iter = items_->find(item);
    if (iter != items_->end()) {
      throw Exception(errors::LogicError)
        << "ProductHolderIndexHelper::insert - Attempt to insert duplicate entry.\n";
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
    TypeWithDict containedType;
    if((is_RefVector(typeWithDict, containedType) ||
        is_PtrVector(typeWithDict, containedType) ||
        is_RefToBaseVector(typeWithDict, containedType) ||
        value_type_of(typeWithDict, containedType))
        && bool(containedType)) {

      TypeID containedTypeID(containedType.typeInfo());
      Item containedItem(ELEMENT_TYPE, containedTypeID, moduleLabel, instance, process, savedProductIndex);
      iter = items_->find(containedItem);
      if (iter != items_->end()) {
        containedItem.setIndex(ProductHolderIndexAmbiguous);
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
      std::vector<TypeWithDict> baseTypes;
      public_base_classes(containedType, baseTypes);

      for(TypeWithDict const& baseType : baseTypes) {

        TypeID baseTypeID(baseType.typeInfo());
        Item baseItem(ELEMENT_TYPE, baseTypeID, moduleLabel, instance, process, savedProductIndex);
        iter = items_->find(baseItem);
        if (iter != items_->end()) {
          baseItem.setIndex(ProductHolderIndexAmbiguous);
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
    return savedProductIndex;
  }

  void ProductHolderIndexHelper::setFrozen() {

    if (!items_) return;

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

      if (iFirstThisType ||
          item.moduleLabel() != previousModuleLabel ||
          item.instance() != previousInstance) {
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

        if (iFirstThisType ||
            item.moduleLabel() != previousModuleLabel ||
            item.instance() != previousInstance) {

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
            << "ProductHolderIndexHelper::setFrozen - Process not found in processNames_.\n";
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
    items_.reset();
    processItems_.reset();
  }

  std::vector<std::string> const& ProductHolderIndexHelper::lookupProcessNames() const {
    if (items_) {
      throw Exception(errors::LogicError)
        << "ProductHolderIndexHelper::lookupProcessNames - Attempt to access names before frozen.\n";
    }
    return lookupProcessNames_;
  }

  unsigned int
  ProductHolderIndexHelper::indexToIndexAndNames(KindOfType kindOfType,
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

      ProductHolderIndexHelper::Range const& range = ranges_[iType];
      unsigned int begin = range.begin();
      unsigned int end = range.end();

      while (begin < end) {

        unsigned int midpoint = begin + ((end - begin) / 2);
        char const* namePtr = &bigNamesContainer_[indexAndNames_[midpoint].startInBigNamesContainer()];

        // Compare the module label
        char const* label = moduleLabel;
        while (*namePtr && (*namePtr == *label)) {
          ++namePtr; ++label;
        }
        if (*namePtr == *label) { // true only if both are at the '\0' at the end of the C string
          ++namePtr;              // move to the next C string

          // Compare the instance name
          char const* instanceName = instance;
          while (*namePtr && (*namePtr == *instanceName)) {
            ++namePtr; ++instanceName;
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
                  if (indexAndNames_[midpoint].startInProcessNames() == 0) break;
                }
              } else {
                while (true) {
                  ++midpoint;
                  if (midpoint == indexAndNames_.size()) break;
                  if (indexAndNames_[midpoint].startInProcessNames() == 0) break;
                  if (indexAndNames_[midpoint].startInProcessNames() == startProcess) {
                    return midpoint;
                  }
                }                
              }
              break;
            }
          } else if (*namePtr < *instanceName) {
            if (begin == midpoint) break;
            begin = midpoint;
          } else {
            end = midpoint;
          }
        } else if (*namePtr < *label) {
          if (begin == midpoint) break;
          begin = midpoint;
        } else {
          end = midpoint;
        }
      } // end while (begin < end)
    }
    return std::numeric_limits<unsigned int>::max();
  }

  unsigned int
  ProductHolderIndexHelper::indexToType(KindOfType kindOfType,
                                        TypeID const& typeID) const {

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
        if (beginType == midpointType) break;
        beginType = midpointType;
      } else {
        endType = midpointType;
      }
    }
    return std::numeric_limits<unsigned int>::max(); // Failed to find it
  }

  unsigned int ProductHolderIndexHelper::processIndex(char const* process)  const {

    char const* ptr = &processNames_[0];
    char const* begin = ptr;
    while (true) {
      char const* p = process;
      char const* beginName = ptr;
      while (*ptr && (*ptr == *p)) {
        ++ptr; ++p;
      }
      if (*ptr == *p) {
        return beginName - begin;
      }
      while (*ptr) {
        ++ptr;
      }
      ++ptr;
      if (static_cast<unsigned>(ptr - begin) >=  processNames_.size()) {
        return std::numeric_limits<unsigned int>::max();
      }
    }
    return 0;
  }

  void ProductHolderIndexHelper::sanityCheck() const {
    bool sanityChecksPass = true;
    if (sortedTypeIDs_.size() != ranges_.size()) sanityChecksPass = false;

    unsigned int previousEnd = 0;
    for ( auto const& range : ranges_) {
      if (range.begin() != previousEnd) sanityChecksPass = false;
      if (range.begin() >= range.end()) sanityChecksPass = false;
      previousEnd = range.end();
    }
    if (previousEnd != indexAndNames_.size()) sanityChecksPass = false;


    unsigned maxStart = 0;
    unsigned maxStartProcess = 0;
    for (auto const& indexAndName : indexAndNames_) {
      if (indexAndName.index() >= nextIndexValue_ && indexAndName.index() != ProductHolderIndexAmbiguous) sanityChecksPass = false;

      if (indexAndName.startInBigNamesContainer() >= bigNamesContainer_.size()) sanityChecksPass = false;
      if (indexAndName.startInProcessNames() >= processNames_.size()) sanityChecksPass = false;

      if (indexAndName.startInBigNamesContainer() > maxStart) maxStart = indexAndName.startInBigNamesContainer();
      if (indexAndName.startInProcessNames() > maxStartProcess) maxStartProcess = indexAndName.startInProcessNames();
    }

    if (!indexAndNames_.empty()) {
      if (bigNamesContainer_.back() != '\0')  sanityChecksPass = false;
      if (processNames_.back() != '\0')  sanityChecksPass = false;
      if (maxStart >= bigNamesContainer_.size()) sanityChecksPass = false;
      unsigned int countZeroes = 0;
      for (unsigned j = maxStart; j < bigNamesContainer_.size(); ++j) {
        if (bigNamesContainer_[j] == '\0') {
          ++countZeroes;
        }
      }
      if (countZeroes != 2) sanityChecksPass = false;
      if (maxStartProcess >= processNames_.size()) sanityChecksPass = false;
      countZeroes = 0;
      for (unsigned j = maxStartProcess; j < processNames_.size(); ++j) {
        if (processNames_[j] == '\0') {
          ++countZeroes;
        }        
      }
      if (countZeroes != 1) sanityChecksPass = false;
    }

    if (!sanityChecksPass) {
      throw Exception(errors::LogicError)
        << "ProductHolderIndexHelper::setFrozen - Detected illegal state.\n";
    }
  }

  ProductHolderIndexHelper::Item::Item(KindOfType kindOfType,
                                       TypeID const& typeID,
                                       std::string const& moduleLabel,
                                       std::string const& instance,
                                       std::string const& process,
                                       ProductHolderIndex index) :
    kindOfType_(kindOfType),
    typeID_(typeID),
    moduleLabel_(moduleLabel),
    instance_(instance),
    process_(process),
    index_(index) {
  }

  bool
  ProductHolderIndexHelper::Item::operator<(Item const& right) const {
    if (kindOfType_ < right.kindOfType_) return true;
    if (kindOfType_ > right.kindOfType_) return false;
    if (typeID_ < right.typeID_) return true;
    if (typeID_ > right.typeID_) return false;
    if (moduleLabel_ < right.moduleLabel_) return true;
    if (moduleLabel_ > right.moduleLabel_) return false;
    if (instance_ < right.instance_) return true;
    if (instance_ > right.instance_) return false;
    return process_ < right.process_;
  }

  void ProductHolderIndexHelper::print(std::ostream& os) const {

    os << "\n******* Dump ProductHolderIndexHelper *************************\n";

    os << "\nnextIndexValue_ = " <<  nextIndexValue_ << "\n";
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
      if (i == '\0') os << '\\' << '0';
      else os << i;
    }
    if (!bigNamesContainer_.empty()) os << "\n";
    os << "******* processNames_ \n";
    for (auto i : processNames_) {
      if (i == '\0') os << '\\' << '0';
      else os << i;
    }
    if (!processNames_.empty()) os << "\n";
    if (items_) {
      os << "******* items_ \n";
      for (auto const& item : *items_) {
        std:: cout << item.kindOfType() << " " << item.moduleLabel() << " " << item.instance() << " " << item.process() << " " << item.index() << " " << item.typeID() << "\n";
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
}
