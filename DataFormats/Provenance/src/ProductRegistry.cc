/**
   \file
   class impl

   \Original author Stefano ARGIRO
   \Current author Bill Tanenbaum
   \date 19 Jul 2005
*/


#include "DataFormats/Provenance/interface/ProductRegistry.h"

#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/DictionaryTools.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"
#include "FWCore/Utilities/interface/WrappedClassName.h"

#include <algorithm>
#include <limits>
#include <sstream>

namespace edm {
  namespace {
    void checkDicts(BranchDescription const& productDesc) {
      if(productDesc.transient()) {
        checkDictionaries(productDesc.fullClassName(), true);
        checkDictionaries(wrappedClassName(productDesc.fullClassName()), true);
      } else {
        checkDictionaries(wrappedClassName(productDesc.fullClassName()), false);
      }
    }
  }

  ProductRegistry::ProductRegistry() :
      productList_(),
      transient_() {
  }

  ProductRegistry::Transients::Transients() :
      frozen_(false),
      constProductList_(),
      productProduced_(),
      anyProductProduced_(false),
      productLookup_(),
      elementLookup_(),
      branchIDToIndex_(),
      producedBranchListIndex_(std::numeric_limits<BranchListIndex>::max()),
      missingDictionaries_() {
    for(bool& isProduced : productProduced_) isProduced = false;
  }

  void
  ProductRegistry::Transients::reset() {
    frozen_ = false;
    constProductList_.clear();
    for(bool& isProduced : productProduced_) isProduced = false;
    anyProductProduced_ = false;
    productLookup_.reset();
    elementLookup_.reset();
    branchIDToIndex_.clear();
    producedBranchListIndex_ = std::numeric_limits<BranchListIndex>::max();
    missingDictionaries_.clear();
  }

  ProductRegistry::ProductRegistry(ProductList const& productList, bool toBeFrozen) :
      productList_(productList),
      transient_() {
    frozen() = toBeFrozen;
  }

  void
  ProductRegistry::addProduct(BranchDescription const& productDesc,
                              bool fromListener) {
    assert(productDesc.produced());
    throwIfFrozen();
    checkDicts(productDesc);
    std::pair<ProductList::iterator, bool> ret =
         productList_.insert(std::make_pair(BranchKey(productDesc), productDesc));
    if(!ret.second) {
      throw Exception(errors::Configuration, "Duplicate Process")
        << "The process name " << productDesc.processName() << " was previously used on these products.\n"
        << "Please modify the configuration file to use a distinct process name.\n";
    }
    addCalled(productDesc, fromListener);
  }

  void
  ProductRegistry::addLabelAlias(BranchDescription const& productDesc,
                                 std::string const& labelAlias,
                                 std::string const& instanceAlias) {
    assert(productDesc.produced());
    assert(productDesc.branchID().isValid());
    throwIfFrozen();
    BranchDescription bd(productDesc, labelAlias, instanceAlias);
    std::pair<ProductList::iterator, bool> ret =
         productList_.insert(std::make_pair(BranchKey(bd), bd));
    assert(ret.second);
    addCalled(bd, false);
  }

  void
  ProductRegistry::copyProduct(BranchDescription const& productDesc) {
    assert(!productDesc.produced());
    throwIfFrozen();
    productDesc.init();
    BranchKey k = BranchKey(productDesc);
    ProductList::iterator iter = productList_.find(k);
    if(iter == productList_.end()) {
      productList_.insert(std::make_pair(k, productDesc));
    } else {
      assert(combinable(iter->second, productDesc));
      iter->second.merge(productDesc);
    }
  }

  bool
  ProductRegistry::anyProducts(BranchType brType) const {
    throwIfNotFrozen();
    for(ProductList::const_iterator it = productList_.begin(), itEnd = productList_.end();
        it != itEnd; ++it) {
      if(it->second.branchType() == brType) {
        return true;
      }
    }
    return false;
  }

  void
  ProductRegistry::setFrozen(bool initializeLookupInfo) const {
    if(frozen()) return;
    frozen() = true;
    if(initializeLookupInfo) {
      initializeLookupTables();
    }
  }

  void
  ProductRegistry::throwIfFrozen() const {
    if(frozen()) {
      throw cms::Exception("ProductRegistry", "throwIfFrozen")
        << "cannot modify the ProductRegistry because it is frozen\n";
    }
  }

  void
  ProductRegistry::throwIfNotFrozen() const {
    if(!frozen()) {
      throw cms::Exception("ProductRegistry", "throwIfNotFrozen")
        << "cannot read the ProductRegistry because it is not yet frozen\n";
    }
  }

  void
  ProductRegistry::addCalled(BranchDescription const&, bool) {
  }

  std::vector<std::string>
  ProductRegistry::allBranchNames() const {
    std::vector<std::string> result;
    result.reserve(productList().size());

    for(auto const& product : productList()) {
      result.push_back(product.second.branchName());
    }
    return result;
  }

  std::vector<BranchDescription const*>
  ProductRegistry::allBranchDescriptions() const {
    std::vector<BranchDescription const*> result;
    result.reserve(productList().size());

    for(auto const& product : productList()) {
      result.push_back(&product.second);
    }
    return result;
  }

  void
  ProductRegistry::updateFromInput(ProductList const& other) {
    for(auto const& product : other) {
      copyProduct(product.second);
    }
  }

  void
  ProductRegistry::updateFromInput(std::vector<BranchDescription> const& other) {
    for(BranchDescription const& branchDescription : other) {
      copyProduct(branchDescription);
    }
  }

  std::string
  ProductRegistry::merge(ProductRegistry const& other,
        std::string const& fileName,
        BranchDescription::MatchMode parametersMustMatch,
        BranchDescription::MatchMode branchesMustMatch) {
    std::ostringstream differences;

    ProductRegistry::ProductList::iterator j = productList_.begin();
    ProductRegistry::ProductList::iterator s = productList_.end();
    ProductRegistry::ProductList::const_iterator i = other.productList().begin();
    ProductRegistry::ProductList::const_iterator e = other.productList().end();

    // Loop over entries in the main product registry.
    while(j != s || i != e) {
      if(j != s && j->second.produced()) {
        // Ignore branches just produced (i.e. not in input file).
        ++j;
      } else if(j == s || (i != e && i->first < j->first)) {
        if(i->second.present()) {
          differences << "Branch '" << i->second.branchName() << "' is in file '" << fileName << "'\n";
          differences << "    but not in previous files.\n";
        } else {
          productList_.insert(*i);
        }
        ++i;
      } else if(i == e || (j != s && j->first < i->first)) {
        if(j->second.present() && branchesMustMatch == BranchDescription::Strict) {
          differences << "Branch '" << j->second.branchName() << "' is in previous files\n";
          differences << "    but not in file '" << fileName << "'.\n";
        }
        ++j;
      } else {
        std::string difs = match(j->second, i->second, fileName, parametersMustMatch);
        if(difs.empty()) {
          if(parametersMustMatch == BranchDescription::Permissive) j->second.merge(i->second);
        } else {
          differences << difs;
        }
        ++i;
        ++j;
      }
    }
    initializeLookupTables();
    return differences.str();
  }

  static void
  fillLookup(TypeID const& type,
             ProductTransientIndex const& index,
             ConstBranchDescription const* branchDesc,
             TransientProductLookupMap::FillFromMap& oMap) {
    oMap[std::make_pair(TypeInBranchType(type,
                                         branchDesc->branchType()),
                                         branchDesc)] = index;
  }

  void ProductRegistry::initializeLookupTables() const {
    StringSet savedMissingTypes;
    savedMissingTypes.swap(missingTypes());
    StringSet savedFoundTypes;
    savedFoundTypes.swap(foundTypes());
    constProductList().clear();
    transient_.branchIDToIndex_.clear();
    ProductTransientIndex index = 0;

    //NOTE it might be possible to remove the need for this temporary map because the productList is ordered by the
    // BranchKey and for the same C++ class type the BranchKey will sort just like CompareTypeInBranchTypeConstBranchDescription
    typedef TransientProductLookupMap::FillFromMap TempLookupMap;
    TempLookupMap tempProductLookupMap;
    TempLookupMap tempElementLookupMap;

    StringSet usedProcessNames;
    StringSet missingDicts;
    for(auto const& product : productList_) {
      auto const& key = product.first;
      auto const& desc = product.second;
      if(desc.produced()) {
        setProductProduced(desc.branchType());
      }

      //insert returns a pair<iterator, bool> and we want the address of the ConstBranchDescription that was created in the map
      // this is safe since items in a map always retain their memory address
      ConstBranchDescription const* pBD = &(constProductList().insert(std::make_pair(key, ConstBranchDescription(desc))).first->second);

      transient_.branchIDToIndex_[desc.branchID()] = index;

      usedProcessNames.insert(pBD->processName());

      //only do the following if the data is supposed to be available in the event
      if(desc.present()) {
        TypeID type(TypeID::byName(desc.className()));
        TypeID wrappedType(TypeID::byName(wrappedClassName(desc.className())));
        if(!bool(type) || !bool(wrappedType)) {
          missingDicts.insert(desc.className());
        } else {
          fillLookup(type, index, pBD, tempProductLookupMap);
  
          if(bool(type)) {
            // Here we look in the object named "type" for a typedef
            // named "value_type" and get the type for it.
            // Then check to ensure the dictionary is defined
            // for this value_type.
            // I do not throw an exception here if the check fails
            // because there are known cases where the dictionary does
            // not exist and we do not need to support those cases.
            TypeID valueType;
            TypeWithDict typeWithDict(type.typeInfo());
            if((is_RefVector(typeWithDict, valueType) ||
                is_PtrVector(typeWithDict, valueType) ||
                is_RefToBaseVector(typeWithDict, valueType) ||
                value_type_of(typeWithDict, valueType))
                && bool(valueType)) {
  
              fillLookup(valueType, index, pBD, tempElementLookupMap);
  
              // Repeat this for all public base classes of the value_type
              std::vector<TypeID> baseTypes;
              public_base_classes(valueType, baseTypes);
  
              for(TypeID const& baseType : baseTypes) {
                fillLookup(baseType, index, pBD, tempElementLookupMap);
              }
            }
          }
        }
      }
      ++index;
    }
    missingDictionaries().reserve(missingDicts.size());
    copy_all(missingDicts, std::back_inserter(missingDictionaries()));
    productLookup().fillFrom(tempProductLookupMap);
    elementLookup().fillFrom(tempElementLookupMap);
    savedMissingTypes.swap(missingTypes());
    savedFoundTypes.swap(foundTypes());
  }

  ProductTransientIndex ProductRegistry::indexFrom(BranchID const& iID) const {
    std::map<BranchID, ProductTransientIndex>::iterator itFind = transient_.branchIDToIndex_.find(iID);
    if(itFind == transient_.branchIDToIndex_.end()) {
      return kInvalidIndex;
    }
    return itFind->second;
  }

  void ProductRegistry::print(std::ostream& os) const {
    for(auto const& product: productList_) {
      os << product.second << "\n-----\n";
    }
  }
}
