/**
   \file
   class impl

   \Original author Stefano ARGIRO
   \Current author Bill Tanenbaum
   \date 19 Jul 2005
*/


#include "DataFormats/Provenance/interface/ProductRegistry.h"

#include "DataFormats/Provenance/interface/ProductResolverIndexHelper.h"

#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/DictionaryTools.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"
#include "FWCore/Utilities/interface/WrappedClassName.h"

#include <cassert>
#include <iterator>
#include <limits>
#include <sstream>
#include <ostream>

namespace edm {

  ProductRegistry::ProductRegistry() :
      productList_(),
      transient_() {
  }

  ProductRegistry::Transients::Transients() :
      frozen_(false),
      productProduced_(),
      anyProductProduced_(false),
      eventProductLookup_(new ProductResolverIndexHelper),
      lumiProductLookup_(new ProductResolverIndexHelper),
      runProductLookup_(new ProductResolverIndexHelper),
      eventNextIndexValue_(0),
      lumiNextIndexValue_(0),
      runNextIndexValue_(0),

      branchIDToIndex_(),
      missingDictionaries_() {
    for(bool& isProduced : productProduced_) isProduced = false;
  }

  void
  ProductRegistry::Transients::reset() {
    frozen_ = false;
    for(bool& isProduced : productProduced_) isProduced = false;
    anyProductProduced_ = false;

    // propagate_const<T> has no reset() function
    eventProductLookup_ = std::make_unique<ProductResolverIndexHelper>();
    lumiProductLookup_ = std::make_unique<ProductResolverIndexHelper>();
    runProductLookup_ = std::make_unique<ProductResolverIndexHelper>();

    eventNextIndexValue_ = 0;
    lumiNextIndexValue_ = 0;
    runNextIndexValue_ = 0;

    branchIDToIndex_.clear();
    missingDictionaries_.clear();
  }

  ProductRegistry::ProductRegistry(ProductList const& productList, bool toBeFrozen) :
      productList_(productList),
      transient_() {
    freezeIt(toBeFrozen);
  }

  void
  ProductRegistry::addProduct(BranchDescription const& productDesc,
                              bool fromListener) {
    assert(productDesc.produced());
    throwIfFrozen();
    std::pair<ProductList::iterator, bool> ret =
         productList_.insert(std::make_pair(BranchKey(productDesc), productDesc));
    if(!ret.second) {
      auto const& previous = *productList_.find(BranchKey(productDesc));
      if(previous.second.produced()) {
        // Duplicate registration in current process
        throw Exception(errors::LogicError , "Duplicate Product")
          << "The produced product " << previous.second << " is registered more than once.\n"
          << "Please remove the redundant 'produces' call(s).\n";
      } else {
        // Duplicate registration in previous process
        throw Exception(errors::Configuration, "Duplicate Process")
          << "The process name " << productDesc.processName() << " was previously used on these products.\n"
          << "Please modify the configuration file to use a distinct process name.\n";
      }
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
    transient_.aliasToOriginal_.emplace_back(labelAlias,
                                             productDesc.moduleLabel());
    addCalled(bd, false);
  }

  void
  ProductRegistry::copyProduct(BranchDescription const& productDesc) {
    assert(!productDesc.produced());
    throwIfFrozen();
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

  std::shared_ptr<ProductResolverIndexHelper const>
  ProductRegistry::productLookup(BranchType branchType) const {
    if (branchType == InEvent) return transient_.eventProductLookup();
    if (branchType == InLumi) return transient_.lumiProductLookup();
    return transient_.runProductLookup();
  }

  std::shared_ptr<ProductResolverIndexHelper>
  ProductRegistry::productLookup(BranchType branchType) {
    if (branchType == InEvent) return transient_.eventProductLookup();
    if (branchType == InLumi) return transient_.lumiProductLookup();
    return transient_.runProductLookup();
  }

  void
  ProductRegistry::setFrozen(bool initializeLookupInfo) {
    if(frozen()) return;
    freezeIt();
    if(initializeLookupInfo) {
      initializeLookupTables(nullptr);
    }
    sort_all(transient_.aliasToOriginal_);
  }

  void
  ProductRegistry::setFrozen(std::set<TypeID> const& typesConsumed) {
    if(frozen()) return;
    freezeIt();
    initializeLookupTables(&typesConsumed);
    sort_all(transient_.aliasToOriginal_);
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
          transient_.branchIDToIndex_[i->second.branchID()] = getNextIndexValue(i->second.branchType());
          ++nextIndexValue(i->second.branchType());
        }
        ++i;
      } else if(i == e || (j != s && j->first < i->first)) {
        if(j->second.present() && branchesMustMatch == BranchDescription::Strict) {
          differences << "Branch '" << j->second.branchName() << "' is in previous files\n";
          differences << "    but not in file '" << fileName << "'.\n";
        }
        ++j;
      } else {
        std::string difs = match(j->second, i->second, fileName);
        if(difs.empty()) {
          j->second.merge(i->second);
        } else {
          differences << difs;
        }
        ++i;
        ++j;
      }
    }
    return differences.str();
  }

  void ProductRegistry::initializeLookupTables(std::set<TypeID> const* typesConsumed) {

    std::map<TypeID, TypeID> containedTypeMap;
    TypeSet missingDicts;

    transient_.branchIDToIndex_.clear();

    for(auto const& product : productList_) {
      auto const& desc = product.second;

      if(desc.produced()) {
        setProductProduced(desc.branchType());
      }

      //only do the following if the data is supposed to be available in the event
      if(desc.present()) {
        if(!bool(desc.unwrappedType())) {
          missingDicts.insert(TypeID(desc.unwrappedType().typeInfo()));
        } else if(!bool(desc.wrappedType())) {
          missingDicts.insert(TypeID(desc.wrappedType().typeInfo()));
        } else {
          TypeID wrappedTypeID(desc.wrappedType().typeInfo());
          TypeID typeID(desc.unwrappedType().typeInfo());
          TypeID containedTypeID;
          auto const& iter = containedTypeMap.find(typeID);
          if(iter != containedTypeMap.end()) {
             containedTypeID = iter->second;
          } else {
             containedTypeID = productholderindexhelper::getContainedTypeFromWrapper(wrappedTypeID, typeID.className());
             containedTypeMap.emplace(typeID, containedTypeID);
          }
          if(typesConsumed != nullptr && !desc.produced()) {
            bool mainTypeConsumed = (typesConsumed->find(typeID) != typesConsumed->end());
            bool hasContainedType = (containedTypeID != TypeID(typeid(void)) && containedTypeID != TypeID());
            bool containedTypeConsumed = hasContainedType && (typesConsumed->find(containedTypeID) != typesConsumed->end());
            if(hasContainedType && !containedTypeConsumed) {
              TypeWithDict containedType(containedTypeID.typeInfo());
              std::vector<TypeWithDict> baseTypes;
              public_base_classes(containedType, baseTypes);
              for(TypeWithDict const& baseType : baseTypes) {
                 if(typesConsumed->find(TypeID(baseType.typeInfo())) != typesConsumed->end()) {
                   containedTypeConsumed = true;
                   break;
                 }
              }
            }
            if(!containedTypeConsumed) {
              if(mainTypeConsumed) {
                // The main type is consumed, but either
                // there is no contained type, or if there is,
                // neither it nor any of its base classes are consumed.
                // Set the contained type, if there is one, to void,
                if(hasContainedType) {
		  containedTypeID = TypeID(typeid(void)); 
                }
              } else {
                // The main type is not consumed, and either
                // there is no contained type, or if there is,
                // neither it nor any of its base classes are consumed.
                // Don't insert anything in the lookup tables.
                continue;
              } 
            }
          }
          ProductResolverIndex index =
            productLookup(desc.branchType())->insert(typeID,
                                                     desc.moduleLabel().c_str(),
                                                     desc.productInstanceName().c_str(),
                                                     desc.processName().c_str(),
                                                     containedTypeID);

          transient_.branchIDToIndex_[desc.branchID()] = index;
        }
      }
    }
    productLookup(InEvent)->setFrozen();
    productLookup(InLumi)->setFrozen();
    productLookup(InRun)->setFrozen();

    transient_.eventNextIndexValue_ = productLookup(InEvent)->nextIndexValue();
    transient_.lumiNextIndexValue_ = productLookup(InLumi)->nextIndexValue();
    transient_.runNextIndexValue_ = productLookup(InRun)->nextIndexValue();

    for(auto const& product : productList_) {
      auto const& desc = product.second;
      if (transient_.branchIDToIndex_.find(desc.branchID()) == transient_.branchIDToIndex_.end()) {
        transient_.branchIDToIndex_[desc.branchID()] = getNextIndexValue(desc.branchType());
        ++nextIndexValue(desc.branchType());
      }
    }

    missingDictionariesForUpdate().reserve(missingDicts.size());
    copy_all(missingDicts, std::back_inserter(missingDictionariesForUpdate()));
  }

  ProductResolverIndex ProductRegistry::indexFrom(BranchID const& iID) const {
    std::map<BranchID, ProductResolverIndex>::const_iterator itFind = transient_.branchIDToIndex_.find(iID);
    if(itFind == transient_.branchIDToIndex_.end()) {
      return ProductResolverIndexInvalid;
    }
    return itFind->second;
  }

  void ProductRegistry::print(std::ostream& os) const {
    for(auto const& product: productList_) {
      os << product.second << "\n-----\n";
    }
  }

  ProductResolverIndex const&
  ProductRegistry::getNextIndexValue(BranchType branchType) const {
    if (branchType == InEvent) return transient_.eventNextIndexValue_;
    if (branchType == InLumi) return  transient_.lumiNextIndexValue_;
    return transient_.runNextIndexValue_;
  }

  ProductResolverIndex&
  ProductRegistry::nextIndexValue(BranchType branchType) {
    if (branchType == InEvent) return transient_.eventNextIndexValue_;
    if (branchType == InLumi) return  transient_.lumiNextIndexValue_;
    return transient_.runNextIndexValue_;
  }
}
