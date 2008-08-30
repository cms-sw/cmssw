/**
   \file
   class impl

   \Original author Stefano ARGIRO
   \Current author Bill Tanenbaum
   \date 19 Jul 2005
*/


#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "FWCore/Utilities/interface/ReflexTools.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include <algorithm>
#include <sstream>

namespace edm {

  ProductRegistry::ProductRegistry() :
      productList_(),
      nextID_(1),
      frozen_(false),
      constProductList_(),
      productLookup_(),
      elementLookup_() {
  }

  ProductRegistry::ProductRegistry(ProductList const& productList, unsigned int nextID) :
      productList_(productList),
      nextID_(nextID),
      frozen_(true),
      constProductList_(),
      productLookup_(),
      elementLookup_() {
  }

  void
  ProductRegistry::addProduct(BranchDescription const& productDesc,
			      bool fromListener) {
    assert(productDesc.produced());
    throwIfFrozen();
    productDesc.init();
    checkDictionaries(productDesc.fullClassName(), productDesc.transient());
    productList_.insert(std::make_pair(BranchKey(productDesc), productDesc));
    addCalled(productDesc,fromListener);
  }
 
  void
  ProductRegistry::copyProduct(BranchDescription const& productDesc) {
    assert(!productDesc.produced());
    throwIfFrozen();
    productDesc.init();
    BranchKey k = BranchKey(productDesc);
    ProductList::iterator iter = productList_.find(k);
    if (iter == productList_.end()) {
      productList_.insert(std::make_pair(k, productDesc));
    } else {
      assert(combinable(iter->second, productDesc));
      iter->second.merge(productDesc);
    }
  }
  
  void
  ProductRegistry::setProductIDs(unsigned int startingID) {
    throwIfNotFrozen();
    if (nextID_ <= productList_.size()) {
      nextID_ = productList_.size() + 1 ;
    }
    if (startingID < nextID_) {
      startingID = nextID_;
    }
    --startingID;
    for (ProductList::iterator it = productList_.begin(), itEnd = productList_.end();
        it != itEnd; ++it) {
      if (it->second.produced() && it->second.branchType() == InEvent) {
        it->second.setProductIDtoAssign(ProductID(++startingID));
      }
    }
    setNextID(startingID + 1);
    initializeTransients();
  }

  bool
  ProductRegistry::anyProducts(BranchType brType) const {
    throwIfNotFrozen();
    for (ProductList::const_iterator it = productList_.begin(), itEnd = productList_.end();
        it != itEnd; ++it) {
      if (it->second.branchType() == brType) {
	return true;
      }
    }
    return false;
  }
  
  void
  ProductRegistry::setFrozen() const {
    checkAllDictionaries();
    if(frozen_) return;
    frozen_ = true;
    initializeTransients();
  }
  
  void
  ProductRegistry::throwIfFrozen() const {
    if (frozen_) {
      throw cms::Exception("ProductRegistry", "throwIfFrozen")
            << "cannot modify the ProductRegistry because it is frozen\n";
    }
  }
  
  void
  ProductRegistry::throwIfNotFrozen() const {
    if (!frozen_) {
      throw cms::Exception("ProductRegistry", "throwIfNotFrozen")
            << "cannot read the ProductRegistry because it is not yet frozen\n";
    }
  }
  
  void
  ProductRegistry::addCalled(BranchDescription const&, bool) {
  }

  std::vector<std::string>
  ProductRegistry::allBranchNames() const
  {
    std::vector<std::string> result;
    result.reserve( productList().size() ); 

    ProductList::const_iterator it  = productList().begin();
    ProductList::const_iterator end = productList().end();

    for ( ; it != end; ++it ) result.push_back(it->second.branchName());

  return result;
  }

  std::vector<BranchDescription const*> 
  ProductRegistry::allBranchDescriptions() const
  {
    std::vector<BranchDescription const*> result;
    result.reserve( productList().size() );

    ProductList::const_iterator it  = productList().begin();
    ProductList::const_iterator end = productList().end();
    
    for ( ; it != end; ++it) result.push_back(&(it->second));    
    return result;
  }
  
  std::string
  ProductRegistry::merge(ProductRegistry const& other,
	std::string const& fileName,
	BranchDescription::MatchMode m) {
    std::ostringstream differences;

    ProductRegistry::ProductList::iterator j = productList_.begin();
    ProductRegistry::ProductList::iterator s = productList_.end();
    ProductRegistry::ProductList::const_iterator i = other.productList().begin();
    ProductRegistry::ProductList::const_iterator e = other.productList().end();

    // Loop over entries in the main product registry.
    while(j != s || i != e) {
      if (j != s && j->second.produced()) {
	// Ignore branches just produced (i.e. not in input file).
	++j;
      } else if (j == s || i != e && i->first < j->first) {
	if (i->second.present()) {
	  differences << "Branch '" << i->second.branchName() << "' is in file '" << fileName << "'\n";
	  differences << "    but not in previous files.\n";
	}
	++i;
      } else if (i == e || j != s && j->first < i->first) {
	// Allow branch to be missing in new file
	++j;
      } else {
	std::string difs = match(j->second, i->second, fileName, m);
	if (difs.empty()) {
	  if (m == BranchDescription::Permissive) j->second.merge(i->second);
	} else {
	  differences << difs;
	}
	++i;
	++j;
      }
    }
    initializeTransients();
    return differences.str();
  }

  void ProductRegistry::initializeTransients() const {
    constProductList_.clear();
    productLookup_.clear();
    elementLookup_.clear();
    for (ProductList::const_iterator i = productList_.begin(), e = productList_.end(); i != e; ++i) {
      constProductList_.insert(std::make_pair(i->first, ConstBranchDescription(i->second)));

      ProcessLookup& processLookup = productLookup_[i->first.friendlyClassName_];
      std::vector<BranchID>& vint = processLookup[i->first.processName_];
      vint.push_back(i->second.branchID());
      //[could use productID instead]
        
      ROOT::Reflex::Type type(ROOT::Reflex::Type::ByName(i->second.className()));
      if (bool(type)) {
        
        // Here we look in the object named "type" for a typedef
        // named "value_type" and get the Reflex::Type for it.
        // Then check to ensure the Reflex dictionary is defined
        // for this value_type.
        // I do not throw an exception here if the check fails
        // because there are known cases where the dictionary does
        // not exist and we do not need to support those cases.
        ROOT::Reflex::Type valueType;
        if ((is_RefVector(type, valueType) || 
	     is_RefToBaseVector(type, valueType ) || 
	     value_type_of(type, valueType)) 
            && bool(valueType)) {
          
          fillElementLookup(valueType, i->second.branchID(), i->first);
          
          // Repeat this for all public base classes of the value_type
          std::vector<ROOT::Reflex::Type> baseTypes;
          public_base_classes(valueType, baseTypes);
          
          for (std::vector<ROOT::Reflex::Type>::iterator iter = baseTypes.begin(),
	       iend = baseTypes.end();
               iter != iend;
               ++iter) {
            fillElementLookup(*iter, i->second.branchID(), i->first);
          }
        }
      }
    }
  }

  void ProductRegistry::fillElementLookup(const ROOT::Reflex::Type & type,
                                          const BranchID& id,
                                          const BranchKey& bk) const
  {
    TypeID typeID(type.TypeInfo());
    std::string friendlyClassName = typeID.friendlyClassName();
    
    ProcessLookup& processLookup = elementLookup_[friendlyClassName];
    std::vector<BranchID>& vint = processLookup[bk.processName_];
    vint.push_back(id);    
  }
  
  void ProductRegistry::print(std::ostream& os) const {
    for (ProductList::const_iterator i = productList_.begin(), e = productList_.end(); i != e; ++i) {
	os << i->second << "\n-----\n";
    }
  }

}
