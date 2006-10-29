/**
   \file
   class impl

   \Original author Stefano ARGIRO
   \Current author Bill Tanenbaum
   \version $Id: ProductRegistry.cc,v 1.9 2006/08/22 05:46:40 wmtan Exp $
   \date 19 Jul 2005
*/

static const char CVSId[] = "$Id: ProductRegistry.cc,v 1.9 2006/08/22 05:46:40 wmtan Exp $";


#include "DataFormats/Common/interface/ProductRegistry.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <algorithm>
#include <sstream>

namespace edm {
  void
  ProductRegistry::addProduct(BranchDescription const& productDesc, bool fromListener) {
    throwIfFrozen();
    productDesc.init();
    productList_.insert(std::make_pair(BranchKey(productDesc), productDesc));
    addCalled(productDesc,fromListener);
  }
  
  void
  ProductRegistry::copyProduct(BranchDescription const& productDesc) {
    throwIfFrozen();
    productDesc.init();
    productList_.insert(std::make_pair(BranchKey(productDesc), productDesc));
    if (productDesc.productID().id_ >= nextID_) {
      nextID_ = productDesc.productID().id_ + 1;
    }
  }
  
  void
  ProductRegistry::setProductIDs() {
    throwIfFrozen();
    for (ProductList::iterator it = productList_.begin(); it != productList_.end(); ++it) {
       if (it->second.productID().id_ == 0) {
          it->second.productID_.id_ = nextID_++;
       }
    }
    frozen_ = true;
  }
  
  void
  ProductRegistry::setFrozen() const {
    if (frozen_) return;
/*
    for (ProductList::const_iterator it = productList_.begin(); it != productList_.end(); ++it) {
      if (it->second.productID_.id_ == 0) {
       throw cms::Exception("ProductRegistry", "setFrozen")
          << "cannot read the ProductRegistry because it is not yet frozen.";
      }
    }
*/
    frozen_ = true;
  }
  
  void
  ProductRegistry::throwIfFrozen() const {
    if (frozen_) {
      throw cms::Exception("ProductRegistry", "throwIfFrozen")
            << "cannot modify the ProductRegistry because it is frozen";
    }
  }
  
  void
  ProductRegistry::throwIfNotFrozen() const {
/*
    if (!frozen_) {
      throw cms::Exception("ProductRegistry", "throwIfNotFrozen")
            << "cannot read the ProductRegistry because it is not yet frozen";
    }
*/
  }
  
  void
  ProductRegistry::addCalled(BranchDescription const&, bool) {
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
      } else if (j == s || i->first < j->first) {
	differences << "Branch '" << i->second.branchName() << "' is in file '" << fileName << "'\n";
	differences << "    but not in previous files.\n";
	++i;
      } else if (i == e || j->first < i->first) {
	differences << "Branch '" << j->second.branchName() << "' is in previous files\n";
	differences << "    but not in file '" << fileName << "'.\n";
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
    return differences.str();
  }
}
