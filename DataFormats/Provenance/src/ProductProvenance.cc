#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "DataFormats/Provenance/interface/ParentageRegistry.h"
#include <ostream>
#include <cassert>

/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

namespace edm {
  ProductProvenance::Transients::Transients() :
    parentagePtr_(),
    noParentage_(false)
  {}

  ProductProvenance::ProductProvenance() :
    branchID_(),
    productStatus_(productstatus::uninitialized()),
    parentageID_(),
    transients_()
  {}

  ProductProvenance::ProductProvenance(BranchID const& bid) :
    branchID_(bid),
    productStatus_(productstatus::uninitialized()),
    parentageID_(),
    transients_()
  {}

   ProductProvenance::ProductProvenance(BranchID const& bid,
				    ProductStatus status) :
    branchID_(bid),
    productStatus_(status),
    parentageID_(),
    transients_()
  {}

   ProductProvenance::ProductProvenance(BranchID const& bid,
				    ProductStatus status,
				    ParentageID const& edid) :
    branchID_(bid),
    productStatus_(status),
    parentageID_(edid),
    transients_()
  {}

   ProductProvenance::ProductProvenance(BranchID const& bid,
				    ProductStatus status,
				    boost::shared_ptr<Parentage> pPtr) :
    branchID_(bid),
    productStatus_(status),
    parentageID_(pPtr->id()),
    transients_() {
       parentagePtr() = pPtr;
       ParentageRegistry::instance()->insertMapped(*pPtr);
  }

  ProductProvenance::ProductProvenance(BranchID const& bid,
		   ProductStatus status,
		   std::vector<BranchID> const& parents) :
    branchID_(bid),
    productStatus_(status),
    parentageID_(),
    transients_() {
      parentagePtr() = boost::shared_ptr<Parentage>(new Parentage);
      parentagePtr()->parents() = parents;
      parentageID_ = parentagePtr()->id();
      ParentageRegistry::instance()->insertMapped(*parentagePtr());
  }

  ProductProvenance
  ProductProvenance::makeProductProvenance() const {
    return *this;
  }

  Parentage const &
  ProductProvenance::parentage() const {
    if (!parentagePtr()) {
      parentagePtr().reset(new Parentage);
      ParentageRegistry::instance()->getMapped(parentageID_, *parentagePtr());
    }
    return *parentagePtr();
  }

  void
  ProductProvenance::setStatus(ProductStatus const& status) {
    if (productstatus::presenceUnknown(productStatus())) {
      productStatus_ = status;
    } else if (productstatus::present(productStatus())) {
      assert(productstatus::present(status));
    } else {
      assert(productstatus::notPresent(productStatus()));
      assert(productstatus::notPresent(status));
      if (!productstatus::neverCreated(status)) {
        productStatus_ = status;
      }
    }
  }

  void
  ProductProvenance::write(std::ostream& os) const {
    os << "branch ID = " << branchID() << '\n';
    os << "product status = " << static_cast<int>(productStatus()) << '\n';
    if (!noParentage()) {
      os << "entry description ID = " << parentageID() << '\n';
    }
  }
    
  bool
  operator==(ProductProvenance const& a, ProductProvenance const& b) {
    if (a.noParentage() != b.noParentage()) return false;
    if (a.noParentage()) {
      return
        a.branchID() == b.branchID()
        && a.productStatus() == b.productStatus();
    }
    return
      a.branchID() == b.branchID()
      && a.productStatus() == b.productStatus()
      && a.parentageID() == b.parentageID();
  }
}
