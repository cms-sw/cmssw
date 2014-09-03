#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "DataFormats/Provenance/interface/ParentageRegistry.h"

#include <cassert>
#include <ostream>

/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

namespace edm {
  ProductProvenance::Transients::Transients() :
    parentagePtr_(),
    noParentage_(false)
  {}

  void
  ProductProvenance::Transients::reset() {
    parentagePtr_.reset();
    noParentage_ = false;
  }

  ProductProvenance::ProductProvenance() :
    branchID_(),
    parentageID_(),
    transient_()
  {}

  ProductProvenance::ProductProvenance(BranchID const& bid) :
    branchID_(bid),
    parentageID_(),
    transient_()
  {}

   ProductProvenance::ProductProvenance(BranchID const& bid,
				    ParentageID const& edid) :
    branchID_(bid),
    parentageID_(edid),
    transient_()
  {}

   ProductProvenance::ProductProvenance(BranchID const& bid,
				    std::shared_ptr<Parentage> pPtr) :
    branchID_(bid),
    parentageID_(pPtr->id()),
    transient_() {
       parentagePtr() = pPtr;
       ParentageRegistry::instance()->insertMapped(*pPtr);
  }

  ProductProvenance::ProductProvenance(BranchID const& bid,
		   std::vector<BranchID> const& parents) :
    branchID_(bid),
    parentageID_(),
    transient_() {
      parentagePtr() = std::make_shared<Parentage>();
      parentagePtr()->setParents(parents);
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
  ProductProvenance::write(std::ostream& os) const {
    os << "branch ID = " << branchID() << '\n';
    if (!noParentage()) {
      os << "entry description ID = " << parentageID() << '\n';
    }
  }
    
  bool
  operator==(ProductProvenance const& a, ProductProvenance const& b) {
    if (a.noParentage() != b.noParentage()) return false;
    if (a.noParentage()) {
      return
        a.branchID() == b.branchID();
    }
    return
      a.branchID() == b.branchID() && a.parentageID() == b.parentageID();
  }
}
