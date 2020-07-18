#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "DataFormats/Provenance/interface/ParentageRegistry.h"

#include <cassert>
#include <ostream>

/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

namespace {
  edm::Parentage const s_emptyParentage;
}
namespace edm {
  ProductProvenance::ProductProvenance() : branchID_(), parentageID_() {}

  ProductProvenance::ProductProvenance(BranchID bid) : branchID_(bid), parentageID_() {}

  ProductProvenance::ProductProvenance(BranchID bid, ParentageID edid)
      : branchID_(bid), parentageID_(std::move(edid)) {}

  ProductProvenance::ProductProvenance(BranchID bid, std::vector<BranchID> const& parents)
      : branchID_(bid), parentageID_() {
    Parentage p;
    p.setParents(parents);
    parentageID_ = p.id();
    ParentageRegistry::instance()->insertMapped(p);
  }

  ProductProvenance::ProductProvenance(BranchID bid, std::vector<BranchID>&& parents) : branchID_(bid), parentageID_() {
    Parentage p;
    p.setParents(std::move(parents));
    parentageID_ = p.id();
    ParentageRegistry::instance()->insertMapped(std::move(p));
  }

  ProductProvenance ProductProvenance::makeProductProvenance() const { return *this; }

  Parentage const& ProductProvenance::parentage() const {
    auto p = ParentageRegistry::instance()->getMapped(parentageID_);
    if (p) {
      return *p;
    }
    return s_emptyParentage;
  }

  void ProductProvenance::write(std::ostream& os) const {
    os << "branch ID = " << branchID() << '\n';
    os << "entry description ID = " << parentageID() << '\n';
  }

  bool operator==(ProductProvenance const& a, ProductProvenance const& b) {
    return a.branchID() == b.branchID() && a.parentageID() == b.parentageID();
  }
}  // namespace edm
