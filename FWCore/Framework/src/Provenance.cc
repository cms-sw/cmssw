#include "FWCore/Framework/interface/Provenance.h"

/*----------------------------------------------------------------------

$Id: Provenance.cc,v 1.6 2005/07/26 04:42:28 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {
  Provenance::Provenance() :
    product(),
    productID_(),
    parents(),
    cid(),
    status(Success)
  { }

  Provenance::Provenance(ProductDescription const& p) :
    product(p),
    productID_(),
    parents(),
    cid(),
    status(Success)
  { }

  void
  Provenance::write(std::ostream& os) const {
    // This is grossly inadequate, but it is not critical for the
    // first pass.
    os << "Provenance for: " << cid;
  }

    
  bool operator==(const Provenance& a, const Provenance& b) {
    return
      a.product == b.product
      && a.cid == b.cid
      && a.status == b.status
      && a.parents == b.parents;
  }

}

