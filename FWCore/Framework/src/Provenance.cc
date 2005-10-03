#include "FWCore/Framework/interface/Provenance.h"

/*----------------------------------------------------------------------

$Id: Provenance.cc,v 1.10 2005/09/28 04:52:41 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {
  Provenance::Provenance() :
    product(),
    event()
  { }

  Provenance::Provenance(BranchDescription const& p) :
    product(p),
    event()
  { }

  void
  Provenance::write(std::ostream& os) const {
    // This is grossly inadequate, but it is not critical for the
    // first pass.
    product.write(os);
    event.write(os);
  }

    
  bool operator==(Provenance const& a, Provenance const& b) {
    return
      a.product == b.product
      && a.event == b.event;
  }

}

