#include "FWCore/Framework/interface/Provenance.h"

/*----------------------------------------------------------------------

$Id: Provenance.cc,v 1.8 2005/08/03 05:31:53 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {
  Provenance::Provenance() :
    product(),
    event()
  { }

  Provenance::Provenance(ProductDescription const& p) :
    product(p),
    event()
  { }

  void
  Provenance::write(std::ostream&) const {
    // This is grossly inadequate, but it is not critical for the
    // first pass.
  }

    
  bool operator==(Provenance const& a, Provenance const& b) {
    return
      a.product == b.product
      && a.event == b.event;
  }

}

