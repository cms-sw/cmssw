#include "FWCore/Framework/interface/Provenance.h"

/*----------------------------------------------------------------------

$Id: Provenance.cc,v 1.9 2005/08/24 19:24:01 wmtan Exp $

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

