#include "DataFormats/Common/interface/Provenance.h"
#include <ostream>

/*----------------------------------------------------------------------

$Id: Provenance.cc,v 1.2 2006/08/24 22:15:44 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {
  Provenance::Provenance(BranchDescription const& p, BranchEntryDescription::CreatorStatus const& status) :
    product(p),
    event(p.productID(), status)
  { }

  Provenance::Provenance(BranchDescription const& p, BranchEntryDescription const& e) :
    product(p),
    event(e)
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

