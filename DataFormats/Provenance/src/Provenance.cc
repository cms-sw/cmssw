#include "DataFormats/Provenance/interface/Provenance.h"
#include <ostream>

/*----------------------------------------------------------------------

$Id: Provenance.cc,v 1.1 2007/03/04 04:48:10 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {
  Provenance::Provenance(BranchDescription const& p, BranchEntryDescription::CreatorStatus const& status) :
    product_(p),
    event_(p.productID(), status)
  { }

  Provenance::Provenance(BranchDescription const& p, BranchEntryDescription const& e) :
    product_(p),
    event_(e)
  { }

  void
  Provenance::write(std::ostream& os) const {
    // This is grossly inadequate, but it is not critical for the
    // first pass.
    product().write(os);
    event().write(os);
  }

    
  bool operator==(Provenance const& a, Provenance const& b) {
    return
      a.product() == b.product()
      && a.event() == b.event();
  }

}

