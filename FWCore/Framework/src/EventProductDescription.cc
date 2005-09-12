#include "FWCore/Framework/interface/EventProductDescription.h"

/*----------------------------------------------------------------------

$Id: EventProductDescription.cc,v 1.1 2005/08/03 07:03:18 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {
  EventProductDescription::EventProductDescription() :
    productID_(),
    parents(),
    cid(),
    status(Success)
  { }

  void
  EventProductDescription::write(std::ostream& os) const {
    // This is grossly inadequate, but it is not critical for the
    // first pass.
    os << "EventProductDescription for: " << cid;
  }

    
  bool
  EventProductDescription::operator==(EventProductDescription const& b) const {
    return
      productID_ == b.productID_
      && cid == b.cid
      && status == b.status
      && parents == b.parents;
  }

}

