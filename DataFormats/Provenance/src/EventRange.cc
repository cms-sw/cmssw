#include "DataFormats/Provenance/interface/EventRange.h"
#include <ostream>
//#include <limits>

namespace edm {

  std::ostream& operator<<(std::ostream& oStream, EventRange const& r) {
    oStream << "'" << r.startRun() << ":" << r.startEvent() << "-"
                   << r.endRun()   << ":" << r.endEvent()   << "'" ;
    return oStream;
  }

/*  bool EventRange::contains(MinimalEventID const& test) const {
    if (test >= startEventID_ && test <= endEventID_) {
      return true;
    }
    return false;
  }*/
  bool contains(EventRange const& lh, MinimalEventID const& rh) {
    if (rh >= lh.startEventID() && rh <= lh.endEventID()) {
      return true;
    }
    return false;
  }

  bool contains(EventRange const& lh, EventRange const& rh) {
    if (contains(lh,rh.startEventID()) && contains(lh,rh.endEventID())) {
      return true;
    }
    return false;
  }

  bool overlaps(EventRange const& lh, EventRange const& rh) {
    if (contains(lh,rh.startEventID()) || contains(lh,rh.endEventID())) {
      return true;
    }
    return false;
  }

  bool distinct(EventRange const& lh, EventRange const& rh) {
    return !overlaps(lh,rh);
  }

}
