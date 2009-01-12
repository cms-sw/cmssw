#include "DataFormats/Provenance/interface/EventRange.h"
#include <ostream>
//#include <limits>



namespace edm {

  std::ostream& operator<<(std::ostream& oStream, EventRange const& r) {
    oStream << "'" << r.startRun() << ":" << r.startEvent() << "-"
                   << r.endRun()   << ":" << r.endEvent()   << "'" ;
    return oStream;
  }

  bool EventRange::contains(EventID const& test) const {
    if (test >= startEventID_ && test <= endEventID_) {
      return true;
    }
    return false;
  }

}
