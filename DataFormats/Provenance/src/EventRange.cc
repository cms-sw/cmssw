#include "DataFormats/Provenance/interface/EventRange.h"
#include <cassert>
#include <ostream>
//#include <limits>

namespace edm {
  EventRange::EventRange():
      // Special cases since 0 means maximum
      startEventID_(0U, 0U, EventID::maxEventNumber()),
      endEventID_(0U, 0U, EventID::maxEventNumber()) {
  }

  EventRange::EventRange(RunNumber_t startRun, LumiNumber_t startLumi, EventNumber_t startEvent,
                         RunNumber_t endRun, LumiNumber_t endLumi, EventNumber_t endEvent) :
      // Special cases since 0 means maximum
      startEventID_(startRun, startLumi, startEvent != 0 ? startEvent : EventID::maxEventNumber()),
      endEventID_(endRun, endLumi, endEvent != 0 ? endEvent : EventID::maxEventNumber()) {
    assert((startLumi == 0) == (endLumi == 0));
  }

  std::ostream& operator<<(std::ostream& oStream, EventRange const& r) {
    if (r.startLumi() == 0) {
      oStream << "'" << r.startRun() << ":" << r.startEvent() << "-"
                     << r.endRun() << ":" << r.endEvent() << "'";
    } else {
      oStream << "'" << r.startRun() << ":" << r.startLumi() << ":" << r.startEvent() << "-"
                     << r.endRun() << ":" << r.endLumi() << ":" << r.endEvent() << "'";
    }
    return oStream;
  }

  bool contains(EventRange const& lh, EventID const& rh) {
    if (lh.startLumi() == 0) {
      return (contains_(lh, EventID(rh.run(), 0U, rh.event())));
    }
    return (contains_(lh, rh));
  }

  bool contains_(EventRange const& lh, EventID const& rh) {
    return (rh >= lh.startEventID() && rh <= lh.endEventID());
  }

  bool contains(EventRange const& lh, EventRange const& rh) {
    assert((lh.startLumi() == 0) == (rh.startLumi() == 0));
    return (contains(lh, rh.startEventID()) && contains(lh, rh.endEventID()));
  }

  bool overlaps(EventRange const& lh, EventRange const& rh) {
    assert((lh.startLumi() == 0) == (rh.startLumi() == 0));
    return (contains(lh, rh.startEventID()) || contains(lh, rh.endEventID()));
  }

  bool distinct(EventRange const& lh, EventRange const& rh) {
    return !overlaps(lh, rh);
  }
}
