#include "DataFormats/Provenance/interface/EventRange.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include <cassert>
#include <ostream>
//#include <limits>

namespace edm {
  EventRange::EventRange() :
      // Special cases since 0 means maximum
      startEventID_(0U, 0U, EventID::maxEventNumber()),
      endEventID_(0U, 0U, EventID::maxEventNumber()) {
  }

  EventRange::EventRange(RunNumber_t startRun, LuminosityBlockNumber_t startLumi, EventNumber_t startEvent,
                         RunNumber_t endRun, LuminosityBlockNumber_t endLumi, EventNumber_t endEvent) :
      // Special cases since 0 means maximum
      startEventID_(startRun, startLumi, startEvent != 0 ? startEvent : EventID::maxEventNumber()),
      endEventID_(endRun, endLumi, endEvent != 0 ? endEvent : EventID::maxEventNumber()) {
    assert((startLumi == 0) == (endLumi == 0));
  }

  EventRange::EventRange(EventID const& begin, EventID const& end) :
      startEventID_(begin),
      endEventID_(end) {
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
    return !distinct(lh, rh);
  }

  bool lessThanSpecial(EventRange const& lh, EventRange const& rh) {
    // First, separate the ranges so that those with 0 lumiID go first.
    if ((lh.startLumi() == 0) != (rh.startLumi() == 0)) {
      return lh.startLumi() == 0;
    }
    return lh.endEventID() < rh.startEventID();
  }

  bool lessThan(EventRange const& lh, EventRange const& rh) {
    assert((lh.startLumi() == 0) == (rh.startLumi() == 0));
    return lh.endEventID() < rh.startEventID();
  }

  bool distinct(EventRange const& lh, EventRange const& rh) {
    assert((lh.startLumi() == 0) == (rh.startLumi() == 0));
    return lessThan(lh, rh) || lessThan(rh, lh); 
  }

  namespace {
    bool mergeSpecial(EventRange& lh, EventRange& rh) {
      // Don't combine a range with 0 lumiID with a range with non-zero lumiID.
      if ((lh.startLumi() == 0) != (rh.startLumi() == 0)) {
        return false;
      }
      if (overlaps(lh, rh)) {
        EventID begin = min(lh.startEventID(), rh.startEventID());
        EventID end = max(lh.endEventID(), rh.endEventID());
        rh = lh = EventRange(begin, end);
        return true;
      }
      return false;
    }

    bool sortByStartEventIDSpecial(EventRange const& lh, EventRange const& rh) {
      // First, separate the ranges so that those with 0 lumiID go first.
      if ((lh.startLumi() == 0) != (rh.startLumi() == 0)) {
        return lh.startLumi() == 0;
      }
      return lh.startEventID() < rh.startEventID();
    }
  }

  std::vector<EventRange>&
  sortAndRemoveOverlaps(std::vector<EventRange>& eventRange) {
    if (eventRange.size() <= 1U) return eventRange;
    sort_all(eventRange, sortByStartEventIDSpecial);
    for (std::vector<EventRange>::iterator i = eventRange.begin() + 1, e = eventRange.end();
        i != e; ++i) {
      std::vector<EventRange>::iterator iprev = i - 1;
      if (mergeSpecial(*iprev, *i)) {
        i = eventRange.erase(iprev);
        e = eventRange.end();
      }
    }
    return eventRange;
  }
}
