/*----------------------------------------------------------------------
$Id: RawInputSource.cc,v 1.4 2006/01/07 00:38:14 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/RawInputSource.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Event.h"

namespace edm {
  RawInputSource::RawInputSource(ParameterSet const& pset,
				       InputSourceDescription const& desc) :
    GenericInputSource(desc),
    remainingEvents_(pset.getUntrackedParameter<int>("maxEvents", -1)),
    ep_(0)
  { }

  RawInputSource::~RawInputSource() {
  }

  std::auto_ptr<EventPrincipal>
  RawInputSource::read() {
    if(remainingEvents_ != 0) {
      std::auto_ptr<Event> e(readOneEvent());
      if(e.get() != 0) {
        --remainingEvents_;
        e->commit_();
      }
    }
    return ep_;
  }

  std::auto_ptr<Event>
  RawInputSource::makeEvent(EventID const& eventId, Timestamp const& tstamp) {
    ep_ = std::auto_ptr<EventPrincipal>(new EventPrincipal(eventId, Timestamp(tstamp), productRegistry()));
    std::auto_ptr<Event> e(new Event(*ep_, module()));
    return e;
  }
}
