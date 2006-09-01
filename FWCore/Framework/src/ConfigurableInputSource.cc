/*----------------------------------------------------------------------
$Id: ConfigurableInputSource.cc,v 1.7 2006/08/16 23:39:53 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConfigurableInputSource.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Event.h"

namespace edm {
  //used for defaults
  static const unsigned long kNanoSecPerSec = 1000000000;
  static const unsigned long kAveEventPerSec = 200;
  
  ConfigurableInputSource::ConfigurableInputSource(ParameterSet const& pset,
				       InputSourceDescription const& desc) :
    InputSource(pset, desc),
    numberEventsInRun_(pset.getUntrackedParameter<unsigned int>("numberEventsInRun", remainingEvents())),
    presentTime_(pset.getUntrackedParameter<unsigned int>("firstTime", 0)),  //time in ns
    origTime_(presentTime_),
    timeBetweenEvents_(pset.getUntrackedParameter<unsigned int>("timeBetweenEvents", kNanoSecPerSec/kAveEventPerSec)),
    numberEventsInThisRun_(0),
    zerothEvent_(pset.getUntrackedParameter<unsigned int>("firstEvent", 1) - 1),
    eventID_(pset.getUntrackedParameter<unsigned int>("firstRun", 1), zerothEvent_),
    origEventID_(eventID_)
  { }

  ConfigurableInputSource::~ConfigurableInputSource() {
  }

  std::auto_ptr<EventPrincipal>
  ConfigurableInputSource::read() {
    setRunAndEventInfo();
    if (eventID_ == EventID()) {
      return std::auto_ptr<EventPrincipal>(0); 
    }
    std::auto_ptr<EventPrincipal> result = 
      std::auto_ptr<EventPrincipal>(new EventPrincipal(eventID_, Timestamp(presentTime_), productRegistry()));
    Event e(*result, moduleDescription());
    if (!produce(e)) {
      return std::auto_ptr<EventPrincipal>(0); 
    }
    e.commit_();
    return result;
  }


  std::auto_ptr<EventPrincipal>
  ConfigurableInputSource::readIt(EventID const& eventID) {
    eventID_ = eventID.previous();
    return read();
  }
 
  void
  ConfigurableInputSource::skip(int offset) {
    for (; offset < 0; ++offset) {
       eventID_ = eventID_.previous();
    }
    for (; offset > 0; --offset) {
       eventID_ = eventID_.next();
    }
  }

  void
  ConfigurableInputSource::setRunAndEventInfo() {
    //NOTE: numberEventsInRun < 0 means go forever in this run
    if (numberEventsInRun_ < 1 || numberEventsInThisRun_ < numberEventsInRun_) {
      ++numberEventsInThisRun_;
      eventID_ = eventID_.next();
    } else {
      eventID_ = eventID_.nextRunFirstEvent();
      //reset this to one since this event is in the new run
      numberEventsInThisRun_ = 1;
    }
    presentTime_ += timeBetweenEvents_;
  }

}
