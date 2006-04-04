/*----------------------------------------------------------------------
$Id: ConfigurableInputSource.cc,v 1.1 2006/01/18 00:38:44 wmtan Exp $
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
    remainingEvents_(maxEvents()),
    numberEventsInRun_(pset.getUntrackedParameter<unsigned int>("numberEventsInRun", remainingEvents_)),
    presentTime_(pset.getUntrackedParameter<unsigned int>("firstTime", 0)),  //time in ns
    timeBetweenEvents_(pset.getUntrackedParameter<unsigned int>("timeBetweenEvents", kNanoSecPerSec/kAveEventPerSec)),
    numberEventsInThisRun_(0),
    eventID_(pset.getUntrackedParameter<unsigned int>("firstRun", 1), 0)
  { }

  ConfigurableInputSource::~ConfigurableInputSource() {
  }

  std::auto_ptr<EventPrincipal>
  ConfigurableInputSource::readOneEvent() {
    setRunAndEventInfo();
    if (eventID_ == EventID()) {
      return std::auto_ptr<EventPrincipal>(0); 
    }
    std::auto_ptr<EventPrincipal> result = 
      std::auto_ptr<EventPrincipal>(new EventPrincipal(eventID_, Timestamp(presentTime_), productRegistry()));
    Event e(*result, module());
    if (!produce(e)) {
      return std::auto_ptr<EventPrincipal>(0); 
    }
    e.commit_();
    return result;
  }

  std::auto_ptr<EventPrincipal>
  ConfigurableInputSource::read() {
    std::auto_ptr<EventPrincipal> result(0);
    
    if (remainingEvents_ != 0) {
      result = readOneEvent();
      if (result.get() != 0) {
        ++numberEventsInThisRun_;
        --remainingEvents_;
      }
    }
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
    if (numberEventsInThisRun_ < numberEventsInRun_) {
      eventID_ = eventID_.next();
    } else {
      eventID_ = eventID_.nextRunFirstEvent();
      numberEventsInThisRun_ = 0;
    }
    presentTime_ += timeBetweenEvents_;
  }

}
