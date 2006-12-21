/*----------------------------------------------------------------------
$Id: ConfigurableInputSource.cc,v 1.10 2006/12/19 00:28:56 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConfigurableInputSource.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/Event.h"

namespace edm {
  //used for defaults
  static const unsigned int kNanoSecPerSec = 1000000000U;
  static const unsigned int kAveEventPerSec = 200U;
  
  ConfigurableInputSource::ConfigurableInputSource(ParameterSet const& pset,
				       InputSourceDescription const& desc) :
    InputSource(pset, desc),
    numberEventsInRun_(pset.getUntrackedParameter<unsigned int>("numberEventsInRun", remainingEvents())),
    numberEventsInLumi_(pset.getUntrackedParameter<unsigned int>("numberEventsInLuminosityBlock", remainingEvents())),
    presentTime_(pset.getUntrackedParameter<unsigned int>("firstTime", 0)),  //time in ns
    origTime_(presentTime_),
    timeBetweenEvents_(pset.getUntrackedParameter<unsigned int>("timeBetweenEvents", kNanoSecPerSec/kAveEventPerSec)),
    numberEventsInThisRun_(0),
    numberEventsInThisLumi_(0),
    zerothEvent_(pset.getUntrackedParameter<unsigned int>("firstEvent", 1) - 1),
    eventID_(pset.getUntrackedParameter<unsigned int>("firstRun", 1), zerothEvent_),
    origEventID_(eventID_),
    luminosityBlockID_(pset.getUntrackedParameter<unsigned int>("firstLuminosityBlock", 1)),
    origLuminosityBlockID_(luminosityBlockID_),
    luminosityBlockPrincipal_()
  { }

  ConfigurableInputSource::~ConfigurableInputSource() {
  }

  std::auto_ptr<EventPrincipal>
  ConfigurableInputSource::read() {
    RunNumber_t oldRun = eventID_.run();
    LuminosityBlockID oldLumi = luminosityBlockID_;
    setRunAndEventInfo();
    if (oldRun != eventID_.run() || luminosityBlockPrincipal_.get() == 0) {
      boost::shared_ptr<RunPrincipal const> runPrincipal(new RunPrincipal(eventID_.run(), productRegistry()));
      luminosityBlockPrincipal_ = boost::shared_ptr<LuminosityBlockPrincipal const>(
			new LuminosityBlockPrincipal(luminosityBlockID_, productRegistry(), runPrincipal));
    } else if (oldLumi != luminosityBlockID_) {
      boost::shared_ptr<RunPrincipal const> runPrincipal = luminosityBlockPrincipal_->runPrincipalConstSharedPtr();
      luminosityBlockPrincipal_ = boost::shared_ptr<LuminosityBlockPrincipal const>(
			new LuminosityBlockPrincipal(luminosityBlockID_, productRegistry(), runPrincipal));
    }
    if (eventID_ == EventID()) {
      return std::auto_ptr<EventPrincipal>(0); 
    }
    std::auto_ptr<EventPrincipal> result = 
      std::auto_ptr<EventPrincipal>(new EventPrincipal(eventID_,
						       Timestamp(presentTime_),
						       productRegistry(),
						       luminosityBlockPrincipal_));
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
      // same run
      ++numberEventsInThisRun_;
      eventID_ = eventID_.next();
      if (numberEventsInLumi_ < 1 || numberEventsInThisLumi_ < numberEventsInLumi_) {
	// same lumi
        ++numberEventsInThisLumi_;
      } else {
        // new lumi
        numberEventsInThisLumi_ = 1;
        ++luminosityBlockID_;
      }
    } else {
      // new run
      eventID_ = eventID_.nextRunFirstEvent();
      luminosityBlockID_ = origLuminosityBlockID_;
      //reset these to one since this event is in the new run
      numberEventsInThisRun_ = 1;
      numberEventsInThisLumi_ = 1;
    }
    presentTime_ += timeBetweenEvents_;
  }

}
