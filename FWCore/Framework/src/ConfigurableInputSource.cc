/*----------------------------------------------------------------------
$Id: ConfigurableInputSource.cc,v 1.30 2007/12/03 00:40:50 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConfigurableInputSource.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"

namespace edm {
  //used for defaults
  static const unsigned int kNanoSecPerSec = 1000000000U;
  static const unsigned int kAveEventPerSec = 200U;
  
  ConfigurableInputSource::ConfigurableInputSource(ParameterSet const& pset,
				       InputSourceDescription const& desc, bool realData) :
    InputSource(pset, desc),
    numberEventsInRun_(pset.getUntrackedParameter<unsigned int>("numberEventsInRun", remainingEvents())),
    numberEventsInLumi_(pset.getUntrackedParameter<unsigned int>("numberEventsInLuminosityBlock", remainingEvents())),
    presentTime_(pset.getUntrackedParameter<unsigned int>("firstTime", 0)),  //time in ns
    origTime_(presentTime_),
    timeBetweenEvents_(pset.getUntrackedParameter<unsigned int>("timeBetweenEvents", kNanoSecPerSec/kAveEventPerSec)),
    eventCreationDelay_(pset.getUntrackedParameter<unsigned int>("eventCreationDelay", 0)),
    numberEventsInThisRun_(0),
    numberEventsInThisLumi_(0),
    zerothEvent_(pset.getUntrackedParameter<unsigned int>("firstEvent", 1) - 1),
    eventID_(pset.getUntrackedParameter<unsigned int>("firstRun", 1), zerothEvent_),
    origEventID_(eventID_),
    luminosityBlock_(pset.getUntrackedParameter<unsigned int>("firstLuminosityBlock", 1)),
    origLuminosityBlockNumber_t_(luminosityBlock_),
    newRun_(true),
    newLumi_(true),
    eventSet_(false),
    ep_(0),
    isRealData_(realData),
    eType_(EventAuxiliary::Any)
  { 
    setTimestamp(Timestamp(presentTime_));
    // We need to map this string to the EventAuxiliary::ExperimentType enumeration
    // std::string eType = pset.getUntrackedParameter<std::string>("experimentType", std::string("Any"))),
  }

  ConfigurableInputSource::~ConfigurableInputSource() {
  }

  boost::shared_ptr<RunPrincipal>
  ConfigurableInputSource::readRun_() {
    Timestamp ts = Timestamp(presentTime_);
    boost::shared_ptr<RunPrincipal> runPrincipal(
        new RunPrincipal(eventID_.run(), ts, Timestamp::invalidTimestamp(), productRegistry(), processConfiguration()));
    RunPrincipal & rp =
       const_cast<RunPrincipal &>(*runPrincipal);
    Run run(rp, moduleDescription());
    beginRun(run);
    run.commit_();
    newRun_ = false;
    return runPrincipal;
  }

  boost::shared_ptr<LuminosityBlockPrincipal>
  ConfigurableInputSource::readLuminosityBlock_() {
    Timestamp ts = Timestamp(presentTime_);
    boost::shared_ptr<LuminosityBlockPrincipal> lumiPrincipal(
        new LuminosityBlockPrincipal(
	    luminosityBlock_, ts, Timestamp::invalidTimestamp(), productRegistry(), runPrincipal(), processConfiguration()));
    LuminosityBlock lb(*lumiPrincipal, moduleDescription());
    beginLuminosityBlock(lb);
    lb.commit_();
    newLumi_ = false;
    return lumiPrincipal;
  }

  std::auto_ptr<EventPrincipal>
  ConfigurableInputSource::readEvent_(boost::shared_ptr<LuminosityBlockPrincipal>) {
    assert(ep_.get() != 0);
    return ep_;
  }

  std::auto_ptr<EventPrincipal>
  ConfigurableInputSource::reallyReadEvent(boost::shared_ptr<LuminosityBlockPrincipal> lbp) {
    std::auto_ptr<EventPrincipal> result(
	new EventPrincipal(eventID_, Timestamp(presentTime_),
	productRegistry(), lbp, processConfiguration(), isRealData_, eType_));
    Event e(*result, moduleDescription());
    if (!produce(e)) {
      return std::auto_ptr<EventPrincipal>(0); 
    }
    e.commit_();
    return result;
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
  ConfigurableInputSource::setRun(RunNumber_t r) {
    // Do nothing if the run is not changed.
    if (r != eventID_.run()) {
      eventID_ = EventID(r, zerothEvent_);
      luminosityBlock_ = origLuminosityBlockNumber_t_;
      numberEventsInThisRun_ = 0;
      numberEventsInThisLumi_ = 0;
      newRun_ = newLumi_ = true;
      resetLuminosityBlockPrincipal();
      resetRunPrincipal();
    }
  }

  void
  ConfigurableInputSource::setLumi(LuminosityBlockNumber_t lb) {
    // Do nothing if the lumi block is not changed.
    if (lb != luminosityBlock_) {
      luminosityBlock_ = lb;
      numberEventsInThisLumi_ = 0;
      newLumi_ = true;
      resetLuminosityBlockPrincipal();
    }
  }

  void
  ConfigurableInputSource::rewind_() {
    luminosityBlock_ = origLuminosityBlockNumber_t_;
    presentTime_ = origTime_;
    eventID_ = origEventID_;
    numberEventsInThisRun_ = 0;
    numberEventsInThisLumi_ = 0;
    newRun_ = newLumi_ = true;
    resetLuminosityBlockPrincipal();
    resetRunPrincipal();
  }
    

  InputSource::ItemType 
  ConfigurableInputSource::getNextItemType() {
    if (newRun_) {
      return IsRun;
    }
    if (newLumi_) {
      return IsLumi;
    }
    if(ep_.get() != 0) return IsEvent;
    EventID oldEventID = eventID_;
    LuminosityBlockNumber_t oldLumi = luminosityBlock_;
    if (!eventSet_) {
      setRunAndEventInfo();
      eventSet_ = true;
    }
    if (eventID_ == EventID()) {
      ep_.reset();
      return IsStop;
    }
    if (oldEventID.run() != eventID_.run()) {
      //  New Run
      // reset these since this event is in the new run
      numberEventsInThisRun_ = 0;
      numberEventsInThisLumi_ = 0;
      luminosityBlock_ = origLuminosityBlockNumber_t_;
      newRun_ = newLumi_ = true;
      resetLuminosityBlockPrincipal();
      resetRunPrincipal();
      return IsRun;
    }
      // Same Run
    if (oldLumi != luminosityBlock_) {
      // New Lumi
      numberEventsInThisLumi_ = 0;
      newLumi_ = true;
      resetLuminosityBlockPrincipal();
      return IsLumi;
    }
    ++numberEventsInThisRun_;
    ++numberEventsInThisLumi_;
    ep_ = reallyReadEvent(luminosityBlockPrincipal());
    if (ep_.get() == 0) {
      return IsStop;
    }
    eventSet_ = false;
    return IsEvent;
  }

  void
  ConfigurableInputSource::setRunAndEventInfo() {
    //NOTE: numberEventsInRun < 0 means go forever in this run
    if (numberEventsInRun_ < 1 || numberEventsInThisRun_ < numberEventsInRun_) {
      // same run
      eventID_ = eventID_.next();
      if (!(numberEventsInLumi_ < 1 || numberEventsInThisLumi_ < numberEventsInLumi_)) {
        // new lumi
        ++luminosityBlock_;
      }
    } else {
      // new run
      eventID_ = eventID_.nextRunFirstEvent();
    }
    presentTime_ += timeBetweenEvents_;
    if (eventCreationDelay_ > 0) {usleep(eventCreationDelay_);}
  }

}
