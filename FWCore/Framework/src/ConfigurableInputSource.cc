/*----------------------------------------------------------------------
$Id: ConfigurableInputSource.cc,v 1.22 2007/06/24 22:59:36 wmtan Exp $
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
    isRealData_(realData),
    eType_(pset.getUntrackedParameter<std::string>("experimentType", std::string("Unspecified"))),
    holder_(0)
  { }

  ConfigurableInputSource::~ConfigurableInputSource() {
  }

  boost::shared_ptr<RunPrincipal>
  ConfigurableInputSource::readRun_() {
    boost::shared_ptr<RunPrincipal> runPrincipal(new RunPrincipal(eventID_.run(), productRegistry(), processConfiguration()));
    RunPrincipal & rp =
       const_cast<RunPrincipal &>(*runPrincipal);
    Run run(rp, moduleDescription());
    beginRun(run);
    run.commit_();
    newRun_ = false;
    return runPrincipal;
  }

  boost::shared_ptr<LuminosityBlockPrincipal>
  ConfigurableInputSource::readLuminosityBlock_(boost::shared_ptr<RunPrincipal> rp) {
    if (newRun_) return boost::shared_ptr<LuminosityBlockPrincipal>();
    boost::shared_ptr<LuminosityBlockPrincipal> luminosityBlockPrincipal(
        new LuminosityBlockPrincipal(luminosityBlock_, productRegistry(), rp, processConfiguration()));
    LuminosityBlockPrincipal & lbp =
       const_cast<LuminosityBlockPrincipal &>(*luminosityBlockPrincipal);
    LuminosityBlock lb(lbp, moduleDescription());
    beginLuminosityBlock(lb);
    lb.commit_();
    newLumi_ = false;
    return luminosityBlockPrincipal;
  }

  void
  ConfigurableInputSource::finishRun(RunPrincipal& rp) {
    Run run(rp, moduleDescription());
    endRun(run);
    run.commit_();
  }

  void
  ConfigurableInputSource::finishLumi(LuminosityBlockPrincipal & lbp) {
    LuminosityBlock lb(lbp, moduleDescription());
    endLuminosityBlock(lb);
    lb.commit_();
  }

  std::auto_ptr<EventPrincipal>
  ConfigurableInputSource::readEvent_(boost::shared_ptr<LuminosityBlockPrincipal> lbp) {
    std::auto_ptr<EventPrincipal> result;
    if (holder_.get() == 0) {
      setRunAndEventInfo();
      if (eventID_ == EventID()) {
        noMoreInput();
        return result; 
      }
      result = std::auto_ptr<EventPrincipal>(
	  new EventPrincipal(eventID_, Timestamp(presentTime_),
	  productRegistry(), lbp, processConfiguration(), isRealData_, eType_));
      if (newRun_ || newLumi_) {
	holder_ = result;
	return result;
      }
    } else {
      result = holder_;
    }
    Event e(*result, moduleDescription());
    if (!produce(e)) {
      noMoreInput();
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
    }
  }

  void
  ConfigurableInputSource::setLumi(LuminosityBlockNumber_t lb) {
    // Do nothing if the lumi block is not changed.
    if (lb != luminosityBlock_) {
      luminosityBlock_ = lb;
      numberEventsInThisLumi_ = 0;
      newLumi_ = true;
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
        ++luminosityBlock_;
	newLumi_ = true;
      }
    } else {
      // new run
      eventID_ = eventID_.nextRunFirstEvent();
      luminosityBlock_ = origLuminosityBlockNumber_t_;
      //reset these to one since this event is in the new run
      numberEventsInThisRun_ = 1;
      numberEventsInThisLumi_ = 1;
      newRun_ = newLumi_ = true;
    }
    presentTime_ += timeBetweenEvents_;
    if (eventCreationDelay_ > 0) {usleep(eventCreationDelay_);}
  }

}
