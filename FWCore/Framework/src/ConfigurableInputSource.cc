/*----------------------------------------------------------------------
$Id: ConfigurableInputSource.cc,v 1.15 2007/02/12 17:53:37 biery Exp $
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
				       InputSourceDescription const& desc) :
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
    luminosityBlockID_(pset.getUntrackedParameter<unsigned int>("firstLuminosityBlock", 1)),
    origLuminosityBlockID_(luminosityBlockID_),
    justBegun_(true),
    luminosityBlockPrincipal_()
  { }

  ConfigurableInputSource::~ConfigurableInputSource() {
  }

  void
  ConfigurableInputSource::startRun() {
    runPrincipal_ = boost::shared_ptr<RunPrincipal>(
	new RunPrincipal(eventID_.run(), productRegistry(), processConfiguration()));
    RunPrincipal & rp =
       const_cast<RunPrincipal &>(*runPrincipal_);
    Run run(rp, moduleDescription());
    beginRun(run);
    run.commit_();
  }

  void
  ConfigurableInputSource::startLumi() {
    luminosityBlockPrincipal_ = boost::shared_ptr<LuminosityBlockPrincipal>(
        new LuminosityBlockPrincipal(luminosityBlockID_, productRegistry(), runPrincipal_, processConfiguration()));
    LuminosityBlockPrincipal & lbp =
       const_cast<LuminosityBlockPrincipal &>(*luminosityBlockPrincipal_);
    LuminosityBlock lb(lbp, moduleDescription());
    beginLuminosityBlock(lb);
    lb.commit_();
  }

  void
  ConfigurableInputSource::finishRun() {
    RunPrincipal & rp =
        const_cast<RunPrincipal &>(*runPrincipal_);
    Run run(rp, moduleDescription());
    endRun(run);
    run.commit_();
  }

  void
  ConfigurableInputSource::finishLumi() {
    LuminosityBlockPrincipal & lbp =
        const_cast<LuminosityBlockPrincipal &>(*luminosityBlockPrincipal_);
    LuminosityBlock lb(lbp, moduleDescription());
    endLuminosityBlock(lb);
    lb.commit_();
  }

  void
  ConfigurableInputSource::endLumiAndRun() {
    if (luminosityBlockPrincipal_.get() == 0) {
      if (runPrincipal_.get() == 0) {
        startRun();
      }
      startLumi();
    }
    finishLumi();
    finishRun();
  }

  std::auto_ptr<EventPrincipal>
  ConfigurableInputSource::read() {
    RunNumber_t oldRun = eventID_.run();
    LuminosityBlockID oldLumi = luminosityBlockID_;
    setRunAndEventInfo();
    if (eventID_ == EventID()) {
      endLumiAndRun();
      return std::auto_ptr<EventPrincipal>(0); 
    }
    bool isNewRun = justBegun_ || oldRun != eventID_.run();
    bool isNewLumi = isNewRun || oldLumi != luminosityBlockID_;
    if(!justBegun_ && isNewLumi) {
      finishLumi();
      if (isNewRun) {
	finishRun();
      }
    }
    justBegun_ = false;
    if (isNewLumi) {
      if (isNewRun) {
        startRun();
      }
      startLumi();
    }
    std::auto_ptr<EventPrincipal> result = 
      std::auto_ptr<EventPrincipal>(
	  new EventPrincipal(eventID_, Timestamp(presentTime_),
	  productRegistry(), luminosityBlockPrincipal_, processConfiguration()));
    Event e(*result, moduleDescription());
    if (!produce(e)) {
      finishLumi();
      finishRun();
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
  ConfigurableInputSource::setRun(RunNumber_t r) {
    // Do nothing if the run is not changed.
    if (r != eventID_.run()) {
      eventID_ = EventID(r, zerothEvent_);
      luminosityBlockID_ = origLuminosityBlockID_;
      numberEventsInThisRun_ = 0;
      numberEventsInThisLumi_ = 0;
    }
  }

  void
  ConfigurableInputSource::setLumi(LuminosityBlockID lb) {
    // Do nothing if the lumi block is not changed.
    if (lb != luminosityBlockID_) {
      luminosityBlockID_ = lb;
      numberEventsInThisLumi_ = 0;
    }
  }

  void
  ConfigurableInputSource::rewind_() {
    if (!justBegun_) {
      finishLumi();
      finishRun();
      luminosityBlockID_ = origLuminosityBlockID_;
      presentTime_ = origTime_;
      eventID_ = origEventID_;
      numberEventsInThisRun_ = 0;
      numberEventsInThisLumi_ = 0;
      justBegun_ = true;      
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
    if (eventCreationDelay_ > 0) {usleep(eventCreationDelay_);}
  }

}
