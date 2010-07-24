/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ConfigurableInputSource.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Event.h"

namespace edm {
  //used for defaults
  static unsigned int const kNanoSecPerSec = 1000000000U;
  static unsigned int const kAveEventPerSec = 200U;
  
  ConfigurableInputSource::ConfigurableInputSource(ParameterSet const& pset,
				       InputSourceDescription const& desc, bool realData) :
    InputSource(pset, desc),
    numberEventsInRun_(pset.getUntrackedParameter<unsigned int>("numberEventsInRun", remainingEvents())),
    numberEventsInLumi_(pset.getUntrackedParameter<unsigned int>("numberEventsInLuminosityBlock", remainingEvents())),
    presentTime_(pset.getUntrackedParameter<unsigned int>("firstTime", 1)),  //time in ns
    origTime_(presentTime_),
    timeBetweenEvents_(pset.getUntrackedParameter<unsigned int>("timeBetweenEvents", kNanoSecPerSec/kAveEventPerSec)),
    eventCreationDelay_(pset.getUntrackedParameter<unsigned int>("eventCreationDelay", 0)),
    numberEventsInThisRun_(0),
    numberEventsInThisLumi_(0),
    zerothEvent_(pset.getUntrackedParameter<unsigned int>("firstEvent", 1) - 1),
    eventID_(pset.getUntrackedParameter<unsigned int>("firstRun", 1), pset.getUntrackedParameter<unsigned int>("firstLuminosityBlock", 1), zerothEvent_),
    origEventID_(eventID_),
    newRun_(true),
    newLumi_(true),
    eventCached_(false),
    lumiSet_(false),
    eventSet_(false),
    isRealData_(realData),
    eType_(EventAuxiliary::Undefined),
    numberOfEventsBeforeBigSkip_(0),
    numberOfEventsInBigSkip_(0),
    numberOfSequentialEvents_(0),
    forkedChildIndex_(0) { 

    setTimestamp(Timestamp(presentTime_));
    // We need to map this string to the EventAuxiliary::ExperimentType enumeration
    // std::string eType = pset.getUntrackedParameter<std::string>("experimentType", std::string("Any"))),
  }

  ConfigurableInputSource::~ConfigurableInputSource() {
  }

  boost::shared_ptr<RunAuxiliary>
  ConfigurableInputSource::readRunAuxiliary_() {
    Timestamp ts = Timestamp(presentTime_);
    newRun_ = false;
    return boost::shared_ptr<RunAuxiliary>(new RunAuxiliary(eventID_.run(), ts, Timestamp::invalidTimestamp()));
  }

  boost::shared_ptr<LuminosityBlockAuxiliary>
  ConfigurableInputSource::readLuminosityBlockAuxiliary_() {
    if (processingMode() == Runs) return boost::shared_ptr<LuminosityBlockAuxiliary>();
    Timestamp ts = Timestamp(presentTime_);
    newLumi_ = false;
    return boost::shared_ptr<LuminosityBlockAuxiliary>(new LuminosityBlockAuxiliary(eventID_.run(), eventID_.luminosityBlock(), ts, Timestamp::invalidTimestamp()));
  }

  EventPrincipal *
  ConfigurableInputSource::readEvent_() {
    assert(eventCached_ || processingMode() != RunsLumisAndEvents);
    eventCached_ = false;
    return eventPrincipalCache();
  }

  void
  ConfigurableInputSource::reallyReadEvent() {
    if (processingMode() != RunsLumisAndEvents) return;
    EventSourceSentry sentry(*this);
    std::auto_ptr<EventAuxiliary> aux(new EventAuxiliary(eventID_, processGUID(), Timestamp(presentTime_), isRealData_, eType_));
    eventPrincipalCache()->fillEventPrincipal(aux, luminosityBlockPrincipal());
    Event e(*eventPrincipalCache(), moduleDescription());
    if (!produce(e)) {
      eventCached_ = false;
      return;
    }
    e.commit_();
    eventCached_ = true;
  }

  void
  ConfigurableInputSource::skip(int offset) {
    for (; offset < 0; ++offset) {
      retreatToPrevious();
    }
    for (; offset > 0; --offset) {
      advanceToNext();
    }
  }


  void
  ConfigurableInputSource::setRun(RunNumber_t r) {
    // No need to check for invalid (zero) run number,
    // as this is a legitimate way of stopping the job.
    // Do nothing if the run is not changed.
    if (r != eventID_.run()) {
      eventID_ = EventID(r, origEventID_.luminosityBlock(), zerothEvent_);
      numberEventsInThisRun_ = 0;
      numberEventsInThisLumi_ = 0;
      newRun_ = newLumi_ = true;
    }
  }

  void
  ConfigurableInputSource::beginRun(Run&) {
  }

  void
  ConfigurableInputSource::endRun(Run&) {
  }

  void
  ConfigurableInputSource::beginLuminosityBlock(LuminosityBlock&) {
  }

  void
  ConfigurableInputSource::endLuminosityBlock(LuminosityBlock&) {
  }

  void
  ConfigurableInputSource::setLumi(LuminosityBlockNumber_t lb) {
    // Protect against invalid lumi.
    if (lb == LuminosityBlockNumber_t()) {
	lb = origEventID_.luminosityBlock();
    }
    // Do nothing if the lumi block is not changed.
    if (lb != eventID_.luminosityBlock()) {
      eventID_.setLuminosityBlockNumber(lb);
      numberEventsInThisLumi_ = 0;
      newLumi_ = true;
    }
    lumiSet_ = true;
  }

  void 
  ConfigurableInputSource::postForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren, unsigned int iNumberOfSequentialEvents) {
    numberOfEventsInBigSkip_ = iNumberOfSequentialEvents*(iNumberOfChildren-1);
    numberOfEventsBeforeBigSkip_ = iNumberOfSequentialEvents + 1;
    forkedChildIndex_ = iChildIndex;
    numberOfSequentialEvents_ = iNumberOfSequentialEvents;
    repeat();
    rewind();
  }

  void
  ConfigurableInputSource::rewind_() {
    presentTime_ = origTime_;
    eventID_ = origEventID_;
    numberEventsInThisRun_ = 0;
    numberEventsInThisLumi_ = 0;

    unsigned int numberToSkip = numberOfSequentialEvents_*forkedChildIndex_;
    numberOfEventsBeforeBigSkip_ = numberOfSequentialEvents_ + 1;
    if(numberOfEventsBeforeBigSkip_ < numberToSkip) {
      numberOfEventsBeforeBigSkip_ = numberToSkip+1;
    }
    skip(numberToSkip);
    numberOfEventsBeforeBigSkip_ = numberOfSequentialEvents_ + 1;
    newRun_ = newLumi_ = true;
  }
    
  InputSource::ItemType 
  ConfigurableInputSource::getNextItemType() {
    if (newRun_) {
      if (eventID_.run() == RunNumber_t()) {
        eventCached_ = false;
        return IsStop;
      }
      return IsRun;
    }
    if (newLumi_) {
      return IsLumi;
    }
    if(eventCached_) {
      return IsEvent;
    }
    EventID oldEventID = eventID_;
    LuminosityBlockNumber_t oldLumi = eventID_.luminosityBlock();
    if (!eventSet_) {
      lumiSet_ = false;
      setRunAndEventInfo();
      eventSet_ = true;
    }
    if (eventID_.run() == RunNumber_t()) {
      eventCached_ = false;
      return IsStop;
    }
    if (oldEventID.run() != eventID_.run()) {
      //  New Run
      // If the user did not explicitly set the luminosity block number,
      // reset it back to the beginning.
      if (!lumiSet_) {
	eventID_.setLuminosityBlockNumber(origEventID_.luminosityBlock());
      }
      newRun_ = newLumi_ = true;
      return IsRun;
    }
      // Same Run
    if (oldLumi != eventID_.luminosityBlock()) {
      // New Lumi
      newLumi_ = true;
      if (processingMode() != Runs) {
        return IsLumi;
      }
    }
    reallyReadEvent();
    if(!eventCached_) {
      return IsStop;
    }
    eventSet_ = false;
    return IsEvent;
  }

  void 
  ConfigurableInputSource::advanceToNext()  {
    if (numberEventsInRun_ < 1 || numberEventsInThisRun_ < numberEventsInRun_) {
      // same run
      ++numberEventsInThisRun_;
      if (!(numberEventsInLumi_ < 1 || numberEventsInThisLumi_ < numberEventsInLumi_)) {
        // new lumi
        eventID_ = eventID_.next(eventID_.luminosityBlock() + 1);
        numberEventsInThisLumi_ = 1;
      } else {
        eventID_ = eventID_.next(eventID_.luminosityBlock());
        ++numberEventsInThisLumi_;
      }
    } else {
      // new run
      eventID_ = eventID_.nextRunFirstEvent(origEventID_.luminosityBlock());
      numberEventsInThisLumi_ = 1;
      numberEventsInThisRun_ = 1;
    }
    presentTime_ += timeBetweenEvents_;
  }

  void 
  ConfigurableInputSource::retreatToPrevious()  {
    if (numberEventsInRun_ < 1 || numberEventsInThisRun_ > 0) {
      // same run
      --numberEventsInThisRun_;
      eventID_ = eventID_.previous(eventID_.luminosityBlock());
      if (!(numberEventsInLumi_ < 1 || numberEventsInThisLumi_ > 0)) {
        // new lumi
        eventID_ = eventID_.previous(eventID_.luminosityBlock() - 1);
        numberEventsInThisLumi_ = numberEventsInLumi_;
      } else {
        --numberEventsInThisLumi_;
      }
    } else {
      // new run
      eventID_ = eventID_.previousRunLastEvent(origEventID_.luminosityBlock() + numberEventsInRun_/numberEventsInLumi_);
      eventID_ = EventID(numberEventsInRun_, eventID_.luminosityBlock(), eventID_.run());
      numberEventsInThisLumi_ = numberEventsInLumi_;
      numberEventsInThisRun_ = numberEventsInRun_;
    }
    presentTime_ -= timeBetweenEvents_;
  }
  
  void
  ConfigurableInputSource::setRunAndEventInfo() {
    if(0 != numberOfEventsInBigSkip_ && 0 == --numberOfEventsBeforeBigSkip_) {
      skip(numberOfEventsInBigSkip_);
      numberOfEventsBeforeBigSkip_ = numberOfSequentialEvents_;
    }
    advanceToNext();
    if (eventCreationDelay_ > 0) {usleep(eventCreationDelay_);}
  }

  void
  ConfigurableInputSource::fillDescription(ParameterSetDescription& desc) {
    desc.addOptionalUntracked<unsigned int>("numberEventsInRun");
    desc.addOptionalUntracked<unsigned int>("numberEventsInLuminosityBlock");
    desc.addUntracked<unsigned int>("firstTime", 1);
    desc.addUntracked<unsigned int>("timeBetweenEvents", kNanoSecPerSec/kAveEventPerSec);
    desc.addUntracked<unsigned int>("eventCreationDelay", 0);
    desc.addUntracked<unsigned int>("firstEvent", 1);
    desc.addUntracked<unsigned int>("firstLuminosityBlock", 1);
    desc.addUntracked<unsigned int>("firstRun", 1);
    InputSource::fillDescription(desc);
  }

}
