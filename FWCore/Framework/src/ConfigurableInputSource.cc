/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include <errno.h>

#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ConfigurableInputSource.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MessageReceiverForSource.h"

namespace edm {
  //used for defaults
  static unsigned long long const kNanoSecPerSec = 1000000000ULL;
  static unsigned long long const kAveEventPerSec = 200ULL;
  
  ConfigurableInputSource::ConfigurableInputSource(ParameterSet const& pset,
				       InputSourceDescription const& desc, bool realData) :
    InputSource(pset, desc),
    numberEventsInRun_(pset.getUntrackedParameter<unsigned int>("numberEventsInRun", remainingEvents())),
    numberEventsInLumi_(pset.getUntrackedParameter<unsigned int>("numberEventsInLuminosityBlock", remainingEvents())),
    presentTime_(pset.getUntrackedParameter<unsigned long long>("firstTime", 1ULL)),  //time in ns
    origTime_(presentTime_),
    timeBetweenEvents_(pset.getUntrackedParameter<unsigned long long>("timeBetweenEvents", kNanoSecPerSec/kAveEventPerSec)),
    eventCreationDelay_(pset.getUntrackedParameter<unsigned int>("eventCreationDelay", 0)),
    numberEventsInThisRun_(0),
    numberEventsInThisLumi_(0),
    zerothEvent_(pset.getUntrackedParameter<unsigned int>("firstEvent", 1) - 1),
    eventID_(pset.getUntrackedParameter<unsigned int>("firstRun", 1), pset.getUntrackedParameter<unsigned int>("firstLuminosityBlock", 1), zerothEvent_),
    origEventID_(eventID_),
    lumiSet_(false),
    eventSet_(false),
    isRealData_(realData),
    eType_(EventAuxiliary::Undefined),
    receiver_(),
    numberOfEventsBeforeBigSkip_(0)
  { 

    setTimestamp(Timestamp(presentTime_));
    // We need to map this string to the EventAuxiliary::ExperimentType enumeration
    // std::string eType = pset.getUntrackedParameter<std::string>("experimentType", std::string("Any"))),
  }

  ConfigurableInputSource::~ConfigurableInputSource() {
  }

  boost::shared_ptr<RunAuxiliary>
  ConfigurableInputSource::readRunAuxiliary_() {
    Timestamp ts = Timestamp(presentTime_);
    resetNewRun();
    return boost::shared_ptr<RunAuxiliary>(new RunAuxiliary(eventID_.run(), ts, Timestamp::invalidTimestamp()));
  }

  boost::shared_ptr<LuminosityBlockAuxiliary>
  ConfigurableInputSource::readLuminosityBlockAuxiliary_() {
    if (processingMode() == Runs) return boost::shared_ptr<LuminosityBlockAuxiliary>();
    Timestamp ts = Timestamp(presentTime_);
    resetNewLumi();
    return boost::shared_ptr<LuminosityBlockAuxiliary>(new LuminosityBlockAuxiliary(eventID_.run(), eventID_.luminosityBlock(), ts, Timestamp::invalidTimestamp()));
  }

  EventPrincipal *
  ConfigurableInputSource::readEvent_(EventPrincipal& eventPrincipal) {
    assert(eventCached() || processingMode() != RunsLumisAndEvents);
    resetEventCached();
    return eventPrincipalCache();
  }

  void
  ConfigurableInputSource::reallyReadEvent() {
    if (processingMode() != RunsLumisAndEvents) return;
    EventSourceSentry sentry(*this);
    EventAuxiliary aux(eventID_, processGUID(), Timestamp(presentTime_), isRealData_, eType_);
    eventPrincipalCache()->fillEventPrincipal(aux, luminosityBlockPrincipal());
    Event e(*eventPrincipalCache(), moduleDescription());
    if (!produce(e)) {
      resetEventCached();
      return;
    }
    e.commit_();
    setEventCached();
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
      setNewRun();
      setNewLumi();
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
      setNewLumi();
    }
    lumiSet_ = true;
  }

  void 
  ConfigurableInputSource::postForkReacquireResources(boost::shared_ptr<edm::multicore::MessageReceiverForSource> iReceiver) {
    receiver_ = iReceiver;
    receiver_->receive();
    numberOfEventsBeforeBigSkip_ = receiver_->numberOfConsecutiveIndices() + 1;
    repeat();
    rewind();
  }

  void
  ConfigurableInputSource::rewind_() {
    presentTime_ = origTime_;
    eventID_ = origEventID_;
    numberEventsInThisRun_ = 0;
    numberEventsInThisLumi_ = 0;

    if(receiver_) {
      unsigned int numberToSkip = receiver_->numberToSkip();
      numberOfEventsBeforeBigSkip_ = receiver_->numberOfConsecutiveIndices() + 1;
      //NOTE: skip() will decrease numberOfEventsBeforeBigSkip_ and therefore we need
      // to be sure it is large enough so that it never goes to 0 during the skipping
      if(numberOfEventsBeforeBigSkip_ < numberToSkip) {
        numberOfEventsBeforeBigSkip_ = numberToSkip+1;
      }
      skip(numberToSkip);
      decreaseRemainingEventsBy(numberToSkip);
      numberOfEventsBeforeBigSkip_ = receiver_->numberOfConsecutiveIndices() + 1;
    }
    setNewRun();
    setNewLumi();
  }
    
  InputSource::ItemType 
  ConfigurableInputSource::getNextItemType() {
    if (newRun()) {
      if (eventID_.run() == RunNumber_t()) {
        resetEventCached();
        return IsStop;
      }
      return IsRun;
    }
    if (newLumi()) {
      return IsLumi;
    }
    if(eventCached()) {
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
      resetEventCached();
      return IsStop;
    }
    if (oldEventID.run() != eventID_.run()) {
      //  New Run
      // If the user did not explicitly set the luminosity block number,
      // reset it back to the beginning.
      if (!lumiSet_) {
	eventID_.setLuminosityBlockNumber(origEventID_.luminosityBlock());
      }
      setNewRun();
      setNewLumi();
      return IsRun;
    }
      // Same Run
    if (oldLumi != eventID_.luminosityBlock()) {
      // New Lumi
      setNewLumi();
      if (processingMode() != Runs) {
        return IsLumi;
      }
    }
    reallyReadEvent();
    if(!eventCached()) {
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
    if(receiver_ && 0 == --numberOfEventsBeforeBigSkip_) {
      receiver_->receive();
      unsigned long numberOfEventsToSkip = receiver_->numberToSkip();
      if (numberOfEventsToSkip !=0) {
        skip(numberOfEventsToSkip);
        decreaseRemainingEventsBy(numberOfEventsToSkip);
      }
      numberOfEventsBeforeBigSkip_ = receiver_->numberOfConsecutiveIndices();
      //Since we decrease 'remaining events' count we need to see if we reached 0 and therefore are at the end
      if(0 == numberOfEventsBeforeBigSkip_ or 0==remainingEvents() or 0 == remainingLuminosityBlocks()) {
        //this means we are to stop
        eventID_ = EventID();
        return;
      }
    }
    advanceToNext();
    if (eventCreationDelay_ > 0) {usleep(eventCreationDelay_);}
  }

  void
  ConfigurableInputSource::fillDescription(ParameterSetDescription& desc) {
    desc.addOptionalUntracked<unsigned int>("numberEventsInRun")->setComment("Number of events to generate in each run.");
    desc.addOptionalUntracked<unsigned int>("numberEventsInLuminosityBlock")->setComment("Number of events to generate in each lumi.");
    desc.addUntracked<unsigned long long>("firstTime", 1)->setComment("Time before first event (ns) (for timestamp).");
    desc.addUntracked<unsigned long long>("timeBetweenEvents", kNanoSecPerSec/kAveEventPerSec)->setComment("Time between consecutive events (ns) (for timestamp).");
    desc.addUntracked<unsigned int>("eventCreationDelay", 0)->setComment("Real time delay between generation of consecutive events (ms).");
    desc.addUntracked<unsigned int>("firstEvent", 1)->setComment("Event number of first event to generate.");
    desc.addUntracked<unsigned int>("firstLuminosityBlock", 1)->setComment("Luminosity block number of first lumi to generate.");
    desc.addUntracked<unsigned int>("firstRun", 1)->setComment("Run number of first run to generate.");
    InputSource::fillDescription(desc);
  }

}
