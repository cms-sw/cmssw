/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include "FWCore/Sources/interface/RawInputSource.h"

#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/MessageReceiverForSource.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {

  RawInputSource::RawInputSource(ParameterSet const& pset, InputSourceDescription const& desc) :
    InputSource(pset, desc),
    // The default value for the following parameter get defined in at least one derived class
    // where it has a different default value.
    inputFileTransitionsEachEvent_(pset.getUntrackedParameter<bool>("inputFileTransitionsEachEvent", false)),
    receiver_(),
    numberOfEventsBeforeBigSkip_(0) {
      setTimestamp(Timestamp::beginOfTime());
  }

  RawInputSource::~RawInputSource() {
  }

  boost::shared_ptr<RunAuxiliary>
  RawInputSource::readRunAuxiliary_() {
    assert(newRun());
    assert(runAuxiliary());
    resetNewRun();
    return runAuxiliary();
  }

  boost::shared_ptr<LuminosityBlockAuxiliary>
  RawInputSource::readLuminosityBlockAuxiliary_() {
    assert(!newRun());
    assert(newLumi());
    assert(luminosityBlockAuxiliary());
    resetNewLumi();
    return luminosityBlockAuxiliary();
  }

  EventPrincipal*
  RawInputSource::readEvent_(EventPrincipal& eventPrincipal) {
    assert(!newRun());
    assert(!newLumi());
    assert(eventCached());
    if(receiver_) {
      --numberOfEventsBeforeBigSkip_;
    }
    eventPrincipal.setLuminosityBlockPrincipal(luminosityBlockPrincipal());
    resetEventCached();
    return read(eventPrincipal);
  }

  EventPrincipal*
  RawInputSource::makeEvent(EventPrincipal& eventPrincipal, EventAuxiliary const& eventAuxiliary) {
    EventSourceSentry sentry(*this);
    eventPrincipal.fillEventPrincipal(eventAuxiliary, luminosityBlockPrincipal());
    return &eventPrincipal;
  }

  void
  RawInputSource::preForkReleaseResources() {
    closeFile(boost::shared_ptr<FileBlock>(), false);
  }

  void
  RawInputSource::postForkReacquireResources(boost::shared_ptr<edm::multicore::MessageReceiverForSource> iReceiver) {
    receiver_ = iReceiver;
    receiver_->receive();
    rewind();
  }

  InputSource::ItemType
  RawInputSource::getNextItemType() {
    if(receiver_ && 0 == numberOfEventsBeforeBigSkip_) {
      receiver_->receive();
      unsigned long toSkip = receiver_->numberToSkip();
      if(0 != toSkip) {
        skip(toSkip);
        decreaseRemainingEventsBy(toSkip);
        resetEventCached();
      }
      numberOfEventsBeforeBigSkip_ = receiver_->numberOfConsecutiveIndices();
      if(0 == numberOfEventsBeforeBigSkip_ or 0 == remainingEvents() or 0 == remainingLuminosityBlocks()) {
        return IsStop;
      }
    }
    if(state() == IsInvalid) {
      return IsFile;
    }
    if(newRun() && runAuxiliary()) {
      return IsRun;
    }
    if(newLumi() && luminosityBlockAuxiliary()) {
      return IsLumi;
    }
    if(eventCached()) {
      return IsEvent;
    }
    if (inputFileTransitionsEachEvent_) {
      resetRunAuxiliary(newRun());
      resetLuminosityBlockAuxiliary(newLumi());
    }
    bool another = checkNextEvent();
    if(!another || !eventCached()) {
      return IsStop;
    } else if(inputFileTransitionsEachEvent_) {
      return IsFile;
    }
    if(newRun()) {
      return IsRun;
    } else if(newLumi()) {
      return IsLumi;
    }
    return IsEvent;
  }

  void
  RawInputSource::reset_() {
    throw Exception(errors::LogicError)
      << "RawInputSource::reset()\n"
      << "Forking is not implemented for this type of RawInputSource\n"
      << "Contact a Framework Developer\n";
  }

  void
  RawInputSource::rewind_() {
    setNewRun();
    setNewLumi();
    resetEventCached();
    reset_();
    if(receiver_) {
      unsigned int numberToSkip = receiver_->numberToSkip();
      numberOfEventsBeforeBigSkip_ = receiver_->numberOfConsecutiveIndices();
      skip(numberToSkip);
      decreaseRemainingEventsBy(numberToSkip);
    }
    resetEventCached();
  }

  void
  RawInputSource:: fillDescription(ParameterSetDescription& description) {
    // The default value for "inputFileTransitionsEachEvent" gets defined in the derived class
    // as it depends on the derived class. So, we cannot redefine it here.
    InputSource::fillDescription(description);
  }
}
