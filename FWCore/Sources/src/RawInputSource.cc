/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include "FWCore/Sources/interface/RawInputSource.h"

#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {

  RawInputSource::RawInputSource(ParameterSet const& pset, InputSourceDescription const& desc) :
    InputSource(pset, desc),
    // The default value for the following parameter get defined in at least one derived class
    // where it has a different default value.
    inputFileTransitionsEachEvent_(pset.getUntrackedParameter<bool>("inputFileTransitionsEachEvent", false)) {
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

  void
  RawInputSource::readEvent_(EventPrincipal& eventPrincipal) {
    assert(!newRun());
    assert(!newLumi());
    assert(eventCached());
    resetEventCached();
    read(eventPrincipal);
  }

  void
  RawInputSource::makeEvent(EventPrincipal& eventPrincipal, EventAuxiliary const& eventAuxiliary) {
    eventPrincipal.fillEventPrincipal(eventAuxiliary, processHistoryRegistry());
  }

  void
  RawInputSource::preForkReleaseResources() {
    closeFile(nullptr, false);
  }

  InputSource::ItemType
  RawInputSource::getNextItemType() {
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
    if(!another || (!newLumi() && !eventCached())) {
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
    reset_();
  }

  void
  RawInputSource:: fillDescription(ParameterSetDescription& description) {
    // The default value for "inputFileTransitionsEachEvent" gets defined in the derived class
    // as it depends on the derived class. So, we cannot redefine it here.
    InputSource::fillDescription(description);
  }
}
