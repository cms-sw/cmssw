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

  EventPrincipal*
  RawInputSource::readEvent_() {
    assert(!newRun());
    assert(!newLumi());
    assert(eventCached());
    resetEventCached();
    eventPrincipalCache()->setLuminosityBlockPrincipal(luminosityBlockPrincipal());
    return eventPrincipalCache();
  }

  EventPrincipal*
  RawInputSource::makeEvent(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event, Timestamp const& tstamp) {
    if(!runAuxiliary()) {
      setRunAuxiliary(new RunAuxiliary(run, tstamp, Timestamp::invalidTimestamp()));
    }
    if(!luminosityBlockAuxiliary()) {
      setLuminosityBlockAuxiliary(new LuminosityBlockAuxiliary(run, lumi, tstamp, Timestamp::invalidTimestamp()));
    }
    EventSourceSentry sentry(*this);
    EventAuxiliary aux(EventID(run, lumi, event), processGUID(), tstamp, true, EventAuxiliary::PhysicsTrigger);
    eventPrincipalCache()->fillEventPrincipal(aux, boost::shared_ptr<LuminosityBlockPrincipal>());
    setEventCached();
    return eventPrincipalCache();
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
    EventPrincipal *ep = read();
    if(ep == nullptr || !eventCached()) {
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

  EventPrincipal*
  RawInputSource::readIt(EventID const&) {
      throw Exception(errors::LogicError, "RawInputSource::readEvent_(EventID const& eventID)")
        << "Random access read cannot be used for RawInputSource.\n"
        << "Contact a Framework developer.\n";
  }

  // Not yet implemented
  void
  RawInputSource::skip(int) {
      throw Exception(errors::LogicError, "RawInputSource::skip(int offset)")
        << "Random access skip cannot be used for RawInputSource\n"
        << "Contact a Framework developer.\n";
  }


  void
  RawInputSource:: fillDescription(ParameterSetDescription& description) {
    // The default value for "inputFileTransitionsEachEvent" gets defined in the derived class
    // as it depends on the derived class. So, we cannot redefine it here.
    InputSource::fillDescription(description);
  }
}
