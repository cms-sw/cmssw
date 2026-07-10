/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include "FWCore/Sources/interface/RawInputSource.h"

#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {

  RawInputSource::RawInputSource(ParameterSet const& pset, InputSourceDescription const& desc)
      : InputSource(pset, desc),
        // The default value for the following parameter get defined in at least one derived class
        // where it has a different default value.
        inputFileTransitionsEachEvent_(pset.getUntrackedParameter<bool>("inputFileTransitionsEachEvent", false)),
        fakeInputFileTransition_(false) {
    setTimestamp(Timestamp::beginOfTime());
  }

  RawInputSource::~RawInputSource() {}

  std::shared_ptr<RunAuxiliary> RawInputSource::readRunAuxiliary_() {
    assert(newRun());
    assert(runAuxiliary());
    resetNewRun();
    return runAuxiliary();
  }

  std::shared_ptr<LuminosityBlockAuxiliary> RawInputSource::readLuminosityBlockAuxiliary_() {
    assert(!newRun());
    assert(newLumi());
    assert(luminosityBlockAuxiliary());
    resetNewLumi();
    return luminosityBlockAuxiliary();
  }

  void RawInputSource::readEvent_(EventPrincipal& eventPrincipal) {
    assert(!newRun());
    assert(!newLumi());
    assert(eventCached());
    resetEventCached();
    read(eventPrincipal);
  }

  void RawInputSource::makeEvent(EventPrincipal& eventPrincipal, EventAuxiliary const& eventAuxiliary) {
    auto history = processHistoryRegistry().getMapped(eventAuxiliary.processHistoryID());
    eventPrincipal.fillEventPrincipal(eventAuxiliary, history);
  }

  InputSource::ItemTypeInfo RawInputSource::getNextItemType() {
    if (state().itemType() == ItemType::IsInvalid) {
      return ItemTypeInfo::isFile();
    }
    if (newRun() && runAuxiliary()) {
      return ItemTypeInfo::isRun();
    }
    if (newLumi() && luminosityBlockAuxiliary()) {
      return ItemTypeInfo::isLumi();
    }
    if (eventCached()) {
      return ItemTypeInfo::isEvent();
    }
    if (inputFileTransitionsEachEvent_) {
      // The following two lines are here because after a source
      // tells the state machine the next ItemType is IsFile,
      // the source should then always follow that with IsRun
      // and then IsLumi. These two lines will cause that to happen.
      resetRunAuxiliary(newRun());
      resetLuminosityBlockAuxiliary(newLumi());
    }
    Next another = checkNext();
    if (another == Next::kStop) {
      return ItemTypeInfo::isStop();
    } else if (another == Next::kEvent and inputFileTransitionsEachEvent_) {
      fakeInputFileTransition_ = true;
      return ItemTypeInfo::isFile();
    } else if (another == Next::kFile) {
      setNewRun();
      setNewLumi();
      resetEventCached();
      return ItemTypeInfo::isFile();
    }
    if (newRun()) {
      return ItemTypeInfo::isRun();
    } else if (newLumi()) {
      return ItemTypeInfo::isLumi();
    }
    return ItemTypeInfo::isEvent();
  }

  void RawInputSource::reset_() {
    throw Exception(errors::LogicError) << "RawInputSource::reset()\n"
                                        << "Forking is not implemented for this type of RawInputSource\n"
                                        << "Contact a Framework Developer\n";
  }

  void RawInputSource::rewind_() { reset_(); }

  void RawInputSource::fillDescription(ParameterSetDescription& description) {
    // The default value for "inputFileTransitionsEachEvent" gets defined in the derived class
    // as it depends on the derived class. So, we cannot redefine it here.
    InputSource::fillDescription(description);
  }

  void RawInputSource::closeFile_() {
    if (!fakeInputFileTransition_) {
      genuineCloseFile();
    }
  }

  std::shared_ptr<edm::FileBlock> RawInputSource::readFile_() {
    if (!fakeInputFileTransition_) {
      genuineReadFile();
    }
    fakeInputFileTransition_ = false;
    return std::make_shared<FileBlock>();
  }

}  // namespace edm
