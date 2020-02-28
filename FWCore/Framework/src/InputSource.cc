/*----------------------------------------------------------------------
----------------------------------------------------------------------*/
#include "FWCore/Framework/interface/InputSource.h"

#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/ExceptionHelpers.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/GlobalIdentifier.h"
#include "FWCore/Utilities/interface/TimeOfDay.h"

#include <cassert>
#include <fstream>
#include <iomanip>

namespace edm {

  namespace {
    std::string const& suffix(int count) {
      static std::string const st("st");
      static std::string const nd("nd");
      static std::string const rd("rd");
      static std::string const th("th");
      // *0, *4 - *9 use "th".
      int lastDigit = count % 10;
      if (lastDigit >= 4 || lastDigit == 0)
        return th;
      // *11, *12, or *13 use "th".
      if (count % 100 - lastDigit == 10)
        return th;
      return (lastDigit == 1 ? st : (lastDigit == 2 ? nd : rd));
    }
  }  // namespace

  InputSource::InputSource(ParameterSet const& pset, InputSourceDescription const& desc)
      : actReg_(desc.actReg_),
        maxEvents_(desc.maxEvents_),
        remainingEvents_(maxEvents_),
        maxLumis_(desc.maxLumis_),
        remainingLumis_(maxLumis_),
        readCount_(0),
        maxSecondsUntilRampdown_(desc.maxSecondsUntilRampdown_),
        processingMode_(RunsLumisAndEvents),
        moduleDescription_(desc.moduleDescription_),
        productRegistry_(desc.productRegistry_),
        processHistoryRegistry_(new ProcessHistoryRegistry),
        branchIDListHelper_(desc.branchIDListHelper_),
        thinnedAssociationsHelper_(desc.thinnedAssociationsHelper_),
        processGUID_(createGlobalIdentifier()),
        time_(),
        newRun_(true),
        newLumi_(true),
        eventCached_(false),
        state_(IsInvalid),
        runAuxiliary_(),
        lumiAuxiliary_(),
        statusFileName_(),
        numberOfEventsBeforeBigSkip_(0) {
    if (pset.getUntrackedParameter<bool>("writeStatusFile", false)) {
      std::ostringstream statusfilename;
      statusfilename << "source_" << getpid();
      statusFileName_ = statusfilename.str();
    }
    if (maxSecondsUntilRampdown_ > 0) {
      processingStart_ = std::chrono::steady_clock::now();
    }

    std::string const defaultMode("RunsLumisAndEvents");
    std::string const runMode("Runs");
    std::string const runLumiMode("RunsAndLumis");

    // The default value provided as the second argument to the getUntrackedParameter function call
    // is not used when the ParameterSet has been validated and the parameters are not optional
    // in the description.  As soon as all primary input sources and all modules with a secondary
    // input sources have defined descriptions, the defaults in the getUntrackedParameterSet function
    // calls can and should be deleted from the code.
    std::string processingMode = pset.getUntrackedParameter<std::string>("processingMode", defaultMode);
    if (processingMode == runMode) {
      processingMode_ = Runs;
    } else if (processingMode == runLumiMode) {
      processingMode_ = RunsAndLumis;
    } else if (processingMode != defaultMode) {
      throw Exception(errors::Configuration)
          << "InputSource::InputSource()\n"
          << "The 'processingMode' parameter for sources has an illegal value '" << processingMode << "'\n"
          << "Legal values are '" << defaultMode << "', '" << runLumiMode << "', or '" << runMode << "'.\n";
    }
  }

  InputSource::~InputSource() noexcept(false) {}

  void InputSource::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    desc.setUnknown();
    descriptions.addDefault(desc);
  }

  void InputSource::prevalidate(ConfigurationDescriptions&) {}

  static std::string const kBaseType("Source");

  std::string const& InputSource::baseType() { return kBaseType; }

  void InputSource::fillDescription(ParameterSetDescription& desc) {
    std::string defaultString("RunsLumisAndEvents");
    desc.addUntracked<std::string>("processingMode", defaultString)
        ->setComment(
            "'RunsLumisAndEvents': process runs, lumis, and events.\n"
            "'RunsAndLumis':       process runs and lumis (not events).\n"
            "'Runs':               process runs (not lumis or events).");
    desc.addUntracked<bool>("writeStatusFile", false)
        ->setComment("Write a status file. Intended for use by workflow management.");
  }

  // This next function is to guarantee that "runs only" mode does not return events or lumis,
  // and that "runs and lumis only" mode does not return events.
  // For input sources that are not random access (e.g. you need to read through the events
  // to get to the lumis and runs), this is all that is involved to implement these modes.
  // For input sources where events or lumis can be skipped, getNextItemType() should
  // implement the skipping internally, so that the performance gain is realized.
  // If this is done for a source, the 'if' blocks in this function will never be entered
  // for that source.
  InputSource::ItemType InputSource::nextItemType_() {
    ItemType itemType = callWithTryCatchAndPrint<ItemType>([this]() { return getNextItemType(); },
                                                           "Calling InputSource::getNextItemType");

    if (itemType == IsEvent && processingMode() != RunsLumisAndEvents) {
      skipEvents(1);
      return nextItemType_();
    }
    if (itemType == IsLumi && processingMode() == Runs) {
      // QQQ skipLuminosityBlock_();
      return nextItemType_();
    }
    return itemType;
  }

  InputSource::ItemType InputSource::nextItemType() {
    ItemType oldState = state_;
    if (eventLimitReached()) {
      // If the maximum event limit has been reached, stop.
      state_ = IsStop;
    } else if (lumiLimitReached()) {
      // If the maximum lumi limit has been reached, stop
      // when reaching a new file, run, or lumi.
      if (oldState == IsInvalid || oldState == IsFile || oldState == IsRun || processingMode() != RunsLumisAndEvents) {
        state_ = IsStop;
      } else {
        ItemType newState = nextItemType_();
        if (newState == IsEvent) {
          assert(processingMode() == RunsLumisAndEvents);
          state_ = IsEvent;
        } else {
          state_ = IsStop;
        }
      }
    } else {
      ItemType newState = nextItemType_();
      if (newState == IsStop) {
        state_ = IsStop;
      } else if (newState == IsSynchronize) {
        state_ = IsSynchronize;
      } else if (newState == IsFile || oldState == IsInvalid) {
        state_ = IsFile;
      } else if (newState == IsRun || oldState == IsFile) {
        runAuxiliary_ = readRunAuxiliary();
        state_ = IsRun;
      } else if (newState == IsLumi || oldState == IsRun) {
        assert(processingMode() != Runs);
        lumiAuxiliary_ = readLuminosityBlockAuxiliary();
        state_ = IsLumi;
      } else {
        assert(processingMode() == RunsLumisAndEvents);
        state_ = IsEvent;
      }
    }
    if (state_ == IsStop) {
      lumiAuxiliary_.reset();
      runAuxiliary_.reset();
    }
    return state_;
  }

  std::shared_ptr<LuminosityBlockAuxiliary> InputSource::readLuminosityBlockAuxiliary() {
    return callWithTryCatchAndPrint<std::shared_ptr<LuminosityBlockAuxiliary> >(
        [this]() { return readLuminosityBlockAuxiliary_(); }, "Calling InputSource::readLuminosityBlockAuxiliary_");
  }

  std::shared_ptr<RunAuxiliary> InputSource::readRunAuxiliary() {
    return callWithTryCatchAndPrint<std::shared_ptr<RunAuxiliary> >([this]() { return readRunAuxiliary_(); },
                                                                    "Calling InputSource::readRunAuxiliary_");
  }

  void InputSource::doBeginJob() { this->beginJob(); }

  void InputSource::doEndJob() { endJob(); }

  std::pair<SharedResourcesAcquirer*, std::recursive_mutex*> InputSource::resourceSharedWithDelayedReader() {
    return resourceSharedWithDelayedReader_();
  }

  std::pair<SharedResourcesAcquirer*, std::recursive_mutex*> InputSource::resourceSharedWithDelayedReader_() {
    return std::pair<SharedResourcesAcquirer*, std::recursive_mutex*>(nullptr, nullptr);
  }

  void InputSource::registerProducts() {}

  // Return a dummy file block.
  std::unique_ptr<FileBlock> InputSource::readFile() {
    assert(state_ == IsFile);
    assert(!limitReached());
    return callWithTryCatchAndPrint<std::unique_ptr<FileBlock> >([this]() { return readFile_(); },
                                                                 "Calling InputSource::readFile_");
  }

  void InputSource::closeFile(FileBlock* fb, bool cleaningUpAfterException) {
    if (fb != nullptr)
      fb->close();
    callWithTryCatchAndPrint<void>(
        [this]() { closeFile_(); }, "Calling InputSource::closeFile_", cleaningUpAfterException);
    return;
  }

  // Return a dummy file block.
  // This function must be overridden for any input source that reads a file
  // containing Products.
  std::unique_ptr<FileBlock> InputSource::readFile_() { return std::make_unique<FileBlock>(); }

  void InputSource::readRun(RunPrincipal& runPrincipal, HistoryAppender&) {
    RunSourceSentry sentry(*this, runPrincipal.index());
    callWithTryCatchAndPrint<void>([this, &runPrincipal]() { readRun_(runPrincipal); },
                                   "Calling InputSource::readRun_");
  }

  void InputSource::readAndMergeRun(RunPrincipal& rp) {
    RunSourceSentry sentry(*this, rp.index());
    callWithTryCatchAndPrint<void>([this, &rp]() { readRun_(rp); }, "Calling InputSource::readRun_");
  }

  void InputSource::readLuminosityBlock(LuminosityBlockPrincipal& lumiPrincipal, HistoryAppender&) {
    LumiSourceSentry sentry(*this, lumiPrincipal.index());
    callWithTryCatchAndPrint<void>([this, &lumiPrincipal]() { readLuminosityBlock_(lumiPrincipal); },
                                   "Calling InputSource::readLuminosityBlock_");
    if (remainingLumis_ > 0) {
      --remainingLumis_;
    }
  }

  void InputSource::readAndMergeLumi(LuminosityBlockPrincipal& lbp) {
    LumiSourceSentry sentry(*this, lbp.index());
    callWithTryCatchAndPrint<void>([this, &lbp]() { readLuminosityBlock_(lbp); },
                                   "Calling InputSource::readLuminosityBlock_");
    if (remainingLumis_ > 0) {
      --remainingLumis_;
    }
  }

  void InputSource::readRun_(RunPrincipal& runPrincipal) {
    // Note: For the moment, we do not support saving and restoring the state of the
    // random number generator if random numbers are generated during processing of runs
    // (e.g. beginRun(), endRun())
    runPrincipal.fillRunPrincipal(processHistoryRegistry());
  }

  void InputSource::readLuminosityBlock_(LuminosityBlockPrincipal& lumiPrincipal) {
    auto history = processHistoryRegistry().getMapped(lumiPrincipal.aux().processHistoryID());
    lumiPrincipal.fillLuminosityBlockPrincipal(history);
  }

  void InputSource::readEvent(EventPrincipal& ep, StreamContext& streamContext) {
    assert(state_ == IsEvent);
    assert(!eventLimitReached());
    {
      // block scope, in order to issue the PostSourceEvent signal before calling postRead and issueReports
      EventSourceSentry sentry(*this, streamContext);

      callWithTryCatchAndPrint<void>([this, &ep]() { readEvent_(ep); }, "Calling InputSource::readEvent_");
    }

    if (remainingEvents_ > 0)
      --remainingEvents_;
    ++readCount_;
    setTimestamp(ep.time());
    issueReports(ep.id(), ep.streamID());
  }

  bool InputSource::readEvent(EventPrincipal& ep, EventID const& eventID, StreamContext& streamContext) {
    bool result = false;

    if (not limitReached()) {
      // the Pre/PostSourceEvent signals should be generated only if the event is actually found.
      // this should be taken care of by an EventSourceSentry in the implementaion of readIt()

      //result = callWithTryCatchAndPrint<bool>( [this,&eventID,&ep](){ return readIt(eventID, ep); }, "Calling InputSource::readIt" );
      result = readIt(eventID, ep, streamContext);

      if (result) {
        if (remainingEvents_ > 0)
          --remainingEvents_;
        ++readCount_;
        issueReports(ep.id(), ep.streamID());
      }
    }
    return result;
  }

  void InputSource::skipEvents(int offset) {
    callWithTryCatchAndPrint<void>([this, &offset]() { skip(offset); }, "Calling InputSource::skip");
  }

  bool InputSource::goToEvent(EventID const& eventID) {
    return callWithTryCatchAndPrint<bool>([this, &eventID]() { return goToEvent_(eventID); },
                                          "Calling InputSource::goToEvent_");
  }

  void InputSource::rewind() {
    state_ = IsInvalid;
    remainingEvents_ = maxEvents_;
    setNewRun();
    setNewLumi();
    resetEventCached();
    callWithTryCatchAndPrint<void>([this]() { rewind_(); }, "Calling InputSource::rewind_");
  }

  void InputSource::issueReports(EventID const& eventID, StreamID streamID) {
    if (isInfoEnabled()) {
      LogVerbatim("FwkReport") << "Begin processing the " << readCount_ << suffix(readCount_) << " record. Run "
                               << eventID.run() << ", Event " << eventID.event() << ", LumiSection "
                               << eventID.luminosityBlock() << " on stream " << streamID.value() << " at "
                               << std::setprecision(3) << TimeOfDay();
    }
    if (!statusFileName_.empty()) {
      std::ofstream statusFile(statusFileName_.c_str());
      statusFile << eventID << " time: " << std::setprecision(3) << TimeOfDay() << '\n';
      statusFile.close();
    }

    // At some point we may want to initiate checkpointing here
  }

  bool InputSource::readIt(EventID const&, EventPrincipal&, StreamContext&) {
    throw Exception(errors::LogicError) << "InputSource::readIt()\n"
                                        << "Random access is not implemented for this type of Input Source\n"
                                        << "Contact a Framework Developer\n";
  }

  void InputSource::setRun(RunNumber_t) {
    throw Exception(errors::LogicError) << "InputSource::setRun()\n"
                                        << "Run number cannot be modified for this type of Input Source\n"
                                        << "Contact a Framework Developer\n";
  }

  void InputSource::setLumi(LuminosityBlockNumber_t) {
    throw Exception(errors::LogicError) << "InputSource::setLumi()\n"
                                        << "Luminosity Block ID cannot be modified for this type of Input Source\n"
                                        << "Contact a Framework Developer\n";
  }

  void InputSource::skip(int) {
    throw Exception(errors::LogicError) << "InputSource::skip()\n"
                                        << "Random access are not implemented for this type of Input Source\n"
                                        << "Contact a Framework Developer\n";
  }

  bool InputSource::goToEvent_(EventID const&) {
    throw Exception(errors::LogicError) << "InputSource::goToEvent_()\n"
                                        << "Random access is not implemented for this type of Input Source\n"
                                        << "Contact a Framework Developer\n";
    return true;
  }

  void InputSource::rewind_() {
    throw Exception(errors::LogicError) << "InputSource::rewind()\n"
                                        << "Random access are not implemented for this type of Input Source\n"
                                        << "Contact a Framework Developer\n";
  }

  void InputSource::decreaseRemainingEventsBy(int iSkipped) {
    if (-1 == remainingEvents_) {
      return;
    }
    if (iSkipped < remainingEvents_) {
      remainingEvents_ -= iSkipped;
    } else {
      remainingEvents_ = 0;
    }
  }

  void InputSource::doBeginRun(RunPrincipal& rp, ProcessContext const*) {}

  void InputSource::doBeginLumi(LuminosityBlockPrincipal& lbp, ProcessContext const*) {}

  bool InputSource::randomAccess() const {
    return callWithTryCatchAndPrint<bool>([this]() { return randomAccess_(); }, "Calling InputSource::randomAccess_");
  }

  ProcessingController::ForwardState InputSource::forwardState() const {
    return callWithTryCatchAndPrint<ProcessingController::ForwardState>([this]() { return forwardState_(); },
                                                                        "Calling InputSource::forwardState_");
  }

  ProcessingController::ReverseState InputSource::reverseState() const {
    return callWithTryCatchAndPrint<ProcessingController::ReverseState>([this]() { return reverseState_(); },
                                                                        "Calling InputSource::reverseState__");
  }

  void InputSource::beginJob() {}

  void InputSource::endJob() {}

  bool InputSource::randomAccess_() const { return false; }

  ProcessingController::ForwardState InputSource::forwardState_() const {
    return ProcessingController::kUnknownForward;
  }

  ProcessingController::ReverseState InputSource::reverseState_() const {
    return ProcessingController::kUnknownReverse;
  }

  ProcessHistoryID const& InputSource::reducedProcessHistoryID() const {
    assert(runAuxiliary());
    return processHistoryRegistry_->reducedProcessHistoryID(runAuxiliary()->processHistoryID());
  }

  RunNumber_t InputSource::run() const {
    assert(runAuxiliary());
    return runAuxiliary()->run();
  }

  LuminosityBlockNumber_t InputSource::luminosityBlock() const {
    assert(luminosityBlockAuxiliary());
    return luminosityBlockAuxiliary()->luminosityBlock();
  }

  InputSource::EventSourceSentry::EventSourceSentry(InputSource const& source, StreamContext& sc)
      : source_(source), sc_(sc) {
    source.actReg()->preSourceSignal_(sc_.streamID());
  }

  InputSource::EventSourceSentry::~EventSourceSentry() { source_.actReg()->postSourceSignal_(sc_.streamID()); }

  InputSource::LumiSourceSentry::LumiSourceSentry(InputSource const& source, LuminosityBlockIndex index)
      : source_(source), index_(index) {
    source_.actReg()->preSourceLumiSignal_(index_);
  }

  InputSource::LumiSourceSentry::~LumiSourceSentry() { source_.actReg()->postSourceLumiSignal_(index_); }

  InputSource::RunSourceSentry::RunSourceSentry(InputSource const& source, RunIndex index)
      : source_(source), index_(index) {
    source_.actReg()->preSourceRunSignal_(index_);
  }

  InputSource::RunSourceSentry::~RunSourceSentry() { source_.actReg()->postSourceRunSignal_(index_); }

  InputSource::FileOpenSentry::FileOpenSentry(InputSource const& source, std::string const& lfn, bool usedFallback)
      : post_(source.actReg()->postOpenFileSignal_), lfn_(lfn), usedFallback_(usedFallback) {
    source.actReg()->preOpenFileSignal_(lfn, usedFallback);
  }

  InputSource::FileOpenSentry::~FileOpenSentry() { post_(lfn_, usedFallback_); }

  InputSource::FileCloseSentry::FileCloseSentry(InputSource const& source, std::string const& lfn, bool usedFallback)
      : post_(source.actReg()->postCloseFileSignal_), lfn_(lfn), usedFallback_(usedFallback) {
    source.actReg()->preCloseFileSignal_(lfn, usedFallback);
  }

  InputSource::FileCloseSentry::~FileCloseSentry() { post_(lfn_, usedFallback_); }
}  // namespace edm
