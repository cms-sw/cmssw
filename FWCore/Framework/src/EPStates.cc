
#include "FWCore/Framework/src/EPStates.h"
#include "FWCore/Framework/interface/IEventProcessor.h"

#include <cassert>
#include <exception>
#include <sstream>
#include <string>

namespace statemachine {
  namespace {
    edm::RunNumber_t const INVALID_RUN_NUMBER = 0;
    edm::LuminosityBlockNumber_t const INVALID_LUMI = 0;
    Run const INVALID_RUN(edm::ProcessHistoryID(), INVALID_RUN_NUMBER);
    HandleLumis::LumiID const InvalidLumiID = HandleLumis::LumiID(edm::ProcessHistoryID(), INVALID_RUN_NUMBER, INVALID_LUMI);
  }

  Run::Run(edm::ProcessHistoryID const& phid, edm::RunNumber_t runNumber) :
    processHistoryID_(phid),
    runNumber_(runNumber) {
  }

  Lumi::Lumi(edm::LuminosityBlockNumber_t id) : id_(id) {}

  Machine::Machine(edm::IEventProcessor* ep,
                   FileMode fileMode,
                   EmptyRunLumiMode emptyRunLumiMode) :
    ep_(ep),
    fileMode_(fileMode),
    emptyRunLumiMode_(emptyRunLumiMode) {
  }

  edm::IEventProcessor& Machine::ep() const { return *ep_; }
  FileMode Machine::fileMode() const { return fileMode_; }
  EmptyRunLumiMode Machine::emptyRunLumiMode() const { return emptyRunLumiMode_; }

  void Machine::startingNewLoop(File const&) {
    ep_->startingNewLoop();
  }

  void Machine::startingNewLoop(Stop const&) {
    if(ep_->alreadyHandlingException()) return;
    ep_->startingNewLoop();
  }

  void Machine::rewindAndPrepareForNextLoop(Restart const&) {
    ep_->prepareForNextLoop();
    ep_->rewindInput();
  }

  Starting::Starting(my_context ctx) : my_base(ctx) { }

  Starting::~Starting() { }

  HandleFiles::HandleFiles(my_context ctx) :
    my_base(ctx),
    ep_(context<Machine>().ep()),
    exitCalled_(false) { }

  void HandleFiles::exit() {
    if(ep_.alreadyHandlingException()) return;
    exitCalled_ = true;
    closeFiles(false);
  }

  HandleFiles::~HandleFiles() {
    if(!exitCalled_) {
      try {
        closeFiles(true);
      }
      catch(...) {
        std::string message("Another exception was caught while trying to clean up files after the primary fatal exception.");
        ep_.setExceptionMessageFiles(message);
      }
    }
  }

  void HandleFiles::closeFiles(bool cleaningUpAfterException) {
    ep_.respondToCloseInputFile();
    ep_.closeInputFile(cleaningUpAfterException);
    ep_.closeOutputFiles();
  }

  void HandleFiles::goToNewInputFile() {
    ep_.respondToCloseInputFile();
    ep_.closeInputFile(false);

    ep_.readFile();
    ep_.respondToOpenInputFile();
  }

  bool HandleFiles::shouldWeCloseOutput() {
    if(context<Machine>().fileMode() == NOMERGE) return true;
    return ep_.shouldWeCloseOutput();
  }

  EndingLoop::EndingLoop(my_context ctx) :
      my_base(ctx),
      ep_(context<Machine>().ep()) {
    if(ep_.alreadyHandlingException() || ep_.endOfLoop()) post_event(Stop());
    else post_event(Restart());
  }

  EndingLoop::~EndingLoop() { }

  sc::result EndingLoop::react(Stop const&) {
    return terminate();
  }

  Error::Error(my_context ctx) :
      my_base(ctx),
      ep_(context<Machine>().ep()) {
    post_event(Stop());
    ep_.doErrorStuff();
  }

  Error::~Error() { }

  class HandleNewInputFile1;
  class NewInputAndOutputFiles;

  FirstFile::FirstFile(my_context ctx) :
      my_base(ctx),
      ep_(context<Machine>().ep()) {
    openFiles();
  }

  FirstFile::~FirstFile() { }

  sc::result FirstFile::react(File const&) {
    if(context<HandleFiles>().shouldWeCloseOutput()) {
      return transit<NewInputAndOutputFiles>();
    } else {
      return transit<HandleNewInputFile1>();
    }
  }

  void FirstFile::openFiles() {
    ep_.readFile();
    ep_.respondToOpenInputFile();

    ep_.openOutputFiles();
  }

  HandleNewInputFile1::HandleNewInputFile1(my_context ctx) :
      my_base(ctx) {
    context<HandleFiles>().goToNewInputFile();
  }

  HandleNewInputFile1::~HandleNewInputFile1() { }

  sc::result HandleNewInputFile1::react(File const&) {
    if(context<HandleFiles>().shouldWeCloseOutput()) {
      return transit<NewInputAndOutputFiles>();
    } else {
      return transit<HandleNewInputFile1>();
    }
  }

  NewInputAndOutputFiles::NewInputAndOutputFiles(my_context ctx) :
      my_base(ctx),
      ep_(context<Machine>().ep()) {
    goToNewInputAndOutputFiles();
  }

  NewInputAndOutputFiles::~NewInputAndOutputFiles() { }

  sc::result NewInputAndOutputFiles::react(File const&) {
    if(context<HandleFiles>().shouldWeCloseOutput()) {
      return transit<NewInputAndOutputFiles>();
    } else {
      return transit<HandleNewInputFile1>();
    }
  }

  void NewInputAndOutputFiles::goToNewInputAndOutputFiles() {
    ep_.respondToCloseInputFile();
    ep_.closeInputFile(false);

    ep_.closeOutputFiles();

    ep_.readFile();
    ep_.respondToOpenInputFile();

    ep_.openOutputFiles();
  }

  HandleRuns::HandleRuns(my_context ctx) :
    my_base(ctx),
    ep_(context<Machine>().ep()),
    exitCalled_(false),
    beginRunCalled_(false),
    currentRun_(INVALID_RUN),
    runException_(false) { }

  void HandleRuns::exit() {
    if(ep_.alreadyHandlingException()) return;
    exitCalled_ = true;
    finalizeRun(false);
  }

  HandleRuns::~HandleRuns() {
    if(!exitCalled_) {
      try {
        finalizeRun(true);
      }
      catch(...) {
        std::string message("Another exception was caught while trying to clean up runs after the primary fatal exception.");
        ep_.setExceptionMessageRuns(message);
      }
    }
  }

  bool HandleRuns::beginRunCalled() const { return beginRunCalled_; }
  Run const& HandleRuns::currentRun() const { return currentRun_; }
  bool HandleRuns::runException() const { return runException_; }

  void HandleRuns::setupCurrentRun() {

    runException_ = true;
    currentRun_ = ep_.readAndCacheRun();
    runException_ = false;

    if(context<Machine>().emptyRunLumiMode() != doNotHandleEmptyRunsAndLumis) {
      beginRun(currentRun());
    }
  }

  void HandleRuns::beginRun(Run const& run) {
    beginRunCalled_ = true;

    runException_ = true;
    ep_.beginRun(run);
    runException_ = false;
  }

  void HandleRuns::endRun(Run const& run, bool cleaningUpAfterException) {
    beginRunCalled_ = false;

    runException_ = true;
    ep_.endRun(run, cleaningUpAfterException);
    runException_ = false;
  }

  void HandleRuns::finalizeRun(Run const&) {
    finalizeRun(false);
  }

  void HandleRuns::finalizeRun(bool cleaningUpAfterException) {

    if(runException_) return;
    runException_ = true;

    if(beginRunCalled_) endRun(currentRun(), cleaningUpAfterException);
    ep_.writeRun(currentRun_);
    ep_.deleteRunFromCache(currentRun_);
    currentRun_ = INVALID_RUN;
    runException_ = false;
  }

  void HandleRuns::beginRunIfNotDoneAlready() {
    if(!beginRunCalled_) beginRun(currentRun());
  }

  NewRun::NewRun(my_context ctx) :
      my_base(ctx) {
    assert(context<HandleRuns>().currentRun() == INVALID_RUN);
    context<HandleRuns>().setupCurrentRun();

    // Here we assume that the input source or event processor
    // will throw if we fail to get a valid run.  Therefore
    // we should not ever fail this assert.
    assert(context<HandleRuns>().currentRun() != INVALID_RUN);
  }

  NewRun::~NewRun() { }

  sc::result NewRun::react(Run const& run) {
    if(run == context<HandleRuns>().currentRun()) {
      return transit<ContinueRun1>();
    }
    context<HandleRuns>().finalizeRun(false);
    return transit<NewRun>();
  }

  sc::result NewRun::react(File const&) {
    if(!context<HandleFiles>().shouldWeCloseOutput()) {
      return transit<HandleNewInputFile2>();
    }
    return forward_event();
  }

  HandleNewInputFile2::HandleNewInputFile2(my_context ctx) :
      my_base(ctx) {
    context<HandleFiles>().goToNewInputFile();
    checkInvariant();
  }

  HandleNewInputFile2::~HandleNewInputFile2() {
    checkInvariant();
  }

  bool HandleNewInputFile2::checkInvariant() {
    assert(context<HandleRuns>().currentRun() != INVALID_RUN);
    return true;
  }

  sc::result HandleNewInputFile2::react(Run const& run) {
    checkInvariant();

    if(context<HandleRuns>().currentRun() != run) {
      return transit<NewRun, HandleRuns, Run>(&HandleRuns::finalizeRun, run);
    } else {
      return transit<ContinueRun1>();
    }
  }

  sc::result HandleNewInputFile2::react(File const&) {
    checkInvariant();
    if(!context<HandleFiles>().shouldWeCloseOutput()) {
      return transit<HandleNewInputFile2>();
    }
    return forward_event();
  }

  ContinueRun1::ContinueRun1(my_context ctx) :
      my_base(ctx),
      ep_(context<Machine>().ep()) {
    ep_.readAndMergeRun();
    checkInvariant();
  }

  ContinueRun1::~ContinueRun1() {
    checkInvariant();
  }

  bool ContinueRun1::checkInvariant() {
    assert(context<HandleRuns>().currentRun() != INVALID_RUN);
    return true;
  }

  sc::result ContinueRun1::react(Run const& run) {
    checkInvariant();
    if(context<HandleRuns>().currentRun() != run) {
      return transit<NewRun, HandleRuns, Run>(&HandleRuns::finalizeRun, run);
    } else {
      return transit<ContinueRun1>();
    }
  }

  sc::result ContinueRun1::react(File const&) {
    checkInvariant();
    if(!context<HandleFiles>().shouldWeCloseOutput()) {
      return transit<HandleNewInputFile2>();
    }
    return forward_event();
  }

  HandleLumis::LumiID::LumiID(edm::ProcessHistoryID const& phid, edm::RunNumber_t run, edm::LuminosityBlockNumber_t lumi) :
    processHistoryID_(phid),
    run_(run),
    lumi_(lumi) {
  }

  HandleLumis::HandleLumis(my_context ctx) :
      my_base(ctx),
      ep_(context<Machine>().ep()),
      exitCalled_(false),
      currentLumiEmpty_(true),
      currentLumi_(InvalidLumiID),
      lumiException_(false) {
    checkInvariant();
  }

  void HandleLumis::exit() {
    if(ep_.alreadyHandlingException()) return;
    exitCalled_ = true;
    checkInvariant();
    if(!lumiException_ && !context<HandleRuns>().runException()) {
      finalizeLumi(false);
    }
  }

  HandleLumis::~HandleLumis() {
    if(!exitCalled_) {
      try {
        checkInvariant();
        if(!lumiException_ && !context<HandleRuns>().runException()) {
          finalizeLumi(true);
        }
      }
      catch(...) {
        std::string message("Another exception was caught while trying to clean up lumis after the primary fatal exception.");
        ep_.setExceptionMessageLumis(message);
      }
    }
  }

  bool HandleLumis::checkInvariant() {
    assert(context<HandleRuns>().currentRun() != INVALID_RUN);
    return true;
  }

  HandleLumis::LumiID const& HandleLumis::currentLumi() const { return currentLumi_; }

  bool HandleLumis::currentLumiEmpty() const { return currentLumiEmpty_; }

  void HandleLumis::setupCurrentLumi() {

    Run const& run = context<HandleRuns>().currentRun();
    assert(run != INVALID_RUN);
    lumiException_ = true;
    currentLumi_ = HandleLumis::LumiID(run.processHistoryID(), run.runNumber(), ep_.readAndCacheLumi());

    if(context<Machine>().emptyRunLumiMode() == handleEmptyRunsAndLumis) {
      assert(context<HandleRuns>().beginRunCalled());
      ep_.beginLumi(currentLumi().processHistoryID(), currentLumi().run(), currentLumi().lumi());
    }

    lumiException_ = false;

    currentLumiEmpty_ = true;
  }

  void HandleLumis::finalizeLumi(bool cleaningUpAfterException) {

    lumiException_ = true;

    if(!currentLumiEmpty_ ||
        context<Machine>().emptyRunLumiMode() == handleEmptyRunsAndLumis) {
      ep_.endLumi(currentLumi().processHistoryID(), currentLumi().run(), currentLumi().lumi(), cleaningUpAfterException);
    }

    ep_.writeLumi(currentLumi().processHistoryID(), currentLumi().run(), currentLumi().lumi());
    ep_.deleteLumiFromCache(currentLumi().processHistoryID(), currentLumi().run(), currentLumi().lumi());
    currentLumi_ = InvalidLumiID;

    lumiException_ = false;
  }

  void HandleLumis::markLumiNonEmpty() {
    if(currentLumiEmpty_) {

      if(context<Machine>().emptyRunLumiMode() != handleEmptyRunsAndLumis) {
        lumiException_ = true;
        ep_.beginLumi(currentLumi().processHistoryID(), currentLumi().run(), currentLumi().lumi());
        lumiException_ = false;
      }
      currentLumiEmpty_ = false;
    }
  }

  FirstLumi::FirstLumi(my_context ctx) :
      my_base(ctx) {
    context<HandleLumis>().setupCurrentLumi();
    checkInvariant();
  }

  FirstLumi::~FirstLumi() {
    checkInvariant();
  }

  bool FirstLumi::checkInvariant() {
    assert(context<HandleRuns>().currentRun() != INVALID_RUN);
    assert(context<HandleLumis>().currentLumi().processHistoryID() == context<HandleRuns>().currentRun().processHistoryID());
    assert(context<HandleLumis>().currentLumi().run() == context<HandleRuns>().currentRun().runNumber());
    assert(context<HandleLumis>().currentLumi().lumi() != INVALID_LUMI);
    assert(context<HandleLumis>().currentLumiEmpty() == true);
    return true;
  }

  sc::result FirstLumi::react(Lumi const& lumi) {
    if(lumi.id() == context<HandleLumis>().currentLumi().lumi()) {
      return transit<ContinueLumi>();
    }
    return transit<AnotherLumi>();
  }

  sc::result FirstLumi::react(File const&) {
    checkInvariant();
    if(!context<HandleFiles>().shouldWeCloseOutput()) {
      return transit<HandleNewInputFile3>();
    }
    return forward_event();
  }

  AnotherLumi::AnotherLumi(my_context ctx) :
      my_base(ctx) {
    context<HandleLumis>().finalizeLumi(false);
    context<HandleLumis>().setupCurrentLumi();
    checkInvariant();
  }

  AnotherLumi::~AnotherLumi() {
    checkInvariant();
  }

  bool AnotherLumi::checkInvariant() {
    assert(context<HandleRuns>().currentRun() != INVALID_RUN);
    assert(context<HandleLumis>().currentLumi().processHistoryID() == context<HandleRuns>().currentRun().processHistoryID());
    assert(context<HandleLumis>().currentLumi().run() == context<HandleRuns>().currentRun().runNumber());
    assert(context<HandleLumis>().currentLumi().lumi() != INVALID_LUMI);
    assert(context<HandleLumis>().currentLumiEmpty() == true);
    return true;
  }

  sc::result AnotherLumi::react(Lumi const& lumi) {
    if(lumi.id() == context<HandleLumis>().currentLumi().lumi()) {
      return transit<ContinueLumi>();
    }
    return transit<AnotherLumi>();
  }

  sc::result AnotherLumi::react(File const&) {
    checkInvariant();
    if(!context<HandleFiles>().shouldWeCloseOutput()) {
      return transit<HandleNewInputFile3>();
    }
    return forward_event();
  }

  HandleEvent::HandleEvent(my_context ctx) :
      my_base(ctx),
      ep_(context<Machine>().ep()) {
    readAndProcessEvent();
    checkInvariant();
  }

  HandleEvent::~HandleEvent() {
    checkInvariant();
  }

  bool HandleEvent::checkInvariant() {
    assert(context<HandleRuns>().currentRun() != INVALID_RUN);
    assert(context<HandleRuns>().beginRunCalled());
    assert(context<HandleLumis>().currentLumi().processHistoryID() == context<HandleRuns>().currentRun().processHistoryID());
    assert(context<HandleLumis>().currentLumi().run() == context<HandleRuns>().currentRun().runNumber());
    assert(context<HandleLumis>().currentLumi().lumi() != INVALID_LUMI);
    assert(context<HandleLumis>().currentLumiEmpty() == false);
    return true;
  }

  sc::result HandleEvent::react(File const&) {
    checkInvariant();
    if(!context<HandleFiles>().shouldWeCloseOutput()) {
      return transit<HandleNewInputFile3>();
    }
    return forward_event();
  }

  void HandleEvent::readAndProcessEvent() {
    markNonEmpty();
    ep_.readAndProcessEvent();
    if(ep_.shouldWeStop()) post_event(Stop());
  }

  void HandleEvent::markNonEmpty() {
    context<HandleRuns>().beginRunIfNotDoneAlready();
    context<HandleLumis>().markLumiNonEmpty();
  }


  HandleNewInputFile3::HandleNewInputFile3(my_context ctx) :
      my_base(ctx) {
    context<HandleFiles>().goToNewInputFile();
    checkInvariant();
  }

  HandleNewInputFile3::~HandleNewInputFile3() {
    checkInvariant();
  }

  bool HandleNewInputFile3::checkInvariant() {
    assert(context<HandleRuns>().currentRun() != INVALID_RUN);
    assert(context<HandleLumis>().currentLumi().processHistoryID() == context<HandleRuns>().currentRun().processHistoryID());
    assert(context<HandleLumis>().currentLumi().run() == context<HandleRuns>().currentRun().runNumber());
    assert(context<HandleLumis>().currentLumi().lumi() != INVALID_LUMI);
    return true;
  }

  sc::result HandleNewInputFile3::react(Run const& run) {
    checkInvariant();

    if(context<HandleRuns>().currentRun() == run) {
      return transit<ContinueRun2>();
    }
    return forward_event();
  }

  sc::result HandleNewInputFile3::react(File const&) {
    checkInvariant();
    if(!context<HandleFiles>().shouldWeCloseOutput()) {
      return transit<HandleNewInputFile3>();
    }
    return forward_event();
  }

  ContinueRun2::ContinueRun2(my_context ctx) :
      my_base(ctx),
      ep_(context<Machine>().ep()) {
    ep_.readAndMergeRun();
    checkInvariant();
  }

  ContinueRun2::~ContinueRun2() {
    checkInvariant();
  }

  bool ContinueRun2::checkInvariant() {
    assert(context<HandleRuns>().currentRun() != INVALID_RUN);
    assert(context<HandleLumis>().currentLumi().processHistoryID() == context<HandleRuns>().currentRun().processHistoryID());
    assert(context<HandleLumis>().currentLumi().run() == context<HandleRuns>().currentRun().runNumber());
    assert(context<HandleLumis>().currentLumi().lumi() != INVALID_LUMI);
    return true;
  }

  sc::result ContinueRun2::react(Run const& run) {
    checkInvariant();
    if(context<HandleRuns>().currentRun() != run) {
      return forward_event();
    } else {
      return transit<ContinueRun2>();
    }
  }

  sc::result ContinueRun2::react(Lumi const& lumi) {
    checkInvariant();

    if(context<HandleLumis>().currentLumi().lumi() != lumi.id()) {
      return transit<AnotherLumi>();
    } else {
      return transit<ContinueLumi>();
    }
  }

  sc::result ContinueRun2::react(File const&) {
    checkInvariant();
    if(!context<HandleFiles>().shouldWeCloseOutput()) {
      return transit<HandleNewInputFile3>();
    }
    return forward_event();
  }

  ContinueLumi::ContinueLumi(my_context ctx) :
    my_base(ctx),
    ep_(context<Machine>().ep()) {
    ep_.readAndMergeLumi();
    checkInvariant();
  }

  ContinueLumi::~ContinueLumi() {
    checkInvariant();
  }

  bool ContinueLumi::checkInvariant() {
    assert(context<HandleRuns>().currentRun() != INVALID_RUN);
    assert(context<HandleLumis>().currentLumi().processHistoryID() == context<HandleRuns>().currentRun().processHistoryID());
    assert(context<HandleLumis>().currentLumi().run() == context<HandleRuns>().currentRun().runNumber());
    assert(context<HandleLumis>().currentLumi().lumi() != INVALID_LUMI);
    return true;
  }

  sc::result ContinueLumi::react(Lumi const& lumi) {
    checkInvariant();
    if(context<HandleLumis>().currentLumi().lumi() != lumi.id()) {
      return transit<AnotherLumi>();
    } else {
      return transit<ContinueLumi>();
    }
  }

  sc::result ContinueLumi::react(File const&) {
    checkInvariant();
    if(!context<HandleFiles>().shouldWeCloseOutput()) {
      return transit<HandleNewInputFile3>();
    }
    return forward_event();
  }
}
