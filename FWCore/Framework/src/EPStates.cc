
#include "FWCore/Framework/src/EPStates.h"
#include "FWCore/Framework/interface/IEventProcessor.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <exception>
#include <sstream>
#include <string>
#include <cassert>

namespace statemachine {
  namespace {
    int const INVALID_RUN_NUMBER = 0;
    int const INVALID_LUMI = 0;
    Run const INVALID_RUN(edm::ProcessHistoryID(), INVALID_RUN_NUMBER);
    HandleLumis::LumiID const InvalidLumiID = HandleLumis::LumiID(edm::ProcessHistoryID(), INVALID_RUN_NUMBER, INVALID_LUMI);
  }

  Run::Run(edm::ProcessHistoryID const& phid, int runNumber) :
    processHistoryID_(phid),
    runNumber_(runNumber) {
  }

  Lumi::Lumi(int id) : id_(id) {}

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

  void Machine::startingNewLoop(File const& file) {
    ep_->startingNewLoop();
  }

  void Machine::startingNewLoop(Stop const& stop) {
    if (ep_->alreadyHandlingException()) return;
    ep_->startingNewLoop();
  }

  void Machine::rewindAndPrepareForNextLoop(Restart const& restart) {
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
    if (ep_.alreadyHandlingException()) return;
    exitCalled_ = true;
    closeFiles();
  }

  HandleFiles::~HandleFiles() {
    if (!exitCalled_) {
      try {
        closeFiles();
      }
      catch (cms::Exception& e) {
        std::ostringstream message;
        message << "------------------------------------------------------------\n"
                << "Another exception was caught while trying to clean up files after\n" 
                << "the primary exception.  We give up trying to clean up files at\n"
                << "this point.  The description of this additional exception follows:\n" 
                << "cms::Exception\n"
                << e.explainSelf();
        std::string msg(message.str());
        ep_.setExceptionMessageFiles(msg);
      }
      catch (std::bad_alloc& e) {
        std::ostringstream message;
        message << "------------------------------------------------------------\n"
                << "Another exception was caught while trying to clean up files\n" 
                << "after the primary exception.  We give up trying to clean up files\n"
                << "at this point.  This additional exception was a\n" 
                << "std::bad_alloc exception thrown inside HandleFiles::closeFiles.\n"
                << "The job has probably exhausted the virtual memory available\n"
                << "to the process.\n";
        std::string msg(message.str());
        ep_.setExceptionMessageFiles(msg);
      }
      catch (std::exception& e) {
        std::ostringstream message;
        message << "------------------------------------------------------------\n"
                << "Another exception was caught while trying to clean up files after\n" 
                << "the primary exception.  We give up trying to clean up files at\n"
                << "this point.  This additional exception was a\n" 
                << "standard library exception thrown inside HandleFiles::closeFiles\n"
                << e.what() << "\n";
        std::string msg(message.str());
        ep_.setExceptionMessageFiles(msg);
      }
      catch (...) {
        std::ostringstream message;
        message << "------------------------------------------------------------\n"
                << "Another exception was caught while trying to clean up files after\n" 
                << "the primary exception.  We give up trying to clean up files at\n"
                << "this point.  This additional exception was of unknown type and\n" 
                << "thrown inside HandleFiles::closeFiles\n";
        std::string msg(message.str());
        ep_.setExceptionMessageFiles(msg);
      }
    }
  }

  void HandleFiles::closeFiles() {
    ep_.respondToCloseInputFile();
    ep_.closeInputFile();
    ep_.respondToCloseOutputFiles();
    ep_.closeOutputFiles();
  }

  void HandleFiles::goToNewInputFile() {
    ep_.respondToCloseInputFile();
    ep_.closeInputFile();

    ep_.readFile();
    ep_.respondToOpenInputFile();
  }

  bool HandleFiles::shouldWeCloseOutput() {
    if (context<Machine>().fileMode() == NOMERGE) return true;
    return ep_.shouldWeCloseOutput();
  }

  EndingLoop::EndingLoop(my_context ctx) : 
    my_base(ctx),
    ep_(context<Machine>().ep())
  { 
    if (ep_.alreadyHandlingException() || ep_.endOfLoop()) post_event(Stop());
    else post_event(Restart());
  }

  EndingLoop::~EndingLoop() { }

  sc::result EndingLoop::react(Stop const&)
  {
    return terminate();
  }

  Error::Error(my_context ctx) : 
    my_base(ctx),
    ep_(context<Machine>().ep())
  { 
    post_event(Stop());
    ep_.doErrorStuff();
  }

  Error::~Error() { }

  class HandleNewInputFile1;
  class NewInputAndOutputFiles;

  FirstFile::FirstFile(my_context ctx) :
    my_base(ctx),
    ep_(context<Machine>().ep())
  { 
    openFiles();
  }

  FirstFile::~FirstFile() { }

  sc::result FirstFile::react(File const& file)
  {
    if (context<HandleFiles>().shouldWeCloseOutput()) {
      return transit<NewInputAndOutputFiles>();
    }
    else {
      return transit<HandleNewInputFile1>();
    }
  }

  void FirstFile::openFiles() {
    ep_.readFile();
    ep_.respondToOpenInputFile();

    ep_.openOutputFiles();
    ep_.respondToOpenOutputFiles();
  }

  HandleNewInputFile1::HandleNewInputFile1(my_context ctx) : 
    my_base(ctx)
  { 
    context<HandleFiles>().goToNewInputFile();
  }

  HandleNewInputFile1::~HandleNewInputFile1() { }

  sc::result HandleNewInputFile1::react(File const& file)
  {
    if (context<HandleFiles>().shouldWeCloseOutput()) {
      return transit<NewInputAndOutputFiles>();
    }
    else {
      return transit<HandleNewInputFile1>();
    }
  }

  NewInputAndOutputFiles::NewInputAndOutputFiles(my_context ctx) : 
    my_base(ctx),
    ep_(context<Machine>().ep())
  { 
    goToNewInputAndOutputFiles();
  }

  NewInputAndOutputFiles::~NewInputAndOutputFiles() { }

  sc::result NewInputAndOutputFiles::react(File const& file)
  {
    if (context<HandleFiles>().shouldWeCloseOutput()) {
      return transit<NewInputAndOutputFiles>();
    }
    else {
      return transit<HandleNewInputFile1>();
    }
  }

  void NewInputAndOutputFiles::goToNewInputAndOutputFiles() {
    ep_.respondToCloseInputFile();
    ep_.closeInputFile();

    ep_.respondToCloseOutputFiles();
    ep_.closeOutputFiles();

    ep_.readFile();
    ep_.respondToOpenInputFile();

    ep_.openOutputFiles();
    ep_.respondToOpenOutputFiles();
  }

  HandleRuns::HandleRuns(my_context ctx) : 
    my_base(ctx),
    ep_(context<Machine>().ep()),
    exitCalled_(false),
    beginRunCalled_(false),
    currentRun_(INVALID_RUN),
    runException_(false) { }

  void HandleRuns::exit() {
    if (ep_.alreadyHandlingException()) return;
    exitCalled_ = true;
    finalizeRun();
  }

  HandleRuns::~HandleRuns() {
    if (!exitCalled_) {
      try {
        finalizeRun();
      }
      catch (cms::Exception& e) {
        std::ostringstream message;
        message << "------------------------------------------------------------\n"
                << "Another exception was caught while trying to clean up runs after\n" 
                << "the primary exception.  We give up trying to clean up runs at\n"
                << "this point.  The description of this additional exception follows:\n" 
                << "cms::Exception\n"
                << e.explainSelf();
        std::string msg(message.str());
        ep_.setExceptionMessageRuns(msg);
      }
      catch (std::bad_alloc& e) {
        std::ostringstream message;
        message << "------------------------------------------------------------\n"
                << "Another exception was caught while trying to clean up runs\n" 
                << "after the primary exception.  We give up trying to clean up runs\n"
                << "at this point.  This additional exception was a\n" 
                << "std::bad_alloc exception thrown inside HandleRuns::finalizeRun.\n"
                << "The job has probably exhausted the virtual memory available\n"
                << "to the process.\n";
        std::string msg(message.str());
        ep_.setExceptionMessageRuns(msg);
      }
      catch (std::exception& e) {
        std::ostringstream message;
        message << "------------------------------------------------------------\n"
                << "Another exception was caught while trying to clean up runs after\n" 
                << "the primary exception.  We give up trying to clean up runs at\n"
                << "this point.  This additional exception was a\n" 
                << "standard library exception thrown inside HandleRuns::finalizeRun\n"
                << e.what() << "\n";
        std::string msg(message.str());
        ep_.setExceptionMessageRuns(msg);
      }
      catch (...) {
        std::ostringstream message;
        message << "------------------------------------------------------------\n"
                << "Another exception was caught while trying to clean up runs after\n" 
                << "the primary exception.  We give up trying to clean up runs at\n"
                << "this point.  This additional exception was of unknown type and\n" 
                << "thrown inside HandleRuns::finalizeRun\n";
        std::string msg(message.str());
        ep_.setExceptionMessageRuns(msg);
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

    if (context<Machine>().emptyRunLumiMode() != doNotHandleEmptyRunsAndLumis) {
      beginRun(currentRun());
    }
  }

  void HandleRuns::beginRun(Run const& run) {
    beginRunCalled_ = true;

    runException_ = true;
    ep_.beginRun(run);
    runException_ = false;
  }

  void HandleRuns::endRun(Run const& run) {
    beginRunCalled_ = false;

    runException_ = true;
    ep_.endRun(run);
    runException_ = false;
  }

  void HandleRuns::finalizeRun(Run const&) {
    finalizeRun();
  }

  void HandleRuns::finalizeRun() {

    if (runException_) return;
    runException_ = true;

    if (beginRunCalled_) endRun(currentRun());
    ep_.writeRun(currentRun_);
    ep_.deleteRunFromCache(currentRun_);
    currentRun_ = INVALID_RUN;
    runException_ = false;   
  }

  void HandleRuns::beginRunIfNotDoneAlready() {
    if (!beginRunCalled_) beginRun(currentRun());
  }

  NewRun::NewRun(my_context ctx) :
    my_base(ctx)
  { 
    assert(context<HandleRuns>().currentRun() == INVALID_RUN);
    context<HandleRuns>().setupCurrentRun();

    // Here we assume that the input source or event processor
    // will throw if we fail to get a valid run.  Therefore
    // we should not ever fail this assert.
    assert(context<HandleRuns>().currentRun() != INVALID_RUN);
  }

  NewRun::~NewRun() { }

  sc::result NewRun::react(Run const& run)
  {
    if (run == context<HandleRuns>().currentRun()) {
      return transit<ContinueRun1>();
    }
    context<HandleRuns>().finalizeRun();
    return transit<NewRun>();
  }

  sc::result NewRun::react(File const& file)
  {
    if (!context<HandleFiles>().shouldWeCloseOutput()) {
      return transit<HandleNewInputFile2>();
    }
    return forward_event();
  }

  HandleNewInputFile2::HandleNewInputFile2(my_context ctx) : 
    my_base(ctx)
  { 
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

  sc::result HandleNewInputFile2::react(Run const& run)
  {
    checkInvariant();

    if (context<HandleRuns>().currentRun() != run) {
      return transit<NewRun, HandleRuns, Run>(&HandleRuns::finalizeRun, run);
    }
    else {
      return transit<ContinueRun1>();
    }
  }

  sc::result HandleNewInputFile2::react(File const& file)
  {
    checkInvariant();
    if (!context<HandleFiles>().shouldWeCloseOutput()) {
      return transit<HandleNewInputFile2>();
    }
    return forward_event();
  }

  ContinueRun1::ContinueRun1(my_context ctx) :
    my_base(ctx),
    ep_(context<Machine>().ep())
  { 
    ep_.readAndCacheRun();
    checkInvariant();
  }

  ContinueRun1::~ContinueRun1() {
    checkInvariant();
  }

  bool ContinueRun1::checkInvariant() {
    assert(context<HandleRuns>().currentRun() != INVALID_RUN);
    return true;
  }

  sc::result ContinueRun1::react(File const& file)
  {
    checkInvariant();
    if (!context<HandleFiles>().shouldWeCloseOutput()) {
      return transit<HandleNewInputFile2>();
    }
    return forward_event();
  }

  HandleLumis::LumiID::LumiID(edm::ProcessHistoryID const& phid, int run, int lumi) :
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
    lumiException_(false)
  { 
    checkInvariant();
  }

  void HandleLumis::exit() {
    if (ep_.alreadyHandlingException()) return;
    exitCalled_ = true;
    checkInvariant();
    if (!lumiException_ && !context<HandleRuns>().runException()) {
      finalizeLumi();
    }
  }

  HandleLumis::~HandleLumis() {
    if (!exitCalled_) {
      try {
        checkInvariant();
        if (!lumiException_ && !context<HandleRuns>().runException()) {
          finalizeLumi();
        }
      }
      catch (cms::Exception& e) {
        std::ostringstream message;
        message << "------------------------------------------------------------\n"
                << "Another exception was caught while trying to clean up lumis after\n" 
                << "the primary exception.  We give up trying to clean up lumis at\n"
                << "this point.  The description of this additional exception follows:\n" 
                << "cms::Exception\n"
                << e.explainSelf();
        std::string msg(message.str());
        ep_.setExceptionMessageLumis(msg);
      }
      catch (std::bad_alloc& e) {
        std::ostringstream message;
        message << "------------------------------------------------------------\n"
                << "Another exception was caught while trying to clean up lumis\n" 
                << "after the primary exception.  We give up trying to clean up lumis\n"
                << "at this point.  This additional exception was a\n" 
                << "std::bad_alloc exception thrown inside HandleLumis::finalizeLumi.\n"
                << "The job has probably exhausted the virtual memory available\n"
                << "to the process.\n";
        std::string msg(message.str());
        ep_.setExceptionMessageLumis(msg);
      }
      catch (std::exception& e) {
        std::ostringstream message;
        message << "------------------------------------------------------------\n"
                << "Another exception was caught while trying to clean up lumis after\n" 
                << "the primary exception.  We give up trying to clean up lumis at\n"
                << "this point.  This additional exception was a\n" 
                << "standard library exception thrown inside HandleLumis::finalizeLumi\n"
                << e.what() << "\n";
        std::string msg(message.str());
        ep_.setExceptionMessageLumis(msg);
      }
      catch (...) {
        std::ostringstream message;
        message << "------------------------------------------------------------\n"
                << "Another exception was caught while trying to clean up lumis after\n" 
                << "the primary exception.  We give up trying to clean up lumis at\n"
                << "this point.  This additional exception was of unknown type and\n" 
                << "thrown inside HandleLumis::finalizeLumi\n";
        std::string msg(message.str());
        ep_.setExceptionMessageLumis(msg);
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
    assert (run != INVALID_RUN);
    lumiException_ = true;
    currentLumi_ = HandleLumis::LumiID(run.processHistoryID(), run.runNumber(), ep_.readAndCacheLumi());

    if (context<Machine>().emptyRunLumiMode() == handleEmptyRunsAndLumis) {
      assert(context<HandleRuns>().beginRunCalled());
      ep_.beginLumi(currentLumi().processHistoryID(), currentLumi().run(), currentLumi().lumi());
    }

    lumiException_ = false;

    currentLumiEmpty_ = true;
  }

  void HandleLumis::finalizeLumi() {

    lumiException_ = true;

    if (!currentLumiEmpty_ ||
        context<Machine>().emptyRunLumiMode() == handleEmptyRunsAndLumis) {
      ep_.endLumi(currentLumi().processHistoryID(), currentLumi().run(), currentLumi().lumi());
    }

    ep_.writeLumi(currentLumi().processHistoryID(), currentLumi().run(), currentLumi().lumi());
    ep_.deleteLumiFromCache(currentLumi().processHistoryID(), currentLumi().run(), currentLumi().lumi());
    currentLumi_ = InvalidLumiID;

    lumiException_ = false;
  }

  void HandleLumis::markLumiNonEmpty() {
    if (currentLumiEmpty_) {

      if (context<Machine>().emptyRunLumiMode() != handleEmptyRunsAndLumis) {
        lumiException_ = true;
        ep_.beginLumi(currentLumi().processHistoryID(), currentLumi().run(), currentLumi().lumi());
        lumiException_ = false;
      }
      currentLumiEmpty_ = false;
    }
  }

  FirstLumi::FirstLumi(my_context ctx) :
    my_base(ctx)
  { 
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

  sc::result FirstLumi::react(Lumi const& lumi)
  {
    if (lumi.id() == context<HandleLumis>().currentLumi().lumi()) {
      return transit<ContinueLumi>();
    }
    return transit<AnotherLumi>();
  }

  sc::result FirstLumi::react(File const& file)
  {
    checkInvariant();
    if (!context<HandleFiles>().shouldWeCloseOutput()) {
      return transit<HandleNewInputFile3>();
    }
    return forward_event();
  }

  AnotherLumi::AnotherLumi(my_context ctx) :
    my_base(ctx)
  { 
    context<HandleLumis>().finalizeLumi();
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

  sc::result AnotherLumi::react(Lumi const& lumi)
  {
    if (lumi.id() == context<HandleLumis>().currentLumi().lumi()) {
      return transit<ContinueLumi>();
    }
    return transit<AnotherLumi>();
  }

  sc::result AnotherLumi::react(File const& file)
  {
    checkInvariant();
    if (!context<HandleFiles>().shouldWeCloseOutput()) {
      return transit<HandleNewInputFile3>();
    }
    return forward_event();
  }

  HandleEvent::HandleEvent(my_context ctx) :
    my_base(ctx),
    ep_(context<Machine>().ep())
  { 
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

  sc::result HandleEvent::react(File const& file)
  {
    checkInvariant();
    if (!context<HandleFiles>().shouldWeCloseOutput()) {
      return transit<HandleNewInputFile3>();
    }
    return forward_event();
  }

  void HandleEvent::readAndProcessEvent() {
    markNonEmpty();
    ep_.readAndProcessEvent();
    if (ep_.shouldWeStop()) post_event(Stop());
  }

  void HandleEvent::markNonEmpty() {
    context<HandleRuns>().beginRunIfNotDoneAlready();
    context<HandleLumis>().markLumiNonEmpty();
  }


  HandleNewInputFile3::HandleNewInputFile3(my_context ctx) :
    my_base(ctx)
  { 
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

  sc::result HandleNewInputFile3::react(Run const& run)
  {
    checkInvariant();

    if (context<HandleRuns>().currentRun() == run) {
      return transit<ContinueRun2>();
    }
    return forward_event();
  }

  sc::result HandleNewInputFile3::react(File const& file)
  {
    checkInvariant();
    if (!context<HandleFiles>().shouldWeCloseOutput()) {
      return transit<HandleNewInputFile3>();
    }
    return forward_event();
  }

  ContinueRun2::ContinueRun2(my_context ctx) :
    my_base(ctx),
    ep_(context<Machine>().ep())
  { 
    ep_.readAndCacheRun();
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

  sc::result ContinueRun2::react(Lumi const& lumi)
  {
    checkInvariant();

    if (context<HandleLumis>().currentLumi().lumi() != lumi.id()) {
      return transit<AnotherLumi>();
    }
    else {
      return transit<ContinueLumi>();
    }
  }

  sc::result ContinueRun2::react(File const& file)
  {
    checkInvariant();
    if (!context<HandleFiles>().shouldWeCloseOutput()) {
      return transit<HandleNewInputFile3>();
    }
    return forward_event();
  }

  ContinueLumi::ContinueLumi(my_context ctx) :
    my_base(ctx),
    ep_(context<Machine>().ep())
  { 
    ep_.readAndCacheLumi();
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

  sc::result ContinueLumi::react(File const& file)
  {
    checkInvariant();
    if (!context<HandleFiles>().shouldWeCloseOutput()) {
      return transit<HandleNewInputFile3>();
    }
    return forward_event();
  }
}
