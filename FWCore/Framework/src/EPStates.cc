
// $Id: EPStates.cc,v 1.6 2008/04/08 18:13:36 wdd Exp $

#include "FWCore/Framework/src/EPStates.h"
#include "FWCore/Framework/interface/IEventProcessor.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <exception>
#include <sstream>
#include <string>

namespace statemachine {

  Run::Run(int id) : id_(id) {}
  int Run::id() const { return id_; }

  Lumi::Lumi(int id) : id_(id) {}
  int Lumi::id() const { return id_; }

  Machine::Machine(edm::IEventProcessor* ep,
                   FileMode fileMode,
                   bool handleEmptyRuns,
                   bool handleEmptyLumis) :
    ep_(ep),
    fileMode_(fileMode),
    handleEmptyRuns_(handleEmptyRuns),
    handleEmptyLumis_(handleEmptyLumis) { }

  edm::IEventProcessor& Machine::ep() const { return *ep_; }
  FileMode Machine::fileMode() const { return fileMode_; }
  bool Machine::handleEmptyRuns() const { return handleEmptyRuns_; }
  bool Machine::handleEmptyLumis() const { return handleEmptyLumis_; }

  void Machine::startingNewLoop(const File& file) {
    ep_->startingNewLoop();
  }

  void Machine::rewindAndPrepareForNextLoop(const Restart & restart) {
    ep_->prepareForNextLoop();
    ep_->rewindInput();
  }

  Starting::Starting(my_context ctx) : my_base(ctx) { }

  Starting::~Starting() { }

  sc::result Starting::react( const Stop& stop)
  {
    return terminate();
  }

  HandleFiles::HandleFiles(my_context ctx) :
    my_base(ctx),
    ep_(context< Machine >().ep()),
    exitCalled_(false) { }

  void HandleFiles::exit() {
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
    ep_.writeCache();
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
    if (context< Machine >().fileMode() == SPARSE) return true;
    return ep_.shouldWeCloseOutput();
  }

  EndingLoop::EndingLoop(my_context ctx) : 
    my_base(ctx),
    ep_(context< Machine >().ep())
  { 
    if (ep_.endOfLoop()) post_event(Stop());
    else post_event(Restart());
  }

  EndingLoop::~EndingLoop() { }

  sc::result EndingLoop::react( const Stop & )
  {
    return terminate();
  }

  Error::Error(my_context ctx) : 
    my_base(ctx),
    ep_(context< Machine >().ep())
  { 
    post_event(Stop());
    ep_.doErrorStuff();
  }

  Error::~Error() { }

  class HandleNewInputFile1;
  class NewInputAndOutputFiles;

  FirstFile::FirstFile(my_context ctx) :
    my_base(ctx),
    ep_(context< Machine >().ep())
  { 
    openFiles();
  }

  FirstFile::~FirstFile() { }

  sc::result FirstFile::react( const File & file)
  {
    if (context< HandleFiles >().shouldWeCloseOutput()) {
      return transit< NewInputAndOutputFiles >();
    }
    else {
      return transit< HandleNewInputFile1 >();
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
    context< HandleFiles >().goToNewInputFile();
  }

  HandleNewInputFile1::~HandleNewInputFile1() { }

  sc::result HandleNewInputFile1::react( const File & file)
  {
    if (context< HandleFiles >().shouldWeCloseOutput()) {
      return transit< NewInputAndOutputFiles >();
    }
    else {
      return transit< HandleNewInputFile1 >();
    }
  }

  NewInputAndOutputFiles::NewInputAndOutputFiles(my_context ctx) : 
    my_base(ctx),
    ep_(context< Machine >().ep())
  { 
    goToNewInputAndOutputFiles();
  }

  NewInputAndOutputFiles::~NewInputAndOutputFiles() { }

  sc::result NewInputAndOutputFiles::react( const File & file)
  {
    if (context< HandleFiles >().shouldWeCloseOutput()) {
      return transit< NewInputAndOutputFiles >();
    }
    else {
      return transit< HandleNewInputFile1 >();
    }
  }

  void NewInputAndOutputFiles::goToNewInputAndOutputFiles() {
    ep_.respondToCloseInputFile();
    ep_.closeInputFile();

    ep_.writeCache();
    ep_.respondToCloseOutputFiles();
    ep_.closeOutputFiles();

    ep_.readFile();
    ep_.respondToOpenInputFile();

    ep_.openOutputFiles();
    ep_.respondToOpenOutputFiles();
  }

  HandleRuns::HandleRuns(my_context ctx) : 
    my_base(ctx),
    ep_(context< Machine >().ep()),
    exitCalled_(false),
    beginRunCalled_(false),
    currentRun_(INVALID_RUN),
    runException_(false) { }

  void HandleRuns::exit() {
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
  int HandleRuns::currentRun() const { return currentRun_; }
  bool HandleRuns::runException() const { return runException_; }

  void HandleRuns::setupCurrentRun() {

    runException_ = true;
    currentRun_ = ep_.readAndCacheRun();
    runException_ = false;

    if (context< Machine >().handleEmptyRuns()) {
      beginRun(currentRun());
    }
  }

  void HandleRuns::beginRun(int run) {
    beginRunCalled_ = true;

    runException_ = true;
    ep_.beginRun(run);
    runException_ = false;
  }

  void HandleRuns::endRun(int run) {
    beginRunCalled_ = false;

    runException_ = true;
    ep_.endRun(run);
    runException_ = false;
  }

  void HandleRuns::finalizeRun(const Run &) {
    finalizeRun();
  }

  void HandleRuns::finalizeRun() {

    if (runException_) return;
    runException_ = true;

    if (beginRunCalled_) endRun(currentRun());
    if (context< Machine >().fileMode() == SPARSE) {
      ep_.writeRun(currentRun_);
      ep_.deleteRunFromCache(currentRun_);
    }
    currentRun_ = INVALID_RUN;
    runException_ = false;   
  }

  void HandleRuns::beginRunIfNotDoneAlready() {
    if (!beginRunCalled_) beginRun(currentRun());
  }

  NewRun::NewRun(my_context ctx) :
    my_base(ctx)
  { 
    assert(context< HandleRuns >().currentRun() == INVALID_RUN);
    context< HandleRuns >().setupCurrentRun();

    // Here we assume that the input source or event processor
    // will throw if we fail to get a valid run.  Therefore
    // we should not ever fail this assert.
    assert(context< HandleRuns >().currentRun() != INVALID_RUN);
  }

  NewRun::~NewRun() { }

  sc::result NewRun::react( const File & file)
  {
    if (!context< HandleFiles >().shouldWeCloseOutput()) {
      return transit< HandleNewInputFile2 >();
    }
    return forward_event();
  }

  HandleNewInputFile2::HandleNewInputFile2(my_context ctx) : 
    my_base(ctx)
  { 
    context< HandleFiles >().goToNewInputFile();
    checkInvariant();
  }

  HandleNewInputFile2::~HandleNewInputFile2() {
    checkInvariant();
  }

  bool HandleNewInputFile2::checkInvariant() {
    assert(context< HandleRuns >().currentRun() != INVALID_RUN);
    return true;
  }

  sc::result HandleNewInputFile2::react( const Run & run)
  {
    checkInvariant();

    if ( context< HandleRuns >().currentRun() != run.id() ) {
      return transit< NewRun, HandleRuns, Run >(&HandleRuns::finalizeRun, run);
    }
    else {
      return transit< ContinueRun1 >();
    }
  }

  sc::result HandleNewInputFile2::react( const File & file)
  {
    checkInvariant();
    if (!context< HandleFiles >().shouldWeCloseOutput()) {
      return transit< HandleNewInputFile2 >();
    }
    return forward_event();
  }

  ContinueRun1::ContinueRun1(my_context ctx) :
    my_base(ctx),
    ep_(context< Machine >().ep())
  { 
    ep_.readAndCacheRun();
    checkInvariant();
  }

  ContinueRun1::~ContinueRun1() {
    checkInvariant();
  }

  bool ContinueRun1::checkInvariant() {
    assert(context< HandleRuns >().currentRun() != INVALID_RUN);
    return true;
  }

  sc::result ContinueRun1::react( const File & file)
  {
    checkInvariant();
    if (!context< HandleFiles >().shouldWeCloseOutput()) {
      return transit< HandleNewInputFile2 >();
    }
    return forward_event();
  }

  HandleLumis::HandleLumis(my_context ctx) :
    my_base(ctx),
    ep_(context< Machine >().ep()),
    exitCalled_(false),
    currentLumiEmpty_(true),
    currentLumi_(INVALID_LUMI),
    lumiException_(false)
  { 
    checkInvariant();
  }

  void HandleLumis::exit() {
    exitCalled_ = true;
    checkInvariant();
    finalizeAllLumis();
  }

  HandleLumis::~HandleLumis() {
    if (!exitCalled_) {
      try {
        checkInvariant();
        finalizeAllLumis();
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
                << "std::bad_alloc exception thrown inside HandleLumis::finalizeAllLumis.\n"
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
                << "standard library exception thrown inside HandleLumis::finalizeAllLumis\n"
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
                << "thrown inside HandleLumis::finalizeAllLumis\n";
        std::string msg(message.str());
        ep_.setExceptionMessageLumis(msg);
      }
    }
  }

  bool HandleLumis::checkInvariant() {
    assert(context< HandleRuns >().currentRun() != INVALID_RUN);
    return true;
  }

  int HandleLumis::currentLumi() const { return currentLumi_; }

  bool HandleLumis::currentLumiEmpty() const { return currentLumiEmpty_; }

  const std::vector<int>& HandleLumis::unhandledLumis() const 
  { 
    return unhandledLumis_;
  }

  void HandleLumis::setupCurrentLumi() {

    lumiException_ = true;
    currentLumi_ = ep_.readAndCacheLumi();
    lumiException_ = false;

    currentLumiEmpty_ = true;
  }

  void HandleLumis::finalizeAllLumis() {
    if (lumiException_ || context< HandleRuns >().runException()) return;
    finalizeLumi();
    finalizeOutstandingLumis();
  }

  void HandleLumis::finalizeLumi() {

    lumiException_ = true;

    if (currentLumiEmpty_) {
      if (context< Machine >().handleEmptyLumis()) {
        if (context< HandleRuns >().beginRunCalled()) {
          int run = context< HandleRuns >().currentRun();
          ep_.beginLumi(run, currentLumi());
          ep_.endLumi(run, currentLumi());
          if (context< Machine >().fileMode() == SPARSE) {
            ep_.writeLumi(run, currentLumi());
            ep_.deleteLumiFromCache(run, currentLumi());
          }
        }
        else {
          unhandledLumis_.push_back(currentLumi());
        }
      }
      else {
        if (context< Machine >().fileMode() == SPARSE) {
          int run = context< HandleRuns >().currentRun();
          ep_.writeLumi(run, currentLumi());
          ep_.deleteLumiFromCache(run, currentLumi());
        }
      }
    }
    else { 
      int run = context< HandleRuns >().currentRun();
      ep_.endLumi(run, currentLumi());
      if (context< Machine >().fileMode() == SPARSE) {
        ep_.writeLumi(run, currentLumi());
        ep_.deleteLumiFromCache(run, currentLumi());
      }
    }
    currentLumi_ = INVALID_LUMI;

    lumiException_ = false;
  }

  void HandleLumis::finalizeOutstandingLumis() {

    lumiException_ = true;

    int run = context< HandleRuns >().currentRun();
    for (std::vector<int>::const_iterator iter = unhandledLumis_.begin();
         iter != unhandledLumis_.end();
         ++iter) {
      ep_.beginLumi(run, *iter);
      ep_.endLumi(run, *iter);
      if (context< Machine >().fileMode() == SPARSE) {
        ep_.writeLumi(run, *iter);
        ep_.deleteLumiFromCache(run, *iter);
      }
    }
    unhandledLumis_.clear();

    lumiException_ = false;
  }

  void HandleLumis::markLumiNonEmpty() {
    if (currentLumiEmpty_) {
      finalizeOutstandingLumis();
      int run = context< HandleRuns >().currentRun();

      lumiException_ = true;
      ep_.beginLumi(run, currentLumi());
      lumiException_ = false;

      currentLumiEmpty_ = false;
    }
  }

  FirstLumi::FirstLumi(my_context ctx) :
    my_base(ctx)
  { 
    context< HandleLumis >().setupCurrentLumi();
    checkInvariant();
  }

  FirstLumi::~FirstLumi() {
    checkInvariant();
  }

  bool FirstLumi::checkInvariant() {
    assert(context< HandleRuns >().currentRun() != INVALID_RUN);
    assert(context< HandleLumis >().currentLumi() != INVALID_LUMI);
    assert(context< HandleLumis >().unhandledLumis().empty());
    assert(context< HandleLumis >().currentLumiEmpty() == true);
    return true;
  }

  sc::result FirstLumi::react( const File & file)
  {
    checkInvariant();
    if (!context< HandleFiles >().shouldWeCloseOutput()) {
      return transit< HandleNewInputFile3 >();
    }
    return forward_event();
  }

  AnotherLumi::AnotherLumi(my_context ctx) :
    my_base(ctx)
  { 
    context< HandleLumis >().finalizeLumi();
    context< HandleLumis >().setupCurrentLumi();
    checkInvariant();
  }

  AnotherLumi::~AnotherLumi() {
    checkInvariant();
  }

  bool AnotherLumi::checkInvariant() {
    assert(context< HandleRuns >().currentRun() != INVALID_RUN);
    assert(context< HandleLumis >().currentLumi() != INVALID_LUMI);
    assert(context< HandleLumis >().currentLumiEmpty() == true);
    return true;
  }

  sc::result AnotherLumi::react( const File & file)
  {
    checkInvariant();
    if (!context< HandleFiles >().shouldWeCloseOutput()) {
      return transit< HandleNewInputFile3 >();
    }
    return forward_event();
  }

  HandleEvent::HandleEvent(my_context ctx) :
    my_base(ctx),
    ep_(context< Machine >().ep())
  { 
    readAndProcessEvent();
    checkInvariant();
  }

  HandleEvent::~HandleEvent() {
    checkInvariant();
  }

  bool HandleEvent::checkInvariant() {
    assert(context< HandleRuns >().currentRun() != INVALID_RUN);
    assert(context< HandleRuns >().beginRunCalled());
    assert(context< HandleLumis >().currentLumi() != INVALID_LUMI);
    assert(context< HandleLumis >().unhandledLumis().empty());
    assert(context< HandleLumis >().currentLumiEmpty() == false);
    return true;
  }

  sc::result HandleEvent::react( const File & file)
  {
    checkInvariant();
    if (!context< HandleFiles >().shouldWeCloseOutput()) {
      return transit< HandleNewInputFile3 >();
    }
    return forward_event();
  }

  void HandleEvent::readAndProcessEvent() {
    markNonEmpty();
    ep_.readEvent();
    ep_.processEvent();
    if (ep_.shouldWeStop()) post_event(Stop());
  }

  void HandleEvent::markNonEmpty() {
    context< HandleRuns >().beginRunIfNotDoneAlready();
    context< HandleLumis >().markLumiNonEmpty();
  }


  HandleNewInputFile3::HandleNewInputFile3(my_context ctx) :
    my_base(ctx)
  { 
    context< HandleFiles >().goToNewInputFile();
    checkInvariant();
  }

  HandleNewInputFile3::~HandleNewInputFile3() {
    checkInvariant();
  }

  bool HandleNewInputFile3::checkInvariant() {
    assert(context< HandleRuns >().currentRun() != INVALID_RUN);
    assert(context< HandleLumis >().currentLumi() != INVALID_LUMI);
    return true;
  }

  sc::result HandleNewInputFile3::react( const Run & run)
  {
    checkInvariant();

    if ( context< HandleRuns >().currentRun() == run.id() ) {
      return transit< ContinueRun2 >();
    }
    return forward_event();
  }

  sc::result HandleNewInputFile3::react( const File & file)
  {
    checkInvariant();
    if (!context< HandleFiles >().shouldWeCloseOutput()) {
      return transit< HandleNewInputFile3 >();
    }
    return forward_event();
  }

  ContinueRun2::ContinueRun2(my_context ctx) :
    my_base(ctx),
    ep_(context< Machine >().ep())
  { 
    ep_.readAndCacheRun();
    checkInvariant();
  }

  ContinueRun2::~ContinueRun2() {
    checkInvariant();
  }

  bool ContinueRun2::checkInvariant() {
    assert(context< HandleRuns >().currentRun() != INVALID_RUN);
    assert(context< HandleLumis >().currentLumi() != INVALID_LUMI);
    return true;
  }

  sc::result ContinueRun2::react( const Lumi & lumi)
  {
    checkInvariant();

    if ( context< HandleLumis >().currentLumi() != lumi.id() ) {
      return transit< AnotherLumi >();
    }
    else {
      return transit< ContinueLumi >();
    }
  }

  sc::result ContinueRun2::react( const File & file)
  {
    checkInvariant();
    if (!context< HandleFiles >().shouldWeCloseOutput()) {
      return transit< HandleNewInputFile3 >();
    }
    return forward_event();
  }

  ContinueLumi::ContinueLumi(my_context ctx) :
    my_base(ctx),
    ep_(context< Machine >().ep())
  { 
    ep_.readAndCacheLumi();
    checkInvariant();
  }

  ContinueLumi::~ContinueLumi() {
    checkInvariant();
  }

  bool ContinueLumi::checkInvariant() {
    assert(context< HandleRuns >().currentRun() != INVALID_RUN);
    assert(context< HandleLumis >().currentLumi() != INVALID_LUMI);
    return true;
  }

  sc::result ContinueLumi::react( const File & file)
  {
    checkInvariant();
    if (!context< HandleFiles >().shouldWeCloseOutput()) {
      return transit< HandleNewInputFile3 >();
    }
    return forward_event();
  }
}
