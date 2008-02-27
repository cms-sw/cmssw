
// $Id: EPStates.cc,v 1.3 2008/01/09 23:45:55 wdd Exp $

#include "FWCore/Framework/src/EPStates.h"
#include "FWCore/Framework/interface/IEventProcessor.h"


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
    ep_(context< Machine >().ep()) { }

  HandleFiles::~HandleFiles() {
    closeFiles();
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
    beginRunCalled_(false),
    currentRun_(INVALID_RUN) { }

  HandleRuns::~HandleRuns() {
    finalizeRun();
  }

  bool HandleRuns::beginRunCalled() const { return beginRunCalled_; }
  int HandleRuns::currentRun() const { return currentRun_; }

  void HandleRuns::setupCurrentRun() {
    currentRun_ = ep_.readAndCacheRun();
    if (context< Machine >().handleEmptyRuns()) {
      beginRun(currentRun());
    }
  }

  void HandleRuns::beginRun(int run) {
    beginRunCalled_ = true;
    ep_.smBeginRun(run);
  }

  void HandleRuns::endRun(int run) {
    beginRunCalled_ = false;
    ep_.smEndRun(run);
  }

  void HandleRuns::finalizeRun(const Run &) {
    finalizeRun();
  }

  void HandleRuns::finalizeRun() {
    if (beginRunCalled_) endRun(currentRun());
    if (context< Machine >().fileMode() == SPARSE) {
      ep_.writeRun(currentRun_);
      ep_.deleteRunFromCache(currentRun_);
    }
    currentRun_ = INVALID_RUN;
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
    currentLumiEmpty_(true),
    currentLumi_(INVALID_LUMI)
  { 
    checkInvariant();
  }

  HandleLumis::~HandleLumis() {
    checkInvariant();
    finalizeAllLumis();
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
    currentLumi_ = ep_.readAndCacheLumi();
    currentLumiEmpty_ = true;
  }

  void HandleLumis::finalizeAllLumis() {
    finalizeLumi();
    finalizeOutstandingLumis();
  }

  void HandleLumis::finalizeLumi() {
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
  }

  void HandleLumis::finalizeOutstandingLumis() {
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
  }

  void HandleLumis::markLumiNonEmpty() {
    if (currentLumiEmpty_) {
      finalizeOutstandingLumis();
      int run = context< HandleRuns >().currentRun();
      ep_.beginLumi(run, currentLumi());
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
