/*
*/

#include "FWCore/Framework/test/MockEventProcessor.h"
#include "FWCore/Framework/interface/InputSource.h"
#include <cassert>
#include <sstream>
#include <exception>

namespace {
  // As each data item is read from the mock data it is
  // stored in one of these:
  struct token {
    int value;
    char id;
  };

  std::istream& operator>>(std::istream& is, token& t) {
    if(is >> t.id) is >> t.value;
    return is;
  }
  
  //The TransitionProcessors.icc uses the class name
  // EventProcessor (with no namespace) as the type
  // to which it interacts.
  using EventProcessor = edm::MockEventProcessor;
  
  
  class WaitingTaskHolder;
}

namespace edm {
class LuminosityBlockPrincipal {
public:
  LuminosityBlockPrincipal(int iRun,int iLumi): run_(iRun),lumi_(iLumi){}
  int run_;
  int lumi_;
};
}

#define TEST_NO_FWD_DECL
#include "FWCore/Framework/src/LuminosityBlockProcessingStatus.h"

namespace {
#include "FWCore/Framework/src/TransitionProcessors.icc"
}

namespace edm {

  MockEventProcessor::MockEventProcessor(std::string const& mockData,
                                         std::ostream& output,
                                         bool iDoNotMerge) :
    mockData_(mockData),
    output_(output),
    input_(mockData_),
    run_(0),
    lumi_(0),
    doNotMerge_(iDoNotMerge),
    shouldWeCloseOutput_(true),
    shouldWeEndLoop_(true),
    shouldWeStop_(false),
    eventProcessed_(false),
    reachedEndOfInput_(false),
    shouldThrow_(false)
  {
  }

  InputSource::ItemType
  MockEventProcessor::nextTransitionType()
  {
    token t;
    if( not (input_ >> t)) {
      reachedEndOfInput_ = true;
      return lastTransition_=InputSource::IsStop;
    }

    char ch = t.id;

    eventProcessed_ = false;
    if(ch == 'r') {
      output_ << "    *** nextItemType: Run " << t.value << " ***\n";
      run_ = t.value;
      return lastTransition_=InputSource::IsRun;
    } else if(ch == 'l') {
      output_ << "    *** nextItemType: Lumi " << t.value << " ***\n";
      lumi_ = t.value;
      return lastTransition_=InputSource::IsLumi;
    } else if(ch == 'e') {
      output_ << "    *** nextItemType: Event ***\n";
      // a special value for test purposes only
      if(t.value == 7) {
        shouldWeStop_ = true;
        output_ << "    *** shouldWeStop will return true this event ***\n";
      } else {
        shouldWeStop_ = false;
      }
      return lastTransition_=InputSource::IsEvent;
    } else if(ch == 'f') {
      output_ << "    *** nextItemType: File " << t.value << " ***\n";
      // a special value for test purposes only
      if(t.value == 0) shouldWeCloseOutput_ = false;
      else shouldWeCloseOutput_ = true;
      return lastTransition_=InputSource::IsFile;
    } else if(ch == 's') {
      output_ << "    *** nextItemType: Stop " << t.value << " ***\n";
      // a special value for test purposes only
      if(t.value == 0) shouldWeEndLoop_ = false;
      else shouldWeEndLoop_ = true;
      return lastTransition_=InputSource::IsStop;
    } else if(ch == 'x') {
      output_ << "    *** nextItemType: Restart " << t.value << " ***\n";
      shouldWeEndLoop_ = t.value;
      return lastTransition_=InputSource::IsStop;
    } else if(ch == 't') {
      output_ << "    *** nextItemType: Throw " << t.value << " ***\n";
      shouldThrow_ = true;
      return nextTransitionType();
    }
    return lastTransition_=InputSource::IsInvalid;
  }
  
  InputSource::ItemType
  MockEventProcessor::lastTransitionType() const {
    return lastTransition_;
  }
  
  std::pair<edm::ProcessHistoryID, edm::RunNumber_t>
  MockEventProcessor::nextRunID() {
    return std::make_pair(edm::ProcessHistoryID{}, run_);
  }

  edm::LuminosityBlockNumber_t
  MockEventProcessor::nextLuminosityBlockID() {
    return lumi_;
  }
  
  InputSource::ItemType
  MockEventProcessor::readAndProcessEvents() {
    bool first = true;
    do {
      if(first) {
        first = false;
      } else {
        shouldWeStop();
      }
      readAndProcessEvent();
      if(shouldWeStop()) {
        return InputSource::IsEvent;
      }
    }while(nextTransitionType() == InputSource::IsEvent);
    
    return lastTransitionType();
  }


  void
  MockEventProcessor::runToCompletion() {

    do {
      FilesProcessor fp(doNotMerge_);

      bool firstTime = true;
      do {
        if(not firstTime) {
          prepareForNextLoop();
          rewindInput();
        } else {
          firstTime = false;
        }
        startingNewLoop();
        
        auto trans = fp.processFiles(*this);
        
        fp.normalEnd();
        
        if(trans != InputSource::IsStop) {
          //problem with the source
          doErrorStuff();
          break;
        }
      } while(not endOfLoop());
      output_ <<"Left processing loop.\n";
    } while(not reachedEndOfInput_ and not input_.eof());
      
    return;
  }
  
  void MockEventProcessor::readFile() {
    output_ << " \treadFile\n";
    throwIfNeeded();
  }

  void MockEventProcessor::closeInputFile(bool /*cleaningUpAfterException*/) {
    output_ << "\tcloseInputFile\n";
  }

  void MockEventProcessor::openOutputFiles() {
    output_ << "\topenOutputFiles\n";
  }

  void MockEventProcessor::closeOutputFiles() {
    output_ << "\tcloseOutputFiles\n";
  }

  void MockEventProcessor::respondToOpenInputFile() {
    output_ << "\trespondToOpenInputFile\n";
  }

  void MockEventProcessor::respondToCloseInputFile() {
    output_ << "\trespondToCloseInputFile\n";
  }

  void MockEventProcessor::startingNewLoop() {
    output_ << "\tstartingNewLoop\n";
  }

  bool MockEventProcessor::endOfLoop() {
    output_ << "\tendOfLoop\n";
    return shouldWeEndLoop_;
  }

  void MockEventProcessor::rewindInput() {
    output_ << "\trewind\n";
  }

  void MockEventProcessor::prepareForNextLoop() {
    output_ << "\tprepareForNextLoop\n";
  }

  bool MockEventProcessor::shouldWeCloseOutput() const {
    output_ << "\tshouldWeCloseOutput\n";
    return shouldWeCloseOutput_;
  }

  void MockEventProcessor::doErrorStuff() {
    output_ << "\tdoErrorStuff\n";
  }

  void MockEventProcessor::beginRun(ProcessHistoryID const& phid, RunNumber_t run, bool& globalTransitionSucceeded,
                                    bool& eventSetupForInstanceSucceeded) {
    output_ << "\tbeginRun " << run << "\n";
    eventSetupForInstanceSucceeded = true;
    throwIfNeeded();
    globalTransitionSucceeded = true;
  }

  void MockEventProcessor::endRun(ProcessHistoryID const& phid, RunNumber_t run, bool globalTransitionSucceeded, bool /*cleaningUpAfterException*/ ) {
    auto postfix = globalTransitionSucceeded? "\n" : " global failed\n";
    output_ << "\tendRun " << run << postfix;
  }

  void MockEventProcessor::endUnfinishedRun(ProcessHistoryID const& phid, RunNumber_t run,
                                            bool globalTransitionSucceeded, bool cleaningUpAfterException,
                                            bool eventSetupForInstanceSucceeded ) {
    if (eventSetupForInstanceSucceeded) {
      endRun(phid,run,globalTransitionSucceeded,cleaningUpAfterException);
      if(globalTransitionSucceeded) {
        writeRun(phid,run);
      }
    }
    deleteRunFromCache(phid,run);
  }

  InputSource::ItemType MockEventProcessor::processLumis(std::shared_ptr<void> iRunResource) {
    
    if(lumiStatus_ and
       lumiStatus_->runResource() == iRunResource and
       lumiStatus_->lumiPrincipal()->lumi_ == lumi_) {
      readAndMergeLumi(*lumiStatus_);
      
      if(nextTransitionType() == InputSource::IsEvent) {
        readAndProcessEvents();
        if(shouldWeStop()) {
          return edm::InputSource::IsStop;
        }
      }
    } else {
      endUnfinishedLumi();
      lumiStatus_ = std::make_shared<LuminosityBlockProcessingStatus>(this,1,iRunResource);
      auto lumi = readLuminosityBlock(*lumiStatus_);
      output_ << "\tbeginLumi " << run_ << "/" << lumi << "\n";
      throwIfNeeded();
      lumiStatus_->globalBeginDidSucceed();
      //Need to do event processing here
      if(nextTransitionType() == InputSource::IsEvent) {
        readAndProcessEvents();
        if(shouldWeStop()) {
          return edm::InputSource::IsStop;
        }
      }
    }
    return lastTransitionType();
  }

  void MockEventProcessor::endUnfinishedLumi() {
    if(lumiStatus_) {
      auto tmp = lumiStatus_;
      endLumi();
      if(tmp->didGlobalBeginSucceed()) {
        writeLumi(*tmp);
      }
      deleteLumiFromCache(*tmp);
    }
  }
  
  void MockEventProcessor::endLumi() {
    auto postfix = lumiStatus_->didGlobalBeginSucceed()? "\n" : " global failed\n";
    output_ << "\tendLumi " << lumiStatus_->lumiPrincipal()->run_ << "/" << lumiStatus_->lumiPrincipal()->lumi_ << postfix;
    lumiStatus_.reset();
  }

  std::pair<ProcessHistoryID,RunNumber_t> MockEventProcessor::readRun() {
    output_ << "\treadRun " << run_ << "\n";
    return std::make_pair(ProcessHistoryID(), run_);
  }

  std::pair<ProcessHistoryID,RunNumber_t> MockEventProcessor::readAndMergeRun() {
    output_ << "\treadAndMergeRun " << run_ << "\n";
    return std::make_pair(ProcessHistoryID(), run_);
  }

  int MockEventProcessor::readLuminosityBlock(LuminosityBlockProcessingStatus& iStatus) {
    output_ << "\treadLuminosityBlock " << lumi_ << "\n";
    iStatus.lumiPrincipal() = std::make_shared<LuminosityBlockPrincipal>(run_,lumi_);
    return lumi_;
  }

  int MockEventProcessor::readAndMergeLumi(LuminosityBlockProcessingStatus& iStatus) {
    output_ << "\treadAndMergeLumi " << lumi_ << "\n";
    return lumi_;
  }

  void MockEventProcessor::writeRun(ProcessHistoryID const& phid, RunNumber_t run) {
    output_ << "\twriteRun " << run << "\n";
  }

  void MockEventProcessor::deleteRunFromCache(ProcessHistoryID const& phid, RunNumber_t run) {
    output_ << "\tdeleteRunFromCache " << run << "\n";
  }

  void MockEventProcessor::writeLumi(LuminosityBlockProcessingStatus& iStatus) {
    output_ << "\twriteLumi " << iStatus.lumiPrincipal()->run_ << "/" << iStatus.lumiPrincipal()->lumi_ << "\n";
  }

  void MockEventProcessor::deleteLumiFromCache(LuminosityBlockProcessingStatus& iStatus) {
    output_ << "\tdeleteLumiFromCache " << iStatus.lumiPrincipal()->run_ << "/" << iStatus.lumiPrincipal()->lumi_ << "\n";
  }

  void MockEventProcessor::readAndProcessEvent() {
    output_ << "\treadEvent\n";
    output_ << "\tprocessEvent\n";
    eventProcessed_ = true;
    throwIfNeeded();
  }

  bool MockEventProcessor::shouldWeStop() const {
    output_ << "\tshouldWeStop\n";
    return eventProcessed_ and shouldWeStop_;
  }

  void MockEventProcessor::throwIfNeeded() {
    if(shouldThrow_) {
      shouldThrow_ = false;
      output_ <<"\tthrowing\n";
      throw TestException();
    }
  }
  
  void MockEventProcessor::setExceptionMessageFiles(std::string&) {}
  void MockEventProcessor::setExceptionMessageRuns(std::string&) {}
  void MockEventProcessor::setExceptionMessageLumis(std::string&) {}

  bool MockEventProcessor::setDeferredException(std::exception_ptr) { return true;}

}
