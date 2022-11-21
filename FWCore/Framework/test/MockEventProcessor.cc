#include "FWCore/Framework/test/MockEventProcessor.h"

namespace {
  // As each data item is read from the mock data it is
  // stored in one of these:
  struct token {
    int value;
    char id;
  };

  std::istream& operator>>(std::istream& is, token& t) {
    if (is >> t.id)
      is >> t.value;
    return is;
  }

  //The TransitionProcessors.icc uses the class name
  // EventProcessor (with no namespace) as the type
  // to which it interacts.
  using EventProcessor = edm::MockEventProcessor;

  class WaitingTaskHolder;
}  // namespace

#define TEST_NO_FWD_DECL

namespace {
#include "FWCore/Framework/src/TransitionProcessors.icc"
}

namespace edm {

  MockEventProcessor::MockEventProcessor(std::string const& mockData, std::ostream& output, bool iDoNotMerge)
      : mockData_(mockData),
        output_(output),
        input_(mockData_),
        nextRun_(0),
        nextLumi_(0),
        doNotMerge_(iDoNotMerge),
        shouldWeCloseOutput_(true),
        shouldWeEndLoop_(true),
        shouldWeStop_(false),
        eventProcessed_(false),
        reachedEndOfInput_(false),
        shouldThrow_(false) {}

  InputSource::ItemType MockEventProcessor::nextTransitionType() {
    token t;
    if (not(input_ >> t)) {
      reachedEndOfInput_ = true;
      return lastTransition_ = InputSource::IsStop;
    }

    char ch = t.id;

    eventProcessed_ = false;
    if (ch == 'r') {
      output_ << "    *** nextItemType: Run " << t.value << " ***\n";
      nextRun_ = static_cast<RunNumber_t>(t.value);
      return lastTransition_ = InputSource::IsRun;
    } else if (ch == 'l') {
      output_ << "    *** nextItemType: Lumi " << t.value << " ***\n";
      nextLumi_ = static_cast<LuminosityBlockNumber_t>(t.value);
      return lastTransition_ = InputSource::IsLumi;
    } else if (ch == 'e') {
      output_ << "    *** nextItemType: Event ***\n";
      // a special value for test purposes only
      if (t.value == 7) {
        shouldWeStop_ = true;
        output_ << "    *** shouldWeStop will return true this event ***\n";
      } else {
        shouldWeStop_ = false;
      }
      return lastTransition_ = InputSource::IsEvent;
    } else if (ch == 'f') {
      output_ << "    *** nextItemType: File " << t.value << " ***\n";
      // a special value for test purposes only
      if (t.value == 0)
        shouldWeCloseOutput_ = false;
      else
        shouldWeCloseOutput_ = true;
      return lastTransition_ = InputSource::IsFile;
    } else if (ch == 's') {
      output_ << "    *** nextItemType: Stop " << t.value << " ***\n";
      // a special value for test purposes only
      if (t.value == 0)
        shouldWeEndLoop_ = false;
      else
        shouldWeEndLoop_ = true;
      return lastTransition_ = InputSource::IsStop;
    } else if (ch == 'x') {
      output_ << "    *** nextItemType: Restart " << t.value << " ***\n";
      shouldWeEndLoop_ = t.value;
      return lastTransition_ = InputSource::IsStop;
    } else if (ch == 't') {
      output_ << "    *** nextItemType: Throw " << t.value << " ***\n";
      shouldThrow_ = true;
      return nextTransitionType();
    }
    return lastTransition_ = InputSource::IsInvalid;
  }

  InputSource::ItemType MockEventProcessor::lastTransitionType() const { return lastTransition_; }

  InputSource::ItemType MockEventProcessor::readAndProcessEvents() {
    bool first = true;
    do {
      if (first) {
        first = false;
      } else {
        shouldWeStop();
      }
      readAndProcessEvent();
      if (shouldWeStop()) {
        return InputSource::IsEvent;
      }
    } while (nextTransitionType() == InputSource::IsEvent);

    return lastTransitionType();
  }

  void MockEventProcessor::runToCompletion() {
    do {
      FilesProcessor fp(doNotMerge_);

      bool firstTime = true;
      do {
        if (not firstTime) {
          prepareForNextLoop();
          rewindInput();
        } else {
          firstTime = false;
        }
        startingNewLoop();

        auto trans = fp.processFiles(*this);

        fp.normalEnd();

        if (trans != InputSource::IsStop) {
          //problem with the source
          doErrorStuff();
          break;
        }
      } while (not endOfLoop());
      output_ << "Left processing loop.\n";
    } while (not reachedEndOfInput_ and not input_.eof());

    return;
  }

  void MockEventProcessor::readFile() {
    output_ << " \treadFile\n";
    throwIfNeeded();
  }

  void MockEventProcessor::closeInputFile(bool /*cleaningUpAfterException*/) { output_ << "\tcloseInputFile\n"; }

  void MockEventProcessor::openOutputFiles() { output_ << "\topenOutputFiles\n"; }

  void MockEventProcessor::closeOutputFiles() { output_ << "\tcloseOutputFiles\n"; }

  void MockEventProcessor::respondToOpenInputFile() { output_ << "\trespondToOpenInputFile\n"; }

  void MockEventProcessor::respondToCloseInputFile() { output_ << "\trespondToCloseInputFile\n"; }

  void MockEventProcessor::startingNewLoop() { output_ << "\tstartingNewLoop\n"; }

  bool MockEventProcessor::endOfLoop() {
    output_ << "\tendOfLoop\n";
    return shouldWeEndLoop_;
  }

  void MockEventProcessor::rewindInput() { output_ << "\trewind\n"; }

  void MockEventProcessor::prepareForNextLoop() { output_ << "\tprepareForNextLoop\n"; }

  bool MockEventProcessor::shouldWeCloseOutput() const {
    output_ << "\tshouldWeCloseOutput\n";
    return shouldWeCloseOutput_;
  }

  void MockEventProcessor::doErrorStuff() { output_ << "\tdoErrorStuff\n"; }

  void MockEventProcessor::beginProcessBlock(bool& beginProcessBlockSucceeded) {}
  void MockEventProcessor::inputProcessBlocks() {}
  void MockEventProcessor::endProcessBlock(bool cleaningUpAfterException, bool beginProcessBlockSucceeded) {}

  InputSource::ItemType MockEventProcessor::processRuns() {
    bool finished = false;
    auto nextTransition = edm::InputSource::IsRun;
    do {
      switch (nextTransition) {
        case edm::InputSource::IsRun: {
          processRun();
          nextTransition = nextTransitionType();
          break;
        }
        case edm::InputSource::IsLumi: {
          nextTransition = processLumis();
          break;
        }
        default:
          finished = true;
      }
    } while (not finished);
    return nextTransition;
  }

  void MockEventProcessor::processRun() {
    if ((not currentRun_) or (currentRunNumber_ != nextRun_)) {
      if (currentRun_) {
        endUnfinishedLumi(true);
        endUnfinishedRun(true);
      }
      currentRun_ = true;
      readRun();
      beginRun(currentRunNumber_);
    } else {
      //merge
      readAndMergeRun();
    }
  }

  InputSource::ItemType MockEventProcessor::processLumis() {
    if (lumiStatus_ and currentLumiNumber_ == nextLumi_) {
      readAndMergeLumi();
      if (nextTransitionType() == InputSource::IsEvent) {
        readAndProcessEvents();
        if (shouldWeStop()) {
          return edm::InputSource::IsStop;
        }
      }
    } else {
      endUnfinishedLumi(true);
      lumiStatus_ = true;
      auto lumi = readLuminosityBlock();
      output_ << "\tbeginLumi " << currentRunNumber_ << "/" << lumi << "\n";
      throwIfNeeded();
      didGlobalBeginLumiSucceed_ = true;
      //Need to do event processing here
      if (nextTransitionType() == InputSource::IsEvent) {
        readAndProcessEvents();
        if (shouldWeStop()) {
          return edm::InputSource::IsStop;
        }
      }
    }
    return lastTransitionType();
  }

  void MockEventProcessor::beginRun(RunNumber_t run) {
    output_ << "\tbeginRun " << run << "\n";
    throwIfNeeded();
    didGlobalBeginRunSucceed_ = true;
  }

  void MockEventProcessor::endRun() {
    auto postfix = didGlobalBeginRunSucceed_ ? "\n" : " global failed\n";
    output_ << "\tendRun " << currentRunNumber_ << postfix;
    currentRun_ = false;
  }

  void MockEventProcessor::endUnfinishedRun(bool) {
    endRun();
    if (didGlobalBeginRunSucceed_) {
      writeRun();
    }
    clearRunPrincipal();
  }

  void MockEventProcessor::endUnfinishedLumi(bool) {
    if (lumiStatus_) {
      endLumi();
      if (didGlobalBeginLumiSucceed_) {
        writeLumi();
      }
      clearLumiPrincipal();
    }
  }

  void MockEventProcessor::endLumi() {
    auto postfix = didGlobalBeginLumiSucceed_ ? "\n" : " global failed\n";
    output_ << "\tendLumi " << currentRunNumber_ << "/" << currentLumiNumber_ << postfix;
  }

  void MockEventProcessor::readRun() {
    currentRunNumber_ = nextRun_;
    output_ << "\treadRun " << currentRunNumber_ << "\n";
  }

  void MockEventProcessor::readAndMergeRun() { output_ << "\treadAndMergeRun " << currentRunNumber_ << "\n"; }

  LuminosityBlockNumber_t MockEventProcessor::readLuminosityBlock() {
    output_ << "\treadLuminosityBlock " << nextLumi_ << "\n";
    currentLumiNumber_ = nextLumi_;
    return currentLumiNumber_;
  }

  LuminosityBlockNumber_t MockEventProcessor::readAndMergeLumi() {
    output_ << "\treadAndMergeLumi " << currentLumiNumber_ << "\n";
    return currentLumiNumber_;
  }

  void MockEventProcessor::writeRun() { output_ << "\twriteRun " << currentRunNumber_ << "\n"; }

  void MockEventProcessor::clearRunPrincipal() {
    output_ << "\tclearRunPrincipal " << currentRunNumber_ << "\n";
    didGlobalBeginRunSucceed_ = false;
  }

  void MockEventProcessor::writeLumi() {
    output_ << "\twriteLumi " << currentRunNumber_ << "/" << currentLumiNumber_ << "\n";
  }

  void MockEventProcessor::clearLumiPrincipal() {
    output_ << "\tclearLumiPrincipal " << currentRunNumber_ << "/" << currentLumiNumber_ << "\n";
    lumiStatus_ = false;
    didGlobalBeginLumiSucceed_ = false;
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
    if (shouldThrow_) {
      shouldThrow_ = false;
      output_ << "\tthrowing\n";
      throw TestException();
    }
  }

  void MockEventProcessor::setExceptionMessageFiles(std::string&) {}
  void MockEventProcessor::setExceptionMessageRuns() {}
  void MockEventProcessor::setExceptionMessageLumis() {}

  bool MockEventProcessor::setDeferredException(std::exception_ptr) { return true; }

}  // namespace edm
