/*
*/

#include "FWCore/Framework/test/MockEventProcessor.h"

#include <sstream>

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
}
namespace edm {

  MockEventProcessor::MockEventProcessor(std::string const& mockData,
                                         std::ostream& output,
                                         statemachine::FileMode const& fileMode,
                                         statemachine::EmptyRunLumiMode const& emptyRunLumiMode) :
    mockData_(mockData),
    output_(output),
    fileMode_(fileMode),
    emptyRunLumiMode_(emptyRunLumiMode),
    run_(0),
    lumi_(0),
    shouldWeCloseOutput_(true),
    shouldWeEndLoop_(true),
    shouldWeStop_(false)  {
  }

  edm::MockEventProcessor::StatusCode
  MockEventProcessor::runToCompletion(bool /*onlineStateTransitions*/) {
    statemachine::Machine myMachine(this,
                                    fileMode_,
                                    emptyRunLumiMode_);

    myMachine.initiate();

    // Loop over the mock data items
    std::istringstream input(mockData_);
    token t;
    while(input >> t) {

      char ch = t.id;

      if(ch == 'r') {
        output_ << "    *** nextItemType: Run " << t.value << " ***\n";
        run_ = t.value;
        myMachine.process_event(statemachine::Run(ProcessHistoryID(), t.value));
      } else if(ch == 'l') {
        output_ << "    *** nextItemType: Lumi " << t.value << " ***\n";
        lumi_ = t.value;
        myMachine.process_event(statemachine::Lumi(t.value));
      } else if(ch == 'e') {
        output_ << "    *** nextItemType: Event ***\n";
        // a special value for test purposes only
        if(t.value == 7) {
          shouldWeStop_ = true;
          output_ << "    *** shouldWeStop will return true this event ***\n";
        } else {
          shouldWeStop_ = false;
        }
        myMachine.process_event(statemachine::Event());
      } else if(ch == 'f') {
        output_ << "    *** nextItemType: File " << t.value << " ***\n";
        // a special value for test purposes only
        if(t.value == 0) shouldWeCloseOutput_ = false;
        else shouldWeCloseOutput_ = true;
        myMachine.process_event(statemachine::File());
      } else if(ch == 's') {
        output_ << "    *** nextItemType: Stop " << t.value << " ***\n";
        // a special value for test purposes only
        if(t.value == 0) shouldWeEndLoop_ = false;
        else shouldWeEndLoop_ = true;
        myMachine.process_event(statemachine::Stop());
      } else if(ch == 'x') {
        output_ << "    *** nextItemType: Restart " << t.value << " ***\n";
        shouldWeEndLoop_ = t.value;
        myMachine.process_event(statemachine::Restart());
      }
      if(myMachine.terminated()) {
        output_ << "The state machine reports it has been terminated\n";
      }
    }
    return epSuccess;
  }

  // Not used, this one does nothing
  edm::MockEventProcessor::StatusCode
  MockEventProcessor::runEventCount(int /*numberOfEventsToProcess*/) {
    return epSuccess;
  }

  void MockEventProcessor::readFile() {
    output_ << " \treadFile\n";
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

  void MockEventProcessor::respondToOpenOutputFiles() {
    output_ << "\trespondToOpenOutputFiles\n";
  }

  void MockEventProcessor::respondToCloseOutputFiles() {
    output_ << "\trespondToCloseOutputFiles\n";
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

  void MockEventProcessor::beginRun(statemachine::Run const& run) {
    output_ << "\tbeginRun " << run.runNumber() << "\n";
  }

  void MockEventProcessor::endRun(statemachine::Run const& run, bool /*cleaningUpAfterException*/ ) {
    output_ << "\tendRun " << run.runNumber() << "\n";
  }

  void MockEventProcessor::beginLumi(ProcessHistoryID const&, int run, int lumi) {
    output_ << "\tbeginLumi " << run << "/" << lumi << "\n";
  }

  void MockEventProcessor::endLumi(ProcessHistoryID const&, int run, int lumi, bool /*cleaningUpAfterException*/) {
    output_ << "\tendLumi " << run << "/" << lumi << "\n";
  }

  statemachine::Run MockEventProcessor::readAndCacheRun(bool merge) {
    output_ << "\treadAndCacheRun " << run_ << "\n";
    return statemachine::Run(ProcessHistoryID(), run_);
  }

  int MockEventProcessor::readAndCacheLumi(bool merge) {
    output_ << "\treadAndCacheLumi " << lumi_ << "\n";
    return lumi_;
  }

  void MockEventProcessor::writeRun(statemachine::Run const& run) {
    output_ << "\twriteRun " << run.runNumber() << "\n";
  }

  void MockEventProcessor::deleteRunFromCache(statemachine::Run const& run) {
    output_ << "\tdeleteRunFromCache " << run.runNumber() << "\n";
  }

  void MockEventProcessor::writeLumi(ProcessHistoryID const&, int run, int lumi) {
    output_ << "\twriteLumi " << run << "/" << lumi << "\n";
  }

  void MockEventProcessor::deleteLumiFromCache(ProcessHistoryID const&, int run, int lumi) {
    output_ << "\tdeleteLumiFromCache " << run << "/" << lumi << "\n";
  }

  void MockEventProcessor::readAndProcessEvent() {
    output_ << "\treadEvent\n";
    output_ << "\tprocessEvent\n";
  }

  bool MockEventProcessor::shouldWeStop() const {
    output_ << "\tshouldWeStop\n";
    return shouldWeStop_;
  }

  void MockEventProcessor::setExceptionMessageFiles(std::string&) {}
  void MockEventProcessor::setExceptionMessageRuns(std::string&) {}
  void MockEventProcessor::setExceptionMessageLumis(std::string&) {}

  bool MockEventProcessor::alreadyHandlingException() const { return false; }
}
