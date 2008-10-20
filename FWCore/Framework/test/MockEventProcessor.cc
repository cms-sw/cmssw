
/*
$Id: MockEventProcessor.cc,v 1.12 2008/07/29 02:17:36 wmtan Exp $
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

  std::istream & operator>>(std::istream & is, token & t) {
    if(is >> t.id) is >> t.value;
    return is;
  }
}

namespace edm {

  MockEventProcessor::MockEventProcessor(const std::string& mockData,
                                         std::ostream& output,
                                         const statemachine::FileMode& fileMode,
                                         bool handleEmptyRuns,
                                         bool handleEmptyLumis) :
    mockData_(mockData),
    output_(output),
    fileMode_(fileMode),
    handleEmptyRuns_(handleEmptyRuns),
    handleEmptyLumis_(handleEmptyLumis),
    shouldWeCloseOutput_(true),
    shouldWeEndLoop_(true),
    shouldWeStop_(false)  {
  }

  edm::MockEventProcessor::StatusCode
  MockEventProcessor::runToCompletion(bool onlineStateTransitions) {
    statemachine::Machine myMachine(this,
                                    fileMode_,
                                    handleEmptyRuns_,
                                    handleEmptyLumis_);

  
    myMachine.initiate();

    // Loop over the mock data items
    std::istringstream input(mockData_);
    token t;
    while (input >> t) {

      char ch = t.id;

      if (ch == 'r') {
        output_ << "    *** nextItemType: Run " << t.value << " ***\n";
        run_ = t.value;
        myMachine.process_event( statemachine::Run(t.value) );
      }
      else if (ch == 'l') {
        output_ << "    *** nextItemType: Lumi " << t.value << " ***\n";
        lumi_ = t.value;
        myMachine.process_event( statemachine::Lumi(t.value) );
      }
      else if (ch == 'e') {
        output_ << "    *** nextItemType: Event ***\n";
        // a special value for test purposes only
        if (t.value == 7) {
          shouldWeStop_ = true;
          output_ << "    *** shouldWeStop will return true this event ***\n";
        }
        else {
          shouldWeStop_ = false;
        }
        myMachine.process_event( statemachine::Event() );
      }
      else if (ch == 'f') {
        output_ << "    *** nextItemType: File " << t.value << " ***\n";
        // a special value for test purposes only
        if (t.value == 0) shouldWeCloseOutput_ = false;
        else shouldWeCloseOutput_ = true;
        myMachine.process_event( statemachine::File() );
      }
      else if (ch == 's') {
        output_ << "    *** nextItemType: Stop " << t.value << " ***\n";
        // a special value for test purposes only
        if (t.value == 0) shouldWeEndLoop_ = false;
        else shouldWeEndLoop_ = true;
        myMachine.process_event( statemachine::Stop() );
      }
      else if (ch == 'x') {
        output_ << "    *** nextItemType: Restart " << t.value << " ***\n";
        shouldWeEndLoop_ = t.value;
        myMachine.process_event( statemachine::Restart() );
      }

      if (myMachine.terminated()) {
	output_ << "The state machine reports it has been terminated\n";
      }
    }
    return epSuccess;
  }

  // Not used, this one does nothing
  edm::MockEventProcessor::StatusCode
  MockEventProcessor::runEventCount(int numberOfEventsToProcess) {
    return epSuccess;
  }

  void MockEventProcessor::readFile() {
    output_ << " \treadFile\n";
  }

  void MockEventProcessor::closeInputFile() {
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

  void MockEventProcessor::writeLumiCache() {
    output_ << "\twriteLumiCache\n";
  }

  void MockEventProcessor::writeRunCache() {
    output_ << "\twriteRunCache\n";
  }

  bool MockEventProcessor::shouldWeCloseOutput() const {
    output_ << "\tshouldWeCloseOutput\n";
    return shouldWeCloseOutput_;
  }

  void MockEventProcessor::doErrorStuff() {
    output_ << "\tdoErrorStuff\n";
  }

  void MockEventProcessor::beginRun(int run) {
    output_ << "\tbeginRun " << run << "\n";
  }

  void MockEventProcessor::endRun(int run) {
    output_ << "\tendRun " << run << "\n";
  }

  void MockEventProcessor::beginLumi(int run, int lumi) {
    output_ << "\tbeginLumi " << run << "/" << lumi << "\n";
  }

  void MockEventProcessor::endLumi(int run, int lumi) {
    output_ << "\tendLumi " << run << "/" << lumi << "\n";
  }

  int MockEventProcessor::readAndCacheRun() {
    output_ << "\treadAndCacheRun " << run_ << "\n";
    return run_;
  }

  int MockEventProcessor::readAndCacheLumi() {
    output_ << "\treadAndCacheLumi " << lumi_ << "\n";
    return lumi_;
  }

  void MockEventProcessor::writeRun(int run) {
    output_ << "\twriteRun " << run << "\n";
  }

  void MockEventProcessor::deleteRunFromCache(int run) {
    output_ << "\tdeleteRunFromCache " << run << "\n";
  }

  void MockEventProcessor::writeLumi(int run, int lumi) {
    output_ << "\twriteLumi " << run << "/" << lumi << "\n";
  }

  void MockEventProcessor::deleteLumiFromCache(int run, int lumi) {
    output_ << "\tdeleteLumiFromCache " << run << "/" << lumi << "\n";
  }

  void MockEventProcessor::readEvent() {
    output_ << "\treadEvent\n";
  }

  void MockEventProcessor::processEvent() {
    output_ << "\tprocessEvent\n";
  }

  bool MockEventProcessor::shouldWeStop() const {
    output_ << "\tshouldWeStop\n";
    return shouldWeStop_;
  }

  void MockEventProcessor::setExceptionMessageFiles(std::string& message) { }
  void MockEventProcessor::setExceptionMessageRuns(std::string& message) { }
  void MockEventProcessor::setExceptionMessageLumis(std::string& message) { }

  bool MockEventProcessor::alreadyHandlingException() const { return false; }
}
