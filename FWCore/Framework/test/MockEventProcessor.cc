
/*
$Id$
*/

#include "FWCore/Framework/test/MockEventProcessor.h"

#include <iostream>
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
                                         const statemachine::Filemode& filemode,
                                         bool handleEmptyRuns,
                                         bool handleEmptyLumis) :
    mockData_(mockData),
    output_(output),
    filemode_(filemode),
    handleEmptyRuns_(handleEmptyRuns),
    handleEmptyLumis_(handleEmptyLumis),
    shouldWeCloseOutput_(true),
    shouldWeEndLoop_(true)  {
  }

  void MockEventProcessor::runToCompletion() {
    statemachine::Machine myMachine(this,
                                    filemode_,
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
        myMachine.process_event( statemachine::Event() );
      }
      else if (ch == 'f') {
        output_ << "    *** nextItemType: File " << t.value << " ***\n";
        if (t.value == 0) shouldWeCloseOutput_ = false;
        else shouldWeCloseOutput_ = true;
        myMachine.process_event( statemachine::File() );
      }
      else if (ch == 's') {
        output_ << "    *** nextItemType: Stop " << t.value << " ***\n";
        shouldWeEndLoop_ = t.value;
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

  void MockEventProcessor::rewind() {
    output_ << "\trewind\n";
  }

  void MockEventProcessor::prepareForNextLoop() {
    output_ << "\tprepareForNextLoop\n";
  }

  void MockEventProcessor::writeCache() {
    output_ << "\twriteCache\n";
  }

  bool MockEventProcessor::shouldWeCloseOutput() {
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

  void MockEventProcessor::writeEvent() {
    output_ << "\twriteEvent\n";
  }
}
