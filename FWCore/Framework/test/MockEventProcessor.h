#ifndef FWCore_Framework_MockEventProcessor_h
#define FWCore_Framework_MockEventProcessor_h

/*
Version of the Event Processor used for tests of
the state machine and other tests.

Original Authors: W. David Dagenhart, Marc Paterno
*/

#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "FWCore/Framework/interface/InputSource.h"

#include <iostream>
#include <string>
#include <sstream>
#include <exception>

namespace edm {
  class LuminosityBlockProcessingStatus;

  class MockEventProcessor {
  public:
    class TestException : public std::exception {
    public:
      TestException() noexcept : std::exception() {}
    };

    MockEventProcessor(std::string const& mockData, std::ostream& output, bool iDoNotMerge);

    void runToCompletion();

    InputSource::ItemType nextTransitionType();
    InputSource::ItemType lastTransitionType() const;
    std::pair<edm::ProcessHistoryID, edm::RunNumber_t> nextRunID();
    edm::LuminosityBlockNumber_t nextLuminosityBlockID();

    void readFile();
    bool fileBlockValid() { return true; }
    void closeInputFile(bool cleaningUpAfterException);
    void openOutputFiles();
    void closeOutputFiles();

    void respondToOpenInputFile();
    void respondToCloseInputFile();

    void startingNewLoop();
    bool endOfLoop();
    void rewindInput();
    void prepareForNextLoop();
    bool shouldWeCloseOutput() const;

    void doErrorStuff();

    void beginProcessBlock(bool& beginProcessBlockSucceeded);
    void inputProcessBlocks();
    void endProcessBlock(bool cleaningUpAfterException, bool beginProcessBlockSucceeded);

    void beginRun(ProcessHistoryID const& phid,
                  RunNumber_t run,
                  bool& globalTransitionSucceeded,
                  bool& eventSetupForInstanceSucceeded);
    void endUnfinishedRun(ProcessHistoryID const& phid,
                          RunNumber_t run,
                          bool globalTranstitionSucceeded,
                          bool cleaningUpAfterException,
                          bool eventSetupForInstanceSucceeded);

    void endRun(ProcessHistoryID const& phid,
                RunNumber_t run,
                bool globalTranstitionSucceeded,
                bool cleaningUpAfterException);

    InputSource::ItemType processLumis(std::shared_ptr<void>);
    void endUnfinishedLumi();

    std::pair<ProcessHistoryID, RunNumber_t> readRun();
    std::pair<ProcessHistoryID, RunNumber_t> readAndMergeRun();
    int readLuminosityBlock(LuminosityBlockProcessingStatus&);
    int readAndMergeLumi(LuminosityBlockProcessingStatus&);
    void writeRun(ProcessHistoryID const& phid, RunNumber_t run);
    void deleteRunFromCache(ProcessHistoryID const& phid, RunNumber_t run);
    void writeLumi(LuminosityBlockProcessingStatus&);
    void deleteLumiFromCache(LuminosityBlockProcessingStatus&);

    bool shouldWeStop() const;

    void setExceptionMessageFiles(std::string& message);
    void setExceptionMessageRuns(std::string& message);
    void setExceptionMessageLumis();

    bool setDeferredException(std::exception_ptr);

  private:
    InputSource::ItemType readAndProcessEvents();
    void readAndProcessEvent();
    void throwIfNeeded();
    void endLumi();

    std::string mockData_;
    std::ostream& output_;
    std::istringstream input_;

    std::shared_ptr<LuminosityBlockProcessingStatus> lumiStatus_;
    InputSource::ItemType lastTransition_;

    int run_;
    int lumi_;

    bool doNotMerge_;
    bool shouldWeCloseOutput_;
    bool shouldWeEndLoop_;
    bool shouldWeStop_;
    bool eventProcessed_;
    bool reachedEndOfInput_;
    bool shouldThrow_;
  };
}  // namespace edm

#endif
