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
  class MockEventProcessor {
  public:
    class TestException : public std::exception {
    public:
      TestException() noexcept
      :std::exception() {}
    };
    
    MockEventProcessor(std::string const& mockData,
                       std::ostream& output,
                       bool iDoNotMerge);

    void runToCompletion();

    InputSource::ItemType nextTransitionType();
    std::pair<edm::ProcessHistoryID, edm::RunNumber_t> nextRunID();
    edm::LuminosityBlockNumber_t nextLuminosityBlockID();

    void readFile();
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

    void beginRun(ProcessHistoryID const& phid, RunNumber_t run);
    void endRun(ProcessHistoryID const& phid, RunNumber_t run, bool cleaningUpAfterException);

    void beginLumi(ProcessHistoryID const& phid, RunNumber_t run, LuminosityBlockNumber_t lumi);
    void endLumi(ProcessHistoryID const& phid, RunNumber_t run, LuminosityBlockNumber_t lumi, bool cleaningUpAfterException);

    std::pair<ProcessHistoryID,RunNumber_t> readRun();
    std::pair<ProcessHistoryID,RunNumber_t> readAndMergeRun();
    int readLuminosityBlock();
    int readAndMergeLumi();
    void writeRun(ProcessHistoryID const& phid, RunNumber_t run);
    void deleteRunFromCache(ProcessHistoryID const& phid, RunNumber_t run);
    void writeLumi(ProcessHistoryID const& phid, RunNumber_t run, LuminosityBlockNumber_t lumi);
    void deleteLumiFromCache(ProcessHistoryID const& phid, RunNumber_t run, LuminosityBlockNumber_t lumi);

    bool shouldWeStop() const;

    void setExceptionMessageFiles(std::string& message);
    void setExceptionMessageRuns(std::string& message);
    void setExceptionMessageLumis(std::string& message);

    InputSource::ItemType readAndProcessEvents();

    bool setDeferredException(std::exception_ptr);

  private:
    void readAndProcessEvent();
    void throwIfNeeded();

    std::string mockData_;
    std::ostream & output_;
    std::istringstream input_;
    
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
}

#endif
