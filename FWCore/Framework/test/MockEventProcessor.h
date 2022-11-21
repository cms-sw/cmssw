#ifndef FWCore_Framework_MockEventProcessor_h
#define FWCore_Framework_MockEventProcessor_h

/*
Version of the Event Processor used for tests of
TransitionProcessors.icc.

The tests that use this class are less useful than they used
to be. MockEventProcessor is mainly used to test the code in
TransitionProcessors.icc (historical sidenote: at the time
MockEventProcessor was originally created a long time ago,
the functionality in TransitionProcessors.icc was
implemented using a boost state machine and MockEventProcessor
was originally designed to test that).  When
concurrent runs and concurrent lumis were implemented,
a lot of functionality was moved from TransitionProcessors.icc
into EventProcessors.cc. In the tests, MockEventProcessor
replaces EventProcessor and therefore it cannot be used
to test code in EventProcessor. Originally, this tested
the loops over runs, lumis, and events in addition to the
loops over files. At this point, it is really
testing only the code related to the loop over files in
TransitionProcessors.icc and we could clean things up by
removing the code and parts of the tests that are intended to
test runs, lumis, and events. That part of the code is not
serving any purpose anymore. This cleanup would be a lot
of tedious work for very little practical gain though...
It might never happen.

Original Authors: W. David Dagenhart, Marc Paterno
*/

#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"
#include "FWCore/Framework/interface/InputSource.h"

#include <exception>
#include <ostream>
#include <memory>
#include <sstream>
#include <string>

namespace edm {

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

    InputSource::ItemType processRuns();
    void processRun();
    InputSource::ItemType processLumis();

    void beginRun(RunNumber_t run);

    void endUnfinishedRun(bool);

    void endRun();

    void endUnfinishedLumi(bool);

    void readRun();
    void readAndMergeRun();
    LuminosityBlockNumber_t readLuminosityBlock();
    LuminosityBlockNumber_t readAndMergeLumi();
    void writeRun();
    void clearRunPrincipal();
    void writeLumi();
    void clearLumiPrincipal();

    bool shouldWeStop() const;

    void setExceptionMessageFiles(std::string& message);
    void setExceptionMessageRuns();
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

    bool lumiStatus_ = false;
    LuminosityBlockNumber_t currentLumiNumber_ = 0;
    bool didGlobalBeginLumiSucceed_ = false;
    InputSource::ItemType lastTransition_ = InputSource::IsInvalid;

    bool currentRun_ = false;
    RunNumber_t currentRunNumber_ = 0;
    bool didGlobalBeginRunSucceed_ = false;

    RunNumber_t nextRun_;
    LuminosityBlockNumber_t nextLumi_;

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
