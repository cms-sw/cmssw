#ifndef FWCore_Framework_MockEventProcessor_h
#define FWCore_Framework_MockEventProcessor_h

/*
Version of the Event Processor used for tests of
the state machine and other tests.

Original Authors: W. David Dagenhart, Marc Paterno
*/

#include "FWCore/Framework/interface/IEventProcessor.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "FWCore/Framework/src/EPStates.h"

#include <iostream>
#include <string>

namespace edm {
  class MockEventProcessor : public IEventProcessor {
  public:

    MockEventProcessor(std::string const& mockData,
                       std::ostream& output,
                       statemachine::FileMode const& fileMode,
                       statemachine::EmptyRunLumiMode const& emptyRunLumiMode);

    StatusCode runToCompletion() override;

    void readFile() override;
    void closeInputFile(bool cleaningUpAfterException) override;
    void openOutputFiles() override;
    void closeOutputFiles() override;

    void respondToOpenInputFile() override;
    void respondToCloseInputFile() override;

    void startingNewLoop() override;
    bool endOfLoop() override;
    void rewindInput() override;
    void prepareForNextLoop() override;
    bool shouldWeCloseOutput() const override;

    void doErrorStuff() override;

    void beginRun(statemachine::Run const& run) override;
    void endRun(statemachine::Run const& run, bool cleaningUpAfterException) override;

    void beginLumi(ProcessHistoryID const& phid, RunNumber_t run, LuminosityBlockNumber_t lumi) override;
    void endLumi(ProcessHistoryID const& phid, RunNumber_t run, LuminosityBlockNumber_t lumi, bool cleaningUpAfterException) override;

    statemachine::Run readRun() override;
    statemachine::Run readAndMergeRun() override;
    int readLuminosityBlock() override;
    int readAndMergeLumi() override;
    void writeRun(statemachine::Run const& run) override;
    void deleteRunFromCache(statemachine::Run const& run) override;
    void writeLumi(ProcessHistoryID const& phid, RunNumber_t run, LuminosityBlockNumber_t lumi) override;
    void deleteLumiFromCache(ProcessHistoryID const& phid, RunNumber_t run, LuminosityBlockNumber_t lumi) override;

    void readAndProcessEvent() override;
    bool shouldWeStop() const override;

    void setExceptionMessageFiles(std::string& message) override;
    void setExceptionMessageRuns(std::string& message) override;
    void setExceptionMessageLumis(std::string& message) override;

    bool alreadyHandlingException() const override;

  private:
    std::string mockData_;
    std::ostream & output_;
    statemachine::FileMode fileMode_;
    statemachine::EmptyRunLumiMode emptyRunLumiMode_;

    int run_;
    int lumi_;

    bool shouldWeCloseOutput_;
    bool shouldWeEndLoop_;
    bool shouldWeStop_;
  };
}

#endif
