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

    virtual StatusCode runToCompletion() override;

    virtual void readFile() override;
    virtual void closeInputFile(bool cleaningUpAfterException) override;
    virtual void openOutputFiles() override;
    virtual void closeOutputFiles() override;

    virtual void respondToOpenInputFile() override;
    virtual void respondToCloseInputFile() override;

    virtual void startingNewLoop() override;
    virtual bool endOfLoop() override;
    virtual void rewindInput() override;
    virtual void prepareForNextLoop() override;
    virtual bool shouldWeCloseOutput() const override;

    virtual void doErrorStuff() override;

    virtual void beginRun(statemachine::Run const& run) override;
    virtual void endRun(statemachine::Run const& run, bool cleaningUpAfterException) override;

    virtual void beginLumi(ProcessHistoryID const& phid, RunNumber_t run, LuminosityBlockNumber_t lumi) override;
    virtual void endLumi(ProcessHistoryID const& phid, RunNumber_t run, LuminosityBlockNumber_t lumi, bool cleaningUpAfterException) override;

    virtual statemachine::Run readRun() override;
    virtual statemachine::Run readAndMergeRun() override;
    virtual int readLuminosityBlock() override;
    virtual int readAndMergeLumi() override;
    virtual void writeRun(statemachine::Run const& run) override;
    virtual void deleteRunFromCache(statemachine::Run const& run) override;
    virtual void writeLumi(ProcessHistoryID const& phid, RunNumber_t run, LuminosityBlockNumber_t lumi) override;
    virtual void deleteLumiFromCache(ProcessHistoryID const& phid, RunNumber_t run, LuminosityBlockNumber_t lumi) override;

    virtual void readAndProcessEvent() override;
    virtual bool shouldWeStop() const override;

    virtual void setExceptionMessageFiles(std::string& message) override;
    virtual void setExceptionMessageRuns(std::string& message) override;
    virtual void setExceptionMessageLumis(std::string& message) override;

    virtual bool alreadyHandlingException() const override;

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
