#ifndef FWCore_Framework_MockEventProcessor_h
#define FWCore_Framework_MockEventProcessor_h

/*
Version of the Event Processor used for tests of
the state machine and other tests.

Original Authors: W. David Dagenhart, Marc Paterno
*/

#include "FWCore/Framework/interface/IEventProcessor.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "FWCore/Framework/interface/InputSource.h"

#include <iostream>
#include <string>
#include <sstream>

namespace edm {
  class MockEventProcessor : public IEventProcessor {
  public:

    MockEventProcessor(std::string const& mockData,
                       std::ostream& output,
                       bool iDoNotMerge);

    virtual StatusCode runToCompletion() override;

    InputSource::ItemType nextTransitionType();
    std::pair<edm::ProcessHistoryID, edm::RunNumber_t> nextRunID();
    edm::LuminosityBlockNumber_t nextLuminosityBlockID();

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

    virtual void beginRun(ProcessHistoryID const& phid, RunNumber_t run) override;
    virtual void endRun(ProcessHistoryID const& phid, RunNumber_t run, bool cleaningUpAfterException) override;

    virtual void beginLumi(ProcessHistoryID const& phid, RunNumber_t run, LuminosityBlockNumber_t lumi) override;
    virtual void endLumi(ProcessHistoryID const& phid, RunNumber_t run, LuminosityBlockNumber_t lumi, bool cleaningUpAfterException) override;

    virtual std::pair<ProcessHistoryID,RunNumber_t> readRun() override;
    virtual std::pair<ProcessHistoryID,RunNumber_t> readAndMergeRun() override;
    virtual int readLuminosityBlock() override;
    virtual int readAndMergeLumi() override;
    virtual void writeRun(ProcessHistoryID const& phid, RunNumber_t run) override;
    virtual void deleteRunFromCache(ProcessHistoryID const& phid, RunNumber_t run) override;
    virtual void writeLumi(ProcessHistoryID const& phid, RunNumber_t run, LuminosityBlockNumber_t lumi) override;
    virtual void deleteLumiFromCache(ProcessHistoryID const& phid, RunNumber_t run, LuminosityBlockNumber_t lumi) override;

    virtual void readAndProcessEvent() override;
    virtual bool shouldWeStop() const override;

    virtual void setExceptionMessageFiles(std::string& message) override;
    virtual void setExceptionMessageRuns(std::string& message) override;
    virtual void setExceptionMessageLumis(std::string& message) override;

    virtual bool alreadyHandlingException() const override;

    InputSource::ItemType readAndProcessEvents();

    bool setDeferredException(std::exception_ptr);

  private:
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
  };
}

#endif
