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

    virtual StatusCode runToCompletion(bool onlineStateTransitions);

    virtual void readFile();
    virtual void closeInputFile(bool cleaningUpAfterException);
    virtual void openOutputFiles();
    virtual void closeOutputFiles();

    virtual void respondToOpenInputFile();
    virtual void respondToCloseInputFile();
    virtual void respondToOpenOutputFiles();
    virtual void respondToCloseOutputFiles();

    virtual void startingNewLoop();
    virtual bool endOfLoop();
    virtual void rewindInput();
    virtual void prepareForNextLoop();
    virtual bool shouldWeCloseOutput() const;

    virtual void doErrorStuff();

    virtual void beginRun(statemachine::Run const& run);
    virtual void endRun(statemachine::Run const& run, bool cleaningUpAfterException);

    virtual void beginLumi(ProcessHistoryID const& phid, int run, int lumi);
    virtual void endLumi(ProcessHistoryID const& phid, int run, int lumi, bool cleaningUpAfterException);

    virtual statemachine::Run readAndCacheRun(bool merge);
    virtual int readAndCacheLumi(bool merge);
    virtual void writeRun(statemachine::Run const& run);
    virtual void deleteRunFromCache(statemachine::Run const& run);
    virtual void writeLumi(ProcessHistoryID const& phid, int run, int lumi);
    virtual void deleteLumiFromCache(ProcessHistoryID const& phid, int run, int lumi);

    virtual void readAndProcessEvent();
    virtual bool shouldWeStop() const;

    virtual void setExceptionMessageFiles(std::string& message);
    virtual void setExceptionMessageRuns(std::string& message);
    virtual void setExceptionMessageLumis(std::string& message);

    virtual bool alreadyHandlingException() const;

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
