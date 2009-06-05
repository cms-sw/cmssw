#ifndef FWCore_Framework_MockEventProcessor_h
#define FWCore_Framework_MockEventProcessor_h

/*
Version of the Event Processor used for tests of
the state machine and other tests.

Original Authors: W. David Dagenhart, Marc Paterno
*/

#include "FWCore/Framework/interface/IEventProcessor.h"
#include "FWCore/Framework/src/EPStates.h"
#include <iostream>
#include <string>

namespace edm {
  class MockEventProcessor : public IEventProcessor {
  public:

    MockEventProcessor(std::string const& mockData,
                       std::ostream& output,
                       statemachine::FileMode const& fileMode,
                       bool handleEmptyRuns,
                       bool handleEmptyLumis);

    virtual StatusCode runToCompletion(bool onlineStateTransitions);
    virtual StatusCode runEventCount(int numberOfEventsToProcess);

    virtual void readFile();
    virtual void closeInputFile();
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
    virtual void writeLumiCache();
    virtual void writeRunCache();
    virtual bool shouldWeCloseOutput() const;

    virtual void doErrorStuff();

    virtual void beginRun(int run);
    virtual void endRun(int run);

    virtual void beginLumi(int run, int lumi);
    virtual void endLumi(int run, int lumi);

    virtual int readAndCacheRun();
    virtual int readAndCacheLumi();
    virtual void writeRun(int run);
    virtual void deleteRunFromCache(int run);
    virtual void writeLumi(int run, int lumi);
    virtual void deleteLumiFromCache(int run, int lumi);

    virtual void readAndProcessEvent();
    virtual bool shouldWeStop() const;

    virtual void setExceptionMessageFiles(std::string& message);
    virtual void setExceptionMessageRuns(std::string& message);
    virtual void setExceptionMessageLumis(std::string& message);

    virtual bool alreadyHandlingException() const;

  private:
    std::string mockData_;
    mutable std::ostream & output_;
    statemachine::FileMode fileMode_;
    bool handleEmptyRuns_;
    bool handleEmptyLumis_;

    int run_;
    int lumi_;

    bool shouldWeCloseOutput_;
    bool shouldWeEndLoop_;
    bool shouldWeStop_;
  };
}

#endif
