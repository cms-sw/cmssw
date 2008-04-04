#ifndef FWCore_Framework_MockEventProcessor_h
#define FWCore_Framework_MockEventProcessor_h

/*
$Id: MockEventProcessor.h,v 1.5 2008/03/18 18:41:29 wdd Exp $

Version of the Event Processor used for tests of
the state machine and other tests.

Original Authors: W. David Dagenhart, Marc Paterno
*/

#include "FWCore/Framework/interface/IEventProcessor.h"
#include "FWCore/Framework/src/EPStates.h"
#include <iostream>
#include <string>

namespace edm
{
  class MockEventProcessor : public IEventProcessor {
  public:

    MockEventProcessor(const std::string& mockData,
                       std::ostream& output,
                       const statemachine::FileMode& fileMode,
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
    virtual void writeCache();
    virtual bool shouldWeCloseOutput();

    virtual void doErrorStuff();

    virtual void smBeginRun(int run);
    virtual void smEndRun(int run);

    virtual void beginLumi(int run, int lumi);
    virtual void endLumi(int run, int lumi);

    virtual int readAndCacheRun();
    virtual int readAndCacheLumi();
    virtual void writeRun(int run);
    virtual void deleteRunFromCache(int run);
    virtual void writeLumi(int run, int lumi);
    virtual void deleteLumiFromCache(int run, int lumi);

    virtual void readEvent();
    virtual void processEvent();
    virtual bool shouldWeStop();

  private:
    std::string mockData_;
    std::ostream & output_;
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
