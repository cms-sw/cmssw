#ifndef FWCore_Framework_IEventProcessor_h
#define FWCore_Framework_IEventProcessor_h

/*
$Id$

Abstract base class for Event Processors

Original Authors: W. David Dagenhart, Marc Paterno
*/

namespace statemachine {
  class Restart;
}

namespace edm
{
  class IEventProcessor {
  public:

    virtual ~IEventProcessor();

    virtual void runToCompletion() = 0;

    virtual void readFile() = 0;
    virtual void closeInputFile() = 0;
    virtual void openOutputFiles() = 0;
    virtual void closeOutputFiles() = 0;

    virtual void respondToOpenInputFile() = 0;
    virtual void respondToCloseInputFile() = 0;
    virtual void respondToOpenOutputFiles() = 0;
    virtual void respondToCloseOutputFiles() = 0;

    virtual void startingNewLoop() = 0;
    virtual bool endOfLoop() = 0;
    virtual void rewind() = 0;
    virtual void prepareForNextLoop() = 0;
    virtual void writeCache() = 0;
    virtual bool shouldWeCloseOutput() = 0;

    virtual void doErrorStuff() = 0;

    virtual void beginRun(int run) = 0;
    virtual void endRun(int run) = 0;

    virtual void beginLumi(int run, int lumi) = 0;
    virtual void endLumi(int run, int lumi) = 0;

    virtual int readAndCacheRun() = 0;
    virtual int readAndCacheLumi() = 0;
    virtual void writeRun(int run) = 0;
    virtual void deleteRunFromCache(int run) = 0;
    virtual void writeLumi(int run, int lumi) = 0;
    virtual void deleteLumiFromCache(int run, int lumi) = 0;

    virtual void readEvent() = 0;
    virtual void processEvent() = 0;
    virtual void writeEvent() = 0;
  };
}

#endif
