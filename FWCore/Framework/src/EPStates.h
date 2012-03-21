#ifndef Framework_EPStates_h
#define Framework_EPStates_h

/*
$Id: EPStates.h,v 1.10 2010/08/05 16:05:49 wdd Exp $

The state machine that controls the processing of runs, luminosity
blocks, events, and loops is implemented using the boost statechart
library and the states and events defined here.  This machine is
used by the EventProcessor.

Original Authors: W. David Dagenhart, Marc Paterno
*/

#include "DataFormats/Provenance/interface/ProcessHistoryID.h"

#include "boost/statechart/event.hpp"
#include "boost/statechart/state_machine.hpp"
#include <boost/statechart/state.hpp>
#include <boost/statechart/transition.hpp>
#include <boost/mpl/list.hpp>
#include <boost/statechart/custom_reaction.hpp>
#include <vector>

namespace sc = boost::statechart;
namespace mpl = boost::mpl;

namespace edm {
  class IEventProcessor;
}

namespace statemachine {

  enum FileMode { NOMERGE, FULLMERGE };

  enum EmptyRunLumiMode { handleEmptyRunsAndLumis,
                          handleEmptyRuns,
                          doNotHandleEmptyRunsAndLumis
  };

  // Define the classes representing the "boost statechart events".
  // There are six of them.

  class Run : public sc::event<Run> {
  public:
    Run(edm::ProcessHistoryID const& phid, int runNumber);
    edm::ProcessHistoryID const& processHistoryID() const { return processHistoryID_; }
    int runNumber() const { return runNumber_; }

    bool operator==(Run const& rh) const {
      return (runNumber_ == rh.runNumber()) &&
	     (processHistoryID_ == rh.processHistoryID());
    }

    bool operator!=(Run const& rh) const {
      return (runNumber_ != rh.runNumber()) ||
	     (processHistoryID_ != rh.processHistoryID());
    }

  private:
    edm::ProcessHistoryID processHistoryID_;
    int runNumber_;
  };

  class Lumi : public sc::event<Lumi> {
  public:
    Lumi(int id);
    int id() const { return id_; }
  private:
    int id_;
  };

  // It is slightly confusing that this one refers to 
  // both physics event and a boost statechart event ...
  class Event : public sc::event<Event> { };

  class File : public sc::event<File> {};
  class Stop : public sc::event<Stop> {};
  class Restart : public sc::event<Restart> {};

  // Now define the machine and the states.
  // For all these classes, the first template argument
  // to the base class is the derived class.  The second
  // argument is the parent state or if it is a top level
  // state the Machine.  If there is a third template
  // argument it is the substate that is entered
  // by default on entry.

  class Starting;

  class Machine : public sc::state_machine<Machine, Starting>
  {
  public:
    Machine(edm::IEventProcessor* ep,
            FileMode fileMode,
            EmptyRunLumiMode emptyRunLumiMode);

    edm::IEventProcessor& ep() const;
    FileMode fileMode() const;
    EmptyRunLumiMode emptyRunLumiMode() const;

    void startingNewLoop(File const& file);
    void startingNewLoop(Stop const& stop);
    void rewindAndPrepareForNextLoop(Restart const& restart);

  private:

    edm::IEventProcessor* ep_;
    FileMode fileMode_;
    EmptyRunLumiMode emptyRunLumiMode_;
  };

  class Error;
  class HandleFiles;
  class EndingLoop;

  class Starting : public sc::state<Starting, Machine>
  {
  public:
    Starting(my_context ctx);
    ~Starting();
    
    typedef mpl::list<
      sc::transition<Event, Error>,
      sc::transition<Lumi, Error>,
      sc::transition<Run, Error>,
      sc::transition<File, HandleFiles, Machine, &Machine::startingNewLoop>,
      sc::transition<Stop, EndingLoop, Machine, &Machine::startingNewLoop>,
      sc::transition<Restart, Error> > reactions;
  };

  class FirstFile;

  class HandleFiles : public sc::state<HandleFiles, Machine, FirstFile>
  {
  public:
    HandleFiles(my_context ctx);
    void exit();
    ~HandleFiles();
 
    typedef mpl::list<
      sc::transition<Event, Error>,
      sc::transition<Lumi, Error>,
      sc::transition<Run, Error>,
      sc::transition<File, Error>,
      sc::transition<Stop, EndingLoop>,
      sc::transition<Restart, Error> > reactions;

    void closeFiles(bool cleaningUpAfterException);
    void goToNewInputFile();
    bool shouldWeCloseOutput();
  private:
    edm::IEventProcessor & ep_;
    bool exitCalled_;
  };

  class EndingLoop : public sc::state<EndingLoop, Machine>
  {
  public:
    EndingLoop(my_context ctx);
    ~EndingLoop();
    typedef mpl::list<
      sc::transition<Restart, Starting, Machine, &Machine::rewindAndPrepareForNextLoop>,
      sc::custom_reaction<Stop> > reactions;

    sc::result react(Stop const&);
  private:
    edm::IEventProcessor & ep_;
  };

  class Error : public sc::state<Error, Machine>
  {
  public:
    Error(my_context ctx);
    ~Error();
    typedef sc::transition<Stop, EndingLoop> reactions;
  private:
    edm::IEventProcessor & ep_;
  };

  class HandleRuns;

  class FirstFile : public sc::state<FirstFile, HandleFiles>
  {
  public:
    FirstFile(my_context ctx);
    ~FirstFile();
    
    typedef mpl::list<
      sc::transition<Run, HandleRuns>,
      sc::custom_reaction<File> > reactions;

    sc::result react(File const& file);
    void openFiles();
  private:
    edm::IEventProcessor & ep_;
  };

  class HandleNewInputFile1 : public sc::state<HandleNewInputFile1, HandleFiles>
  {
  public:
    HandleNewInputFile1(my_context ctx);
    ~HandleNewInputFile1();

    typedef mpl::list<
      sc::transition<Run, HandleRuns>,
      sc::custom_reaction<File> > reactions;

    sc::result react(File const& file);
  };

  class NewInputAndOutputFiles : public sc::state<NewInputAndOutputFiles, HandleFiles>
  {
  public:
    NewInputAndOutputFiles(my_context ctx);
    ~NewInputAndOutputFiles();

    typedef mpl::list<
      sc::transition<Run, HandleRuns>,
      sc::custom_reaction<File> > reactions;

    sc::result react(File const& file);

  private:

    void goToNewInputAndOutputFiles();

    edm::IEventProcessor & ep_;
  };

  class NewRun;

  class HandleRuns : public sc::state<HandleRuns, HandleFiles, NewRun>
  {
  public:
    HandleRuns(my_context ctx);
    void exit();
    ~HandleRuns();

    typedef sc::transition<File, NewInputAndOutputFiles> reactions;

    bool beginRunCalled() const;
    Run const& currentRun() const;
    bool runException() const;
    void setupCurrentRun();
    void beginRun(Run const& run);
    void endRun(Run const& run, bool cleaningUpAfterException);
    void finalizeRun(Run const&);
    void finalizeRun(bool cleaningUpAfterException);
    void beginRunIfNotDoneAlready();
  private:
    edm::IEventProcessor & ep_;
    bool exitCalled_;
    bool beginRunCalled_;
    Run currentRun_;
    bool runException_;
  };

  class HandleLumis;

  class NewRun : public sc::state<NewRun, HandleRuns>
  {
  public:
    NewRun(my_context ctx);
    ~NewRun();

    typedef mpl::list<
      sc::transition<Lumi, HandleLumis>,
      sc::custom_reaction<Run>,
      sc::custom_reaction<File> > reactions;

    sc::result react(Run const& run);
    sc::result react(File const& file);
  };

  class ContinueRun1;

  class HandleNewInputFile2 : public sc::state<HandleNewInputFile2, HandleRuns>
  {
  public:
    HandleNewInputFile2(my_context ctx);
    ~HandleNewInputFile2();
    bool checkInvariant();

    typedef mpl::list<
      sc::custom_reaction<Run>,
      sc::custom_reaction<File> > reactions;

    sc::result react(Run const& run);
    sc::result react(File const& file);
  };

  class ContinueRun1 : public sc::state<ContinueRun1, HandleRuns>
  {
  public:
    ContinueRun1(my_context ctx);
    ~ContinueRun1();
    bool checkInvariant();

    typedef mpl::list<
      sc::custom_reaction<Run>,
      sc::custom_reaction<File>,
      sc::transition<Lumi, HandleLumis> > reactions;

    sc::result react(Run const& run);
    sc::result react(File const& file);
  private:
    edm::IEventProcessor & ep_;
  }; 

  class FirstLumi;

  class HandleLumis : public sc::state<HandleLumis, HandleRuns, FirstLumi>
  {
  public:
    class LumiID {
    public:
      LumiID(edm::ProcessHistoryID const& phid, int run, int lumi);
      edm::ProcessHistoryID const& processHistoryID() const { return processHistoryID_; }
      int run() const { return run_; }
      int lumi() const { return lumi_; }

    private:
      edm::ProcessHistoryID processHistoryID_;
      int run_;
      int lumi_;
    };
    HandleLumis(my_context ctx);
    void exit();
    ~HandleLumis();
    bool checkInvariant();

    LumiID const& currentLumi() const;
    bool currentLumiEmpty() const;
    void setupCurrentLumi();
    void finalizeLumi(bool cleaningUpAfterException);
    void markLumiNonEmpty();

    typedef sc::transition<Run, NewRun, HandleRuns, &HandleRuns::finalizeRun> reactions;

  private:
    edm::IEventProcessor & ep_;
    bool exitCalled_;
    bool currentLumiEmpty_;
    LumiID currentLumi_;
    bool lumiException_;
  };

  class HandleEvent;
  class AnotherLumi;

  class FirstLumi : public sc::state<FirstLumi, HandleLumis>
  {
  public:
    FirstLumi(my_context ctx);
    ~FirstLumi();
    bool checkInvariant();

    typedef mpl::list<
      sc::transition<Event, HandleEvent>,
      sc::custom_reaction<Lumi>,
      sc::custom_reaction<File> > reactions;

    sc::result react(Lumi const& lumi);
    sc::result react(File const& file);
  };

  class AnotherLumi : public sc::state<AnotherLumi, HandleLumis>
  {
  public:
    AnotherLumi(my_context ctx);
    ~AnotherLumi();
    bool checkInvariant();

    typedef mpl::list<
      sc::transition<Event, HandleEvent>,
      sc::custom_reaction<Lumi>,
      sc::custom_reaction<File> > reactions;

    sc::result react(Lumi const& lumi);
    sc::result react(File const& file);
  };

  class HandleEvent : public sc::state<HandleEvent, HandleLumis>
  {
  public:
    HandleEvent(my_context ctx);
    ~HandleEvent();
    bool checkInvariant();

    typedef mpl::list<
      sc::transition<Event, HandleEvent>,
      sc::transition<Lumi, AnotherLumi>,
      sc::custom_reaction<File> > reactions;

    sc::result react(File const& file);
    void readAndProcessEvent();
    void markNonEmpty();
  private:
    edm::IEventProcessor & ep_;
  };

  class HandleNewInputFile3 : public sc::state<HandleNewInputFile3, HandleLumis>
  {
  public:
    HandleNewInputFile3(my_context ctx);
    ~HandleNewInputFile3();
    bool checkInvariant();

    typedef mpl::list<
      sc::custom_reaction<Run>,
      sc::custom_reaction<File> > reactions;

    sc::result react(Run const& run);
    sc::result react(File const& file);
  };

  class ContinueRun2 : public sc::state<ContinueRun2, HandleLumis>
  {
  public:
    ContinueRun2(my_context ctx);
    ~ContinueRun2();
    bool checkInvariant();

    typedef mpl::list<
      sc::custom_reaction<Run>,
      sc::custom_reaction<Lumi>,
      sc::custom_reaction<File> > reactions;

    sc::result react(Run const& run);
    sc::result react(Lumi const& lumi);
    sc::result react(File const& file);
  private:
    edm::IEventProcessor & ep_;
  };

  class ContinueLumi : public sc::state<ContinueLumi, HandleLumis>
  {
  public:
    ContinueLumi(my_context ctx);
    ~ContinueLumi();
    bool checkInvariant();

    typedef mpl::list<
      sc::transition<Event, HandleEvent>,
      sc::custom_reaction<Lumi>,
      sc::custom_reaction<File> > reactions;

    sc::result react(Lumi const& lumi);
    sc::result react(File const& file);
  private:
    edm::IEventProcessor & ep_;
  };
}

#endif
