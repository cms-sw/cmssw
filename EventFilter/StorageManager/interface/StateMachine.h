// $Id: StateMachine.h,v 1.2 2009/06/10 08:15:23 dshpakov Exp $

#ifndef STATEMACHINE_H
#define STATEMACHINE_H

#include "EventFilter/StorageManager/interface/SharedResources.h"

#include <boost/statechart/event.hpp>
#include <boost/statechart/in_state_reaction.hpp>
#include <boost/statechart/state_machine.hpp>
#include <boost/statechart/state.hpp>
#include <boost/statechart/transition.hpp>
#include <boost/mpl/list.hpp>

#include <iostream>
#include <string>
#include <vector>


namespace bsc = boost::statechart;

namespace stor
{

  // Simple file-based debugging. Will remove when no longer needed.
  void sm_debug( const std::string& file_name_suffix, const std::string& message );

  class I2OChain;
  class DiskWriter;
  class EventDistributor;
  class FragmentStore;
  class Notifier;
  class TransitionRecord;

  ////////////////////////////////////////////////
  //// Forward declarations of state classes: ////
  ////////////////////////////////////////////////

  // Outer states:
  class Failed;
  class Normal;

  // Inner states of Normal:
  class Halted;
  class Ready;
  class Stopped;
  class Enabled;

  // Inner states of Enabled:
  class Starting;
  class Stopping;
  class Halting;
  class Running;

  // Inner states of Running:
  class Processing;
  class DrainingQueues;
  class FinishingDQM;


  ////////////////////////////
  //// Transition events: ////
  ////////////////////////////

  class Configure : public bsc::event<Configure> {};
  class Enable : public bsc::event<Enable> {};
  class Stop : public bsc::event<Stop> {};
  class Halt : public bsc::event<Halt> {};
  class Fail : public bsc::event<Fail> {};
  class Reconfigure : public bsc::event<Reconfigure> {};
  class EmergencyStop : public bsc::event<EmergencyStop> {};
  class QueuesEmpty : public bsc::event<QueuesEmpty> {};
  class StartRun : public bsc::event<StartRun> {};
  class EndRun : public bsc::event<EndRun> {};
  class StopDone : public bsc::event<StopDone> {};
  class HaltDone : public bsc::event<HaltDone> {};

  ////////////////////////////////////////////////////////
  //// Operations -- abstract base for state classes: ////
  ////////////////////////////////////////////////////////

  class Operations
  {

  public:

    Operations();
    virtual ~Operations() = 0;
    void processI2OFragment( I2OChain& frag ) const;

    void noFragmentToProcess() const;

    std::string stateName() const;

  protected:

    virtual void do_processI2OFragment( I2OChain& frag ) const;

    virtual void do_noFragmentToProcess() const;

    virtual std::string do_stateName() const = 0;

    void safeEntryAction( Notifier* );
    virtual void do_entryActionWork() = 0;

    void safeExitAction( Notifier* );
    virtual void do_exitActionWork() = 0; 

  };


  ///////////////////////
  //// StateMachine: ////
  ///////////////////////

  class StateMachine: public bsc::state_machine<StateMachine,Normal>
  {

  public:

    StateMachine( EventDistributor* ed,
                  FragmentStore* fs,
                  Notifier* n,
                  SharedResourcesPtr sr );

    //void processI2OFragment();
    std::string getCurrentStateName() const;
    Operations const& getCurrentState() const;

    void updateHistory( const TransitionRecord& tr );

    EventDistributor* getEventDistributor() const { return _eventDistributor; }
    FragmentStore* getFragmentStore() const { return _fragmentStore; }
    Notifier* getNotifier() { return _notifier; }
    SharedResourcesPtr getSharedResources() const { return _sharedResources; }

    void unconsumed_event( bsc::event_base const& );

    // Remi May 14, 2009: not clear why we originally introduced the _initialized
    // void declareInitialized() { _initialized = true; }

    void setExternallyVisibleState( const std::string& );

  private:

    DiskWriter* _diskWriter;
    EventDistributor* _eventDistributor;
    FragmentStore* _fragmentStore;
    Notifier* _notifier;
    SharedResourcesPtr _sharedResources;

    // Remi May 14, 2009: not clear why we originally introduced the _initialized
    // bool _initialized; // to control access to state name

  };

  ////////////////////////
  //// State classes: ////
  ////////////////////////

  // Failed:
  class Failed: public bsc::state<Failed,StateMachine>, public Operations
  {

  public:

    Failed( my_context );
    virtual ~Failed();

  private:

    virtual std::string do_stateName() const;
    virtual void do_entryActionWork();
    virtual void do_exitActionWork();

  };

  // Normal:
  class Normal: public bsc::state<Normal,StateMachine,Halted>, public Operations
  {

  public:

    typedef bsc::transition<Fail,Failed> FT;
    typedef boost::mpl::list<FT> reactions;

    Normal( my_context );
    virtual ~Normal();

  private:

    virtual std::string do_stateName() const;
    virtual void do_entryActionWork();
    virtual void do_exitActionWork();

  };

  // Halted:
  class Halted: public bsc::state<Halted,Normal>, public Operations
  {

  public:

    typedef bsc::transition<Configure,Ready> RT;
    typedef boost::mpl::list<RT> reactions;

    Halted( my_context );
    virtual ~Halted();

  private:

    virtual std::string do_stateName() const;
    virtual void do_entryActionWork();
    virtual void do_exitActionWork();

  };

  // Ready:
  class Ready: public bsc::state<Ready,Normal,Stopped>, public Operations
  {

  public:

    typedef bsc::transition<Reconfigure,Stopped> ST;
    typedef bsc::transition<Halt,Halted> HT;
    typedef boost::mpl::list<ST,HT> reactions;

    Ready( my_context );
    virtual ~Ready();

  private:

    virtual std::string do_stateName() const;
    virtual void do_entryActionWork();
    virtual void do_exitActionWork();

  };

  // Stopped:
  class Stopped: public bsc::state<Stopped,Ready>, public Operations
  {

  public:

    typedef bsc::transition<Enable,Enabled> ET;
    typedef boost::mpl::list<ET> reactions;

    Stopped( my_context );
    virtual ~Stopped();
    
  private:

    virtual std::string do_stateName() const;
    virtual void do_entryActionWork();
    virtual void do_exitActionWork();

  };

  // Enabled:
  class Enabled: public bsc::state<Enabled,Ready,Starting>, public Operations
  {

  public:

    void logReconfigureRequest( const Reconfigure& request );

    //    typedef bsc::transition<EmergencyStop,Stopped> ET;
    typedef bsc::transition<StopDone,Stopped> DT;
    typedef bsc::transition<HaltDone,Halted> HT;
    typedef bsc::in_state_reaction<Reconfigure,Enabled,&Enabled::logReconfigureRequest> RecfgIR;
    typedef boost::mpl::list<DT,HT,RecfgIR> reactions;

    Enabled( my_context );
    virtual ~Enabled();

  private:

    virtual std::string do_stateName() const;
    virtual void do_entryActionWork();
    virtual void do_exitActionWork();

  };

  // Starting:
  class Starting: public bsc::state<Starting,Enabled>, public Operations
  {

  public:

    void logStopDoneRequest( const StopDone& request );
    void logHaltDoneRequest( const HaltDone& request );

    typedef bsc::transition<StartRun,Running> ST;
    typedef bsc::transition<EmergencyStop,Stopping> ET;
    typedef bsc::in_state_reaction<StopDone,Starting,&Starting::logStopDoneRequest> StopDoneIR;
    typedef bsc::in_state_reaction<HaltDone,Starting,&Starting::logHaltDoneRequest> HaltDoneIR;
    typedef boost::mpl::list<ST,ET,StopDoneIR,HaltDoneIR> reactions;

    Starting( my_context );
    virtual ~Starting();

  private:

    virtual std::string do_stateName() const;
    virtual void do_noFragmentToProcess() const;
    virtual void do_entryActionWork();
    virtual void do_exitActionWork();

    bool workerThreadsConfigured() const;

  };

  // Stopping:
  class Stopping: public bsc::state<Stopping,Enabled>, public Operations
  {

  public:

    void logHaltDoneRequest( const HaltDone& request );

    typedef bsc::transition<StopDone,Stopped> ST;
    typedef bsc::in_state_reaction<HaltDone,Stopping,&Stopping::logHaltDoneRequest> HaltDoneIR;
    typedef boost::mpl::list<ST,HaltDoneIR> reactions;

    Stopping( my_context );
    virtual ~Stopping();

  private:

    virtual std::string do_stateName() const;
    virtual void do_noFragmentToProcess() const;
    virtual void do_entryActionWork();
    virtual void do_exitActionWork();

    bool destructionIsDone() const;

  };

  // Halting:
  class Halting: public bsc::state<Halting,Enabled>, public Operations
  {

  public:

    void logStopDoneRequest( const StopDone& request );

    typedef bsc::transition<HaltDone,Halted> HT;
    typedef bsc::in_state_reaction<StopDone,Halting,&Halting::logStopDoneRequest> StopDoneIR;
    typedef boost::mpl::list<HT,StopDoneIR> reactions;

    Halting( my_context );
    virtual ~Halting();

  private:

    virtual std::string do_stateName() const;
    virtual void do_noFragmentToProcess() const;
    virtual void do_entryActionWork();
    virtual void do_exitActionWork();

    bool destructionIsDone() const;

  };

  // Running:
  class Running: public bsc::state<Running,Enabled,Processing>, public Operations
  {

  public:

    void logStopDoneRequest( const StopDone& request );
    void logHaltDoneRequest( const HaltDone& request );

    typedef bsc::transition<EndRun,Stopping> ET;
    typedef bsc::transition<EmergencyStop,Stopping> EST;
    typedef bsc::transition<Halt,Halting> HT;
    typedef bsc::in_state_reaction<StopDone,Running,&Running::logStopDoneRequest> StopDoneIR;
    typedef bsc::in_state_reaction<HaltDone,Running,&Running::logHaltDoneRequest> HaltDoneIR;
    typedef boost::mpl::list<ET,EST,HT,StopDoneIR,HaltDoneIR> reactions;

    Running( my_context );
    virtual ~Running();

  private:

    virtual std::string do_stateName() const;
    virtual void do_entryActionWork();
    virtual void do_exitActionWork();

  };

  // Processing:
  class Processing: public bsc::state<Processing,Running>, public Operations
  {

  public:

    void logQueuesEmptyRequest( const QueuesEmpty& request );
    void logEndRunRequest( const EndRun& request );

    typedef bsc::transition<Stop,DrainingQueues> DT;
    typedef bsc::in_state_reaction<QueuesEmpty,Processing,&Processing::logQueuesEmptyRequest> QueuesEmptyIR;
    typedef bsc::in_state_reaction<EndRun,Processing,&Processing::logEndRunRequest> EndRunIR;
    typedef boost::mpl::list<DT,QueuesEmptyIR,EndRunIR> reactions;

    Processing( my_context );
    virtual ~Processing();

  private:

    virtual std::string do_stateName() const;
    virtual void do_processI2OFragment( I2OChain& frag ) const;
    virtual void do_noFragmentToProcess() const;
    virtual void do_entryActionWork();
    virtual void do_exitActionWork();

  };

  // DrainingQueues:
  class DrainingQueues: public bsc::state<DrainingQueues,Running>, public Operations
  {

  public:

    void logStopRequest( const Stop& request );
    void logEndRunRequest( const EndRun& request );

    typedef bsc::transition<QueuesEmpty,FinishingDQM> FT;
    typedef bsc::in_state_reaction<Stop,DrainingQueues,&DrainingQueues::logStopRequest> StopIR;
    typedef bsc::in_state_reaction<EndRun,DrainingQueues,&DrainingQueues::logEndRunRequest> EndRunIR;
    typedef boost::mpl::list<FT,StopIR,EndRunIR> reactions;

    DrainingQueues( my_context );
    virtual ~DrainingQueues();

  private:
    virtual std::string do_stateName() const;
    virtual void do_noFragmentToProcess() const;
    virtual void do_entryActionWork();
    virtual void do_exitActionWork();

    bool allQueuesAndWorkersAreEmpty() const;
    void processStaleFragments() const;
  };

  // FinishingDQM:
  class FinishingDQM: public bsc::state<FinishingDQM,Running>, public Operations
  {

  public:

    void logStopRequest( const Stop& request );
    void logQueuesEmptyRequest( const QueuesEmpty& request );

    typedef bsc::in_state_reaction<Stop,FinishingDQM,&FinishingDQM::logStopRequest> StopIR;
    typedef bsc::in_state_reaction<QueuesEmpty,FinishingDQM,&FinishingDQM::logQueuesEmptyRequest> QueuesEmptyIR;
    typedef boost::mpl::list<StopIR,QueuesEmptyIR> reactions;

    FinishingDQM( my_context );
    virtual ~FinishingDQM();

  private:

    virtual std::string do_stateName() const;
    virtual void do_noFragmentToProcess() const;
    virtual void do_entryActionWork();
    virtual void do_exitActionWork();

    bool endOfRunProcessingIsDone() const;

  };

} // end namespace stor

#endif

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
