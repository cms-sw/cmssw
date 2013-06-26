// $Id: StateMachine.h,v 1.11 2011/03/07 15:31:32 mommsen Exp $
/// @file: StateMachine.h 

#ifndef EventFilter_StorageManager_StateMachine_h
#define EventFilter_StorageManager_StateMachine_h

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
  class Constructed;
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

  /**
     Abstract base for state classes

     $Author: mommsen $
     $Revision: 1.11 $
     $Date: 2011/03/07 15:31:32 $
  */

  class Operations
  {

  public:

    Operations();
    virtual ~Operations() = 0;
    void processI2OFragment( I2OChain& frag ) const;

    void noFragmentToProcess() const;

    std::string stateName() const;

    void moveToFailedState( xcept::Exception& exception ) const;

  protected:

    virtual void do_processI2OFragment( I2OChain& frag ) const;

    virtual void do_noFragmentToProcess() const;

    virtual std::string do_stateName() const = 0;

    virtual void do_moveToFailedState( xcept::Exception& exception ) const = 0;

    void safeEntryAction();
    virtual void do_entryActionWork() = 0;

    void safeExitAction();
    virtual void do_exitActionWork() = 0; 

  };


  /**
     State machine class

     $Author: mommsen $
     $Revision: 1.11 $
     $Date: 2011/03/07 15:31:32 $
  */

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

    EventDistributor* getEventDistributor() const { return eventDistributor_; }
    FragmentStore* getFragmentStore() const { return fragmentStore_; }
    Notifier* getNotifier() { return notifier_; }
    SharedResourcesPtr getSharedResources() const { return sharedResources_; }

    void unconsumed_event( bsc::event_base const& );

    void setExternallyVisibleState( const std::string& );

  private:

    DiskWriter* diskWriter_;
    EventDistributor* eventDistributor_;
    FragmentStore* fragmentStore_;
    Notifier* notifier_;
    SharedResourcesPtr sharedResources_;

  };

  ////////////////////////
  //// State classes: ////
  ////////////////////////

  /**
     Failed state

     $Author: mommsen $
     $Revision: 1.11 $
     $Date: 2011/03/07 15:31:32 $
  */
  class Failed: public bsc::state<Failed,StateMachine>, public Operations
  {

  public:

    Failed( my_context );
    virtual ~Failed();

  private:

    virtual std::string do_stateName() const;
    virtual void do_entryActionWork();
    virtual void do_exitActionWork();
    virtual void do_moveToFailedState( xcept::Exception& exception ) const;

  };

  /**
     Normal state

     $Author: mommsen $
     $Revision: 1.11 $
     $Date: 2011/03/07 15:31:32 $
  */
  class Normal: public bsc::state<Normal,StateMachine,Constructed>, public Operations
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
    virtual void do_moveToFailedState( xcept::Exception& exception ) const;

  };

  /**
     Constructed state

     $Author: mommsen $
     $Revision: 1.11 $
     $Date: 2011/03/07 15:31:32 $
  */
  class Constructed: public bsc::state<Constructed,Normal>, public Operations
  {

  public:

    typedef bsc::transition<Configure,Ready> RT;
    typedef boost::mpl::list<RT> reactions;

    Constructed( my_context );
    virtual ~Constructed();

  private:

    virtual std::string do_stateName() const;
    virtual void do_entryActionWork();
    virtual void do_exitActionWork();
    virtual void do_moveToFailedState( xcept::Exception& exception ) const;

  };

  /**
     Halted state

     $Author: mommsen $
     $Revision: 1.11 $
     $Date: 2011/03/07 15:31:32 $
  */
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
    virtual void do_moveToFailedState( xcept::Exception& exception ) const;

  };

  /**
     Ready state

     $Author: mommsen $
     $Revision: 1.11 $
     $Date: 2011/03/07 15:31:32 $
  */
  class Ready: public bsc::state<Ready,Normal,Stopped>, public Operations
  {

  public:

    typedef bsc::transition<Reconfigure,Ready> ST;
    typedef bsc::transition<Halt,Halted> HT;
    typedef bsc::transition<HaltDone,Halted> DT;
    typedef boost::mpl::list<ST,HT,DT> reactions;

    Ready( my_context );
    virtual ~Ready();

  private:

    virtual std::string do_stateName() const;
    virtual void do_entryActionWork();
    virtual void do_exitActionWork();
    virtual void do_moveToFailedState( xcept::Exception& exception ) const;

  };

  /**
     Stopped state

     $Author: mommsen $
     $Revision: 1.11 $
     $Date: 2011/03/07 15:31:32 $
  */
  class Stopped: public bsc::state<Stopped,Ready>, public Operations
  {

  public:

    void logHaltDoneRequest( const HaltDone& request );

    typedef bsc::transition<Enable,Enabled> ET;
    typedef bsc::in_state_reaction<HaltDone,Stopped,&Stopped::logHaltDoneRequest> HaltDoneIR;
    typedef boost::mpl::list<ET,HaltDoneIR> reactions;

    Stopped( my_context );
    virtual ~Stopped();
    
  private:

    virtual std::string do_stateName() const;
    virtual void do_entryActionWork();
    virtual void do_exitActionWork();
    virtual void do_moveToFailedState( xcept::Exception& exception ) const;

  };

  /**
     Enabled state

     $Author: mommsen $
     $Revision: 1.11 $
     $Date: 2011/03/07 15:31:32 $
  */
  class Enabled: public bsc::state<Enabled,Ready,Starting>, public Operations
  {

  public:

    void logHaltRequest( const Halt& request );
    void logReconfigureRequest( const Reconfigure& request );

    typedef bsc::transition<StopDone,Stopped> DT;
    typedef bsc::in_state_reaction<Halt,Enabled,&Enabled::logHaltRequest> HaltIR;
    typedef bsc::in_state_reaction<Reconfigure,Enabled,&Enabled::logReconfigureRequest> RecfgIR;
    typedef boost::mpl::list<DT,HaltIR,RecfgIR> reactions;

    Enabled( my_context );
    virtual ~Enabled();

  private:

    virtual std::string do_stateName() const;
    virtual void do_entryActionWork();
    virtual void do_exitActionWork();
    virtual void do_moveToFailedState( xcept::Exception& exception ) const;

  };

  /**
     Starting state

     $Author: mommsen $
     $Revision: 1.11 $
     $Date: 2011/03/07 15:31:32 $
  */
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
    virtual void do_moveToFailedState( xcept::Exception& exception ) const;

    bool workerThreadsConfigured() const;

  };

  /**
     Stopping state

     $Author: mommsen $
     $Revision: 1.11 $
     $Date: 2011/03/07 15:31:32 $
  */
  class Stopping: public bsc::state<Stopping,Enabled>, public Operations
  {

  public:

    void logHaltDoneRequest( const HaltDone& request );

    typedef bsc::in_state_reaction<HaltDone,Stopping,&Stopping::logHaltDoneRequest> HaltDoneIR;
    typedef boost::mpl::list<HaltDoneIR> reactions;

    Stopping( my_context );
    virtual ~Stopping();

  private:

    virtual std::string do_stateName() const;
    virtual void do_noFragmentToProcess() const;
    virtual void do_entryActionWork();
    virtual void do_exitActionWork();
    virtual void do_moveToFailedState( xcept::Exception& exception ) const;

    bool destructionIsDone() const;

  };

  /**
     Halting state

     $Author: mommsen $
     $Revision: 1.11 $
     $Date: 2011/03/07 15:31:32 $
  */
  class Halting: public bsc::state<Halting,Enabled>, public Operations
  {

  public:

    void logStopDoneRequest( const StopDone& request );

    typedef bsc::in_state_reaction<StopDone,Halting,&Halting::logStopDoneRequest> StopDoneIR;
    typedef boost::mpl::list<StopDoneIR> reactions;

    Halting( my_context );
    virtual ~Halting();

  private:

    virtual std::string do_stateName() const;
    virtual void do_noFragmentToProcess() const;
    virtual void do_entryActionWork();
    virtual void do_exitActionWork();
    virtual void do_moveToFailedState( xcept::Exception& exception ) const;

    bool destructionIsDone() const;

  };

  /**
     Running state

     $Author: mommsen $
     $Revision: 1.11 $
     $Date: 2011/03/07 15:31:32 $
  */
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
    virtual void do_moveToFailedState( xcept::Exception& exception ) const;

  };

  /**
     Processing state

     $Author: mommsen $
     $Revision: 1.11 $
     $Date: 2011/03/07 15:31:32 $
  */
  class Processing: public bsc::state<Processing,Running>, public Operations
  {

  public:

    void logEndRunRequest( const EndRun& request );

    typedef bsc::transition<Stop,DrainingQueues> DT;
    typedef bsc::in_state_reaction<EndRun,Processing,&Processing::logEndRunRequest> EndRunIR;
    typedef boost::mpl::list<DT,EndRunIR> reactions;

    Processing( my_context );
    virtual ~Processing();

  private:

    virtual std::string do_stateName() const;
    virtual void do_processI2OFragment( I2OChain& frag ) const;
    virtual void do_noFragmentToProcess() const;
    virtual void do_entryActionWork();
    virtual void do_exitActionWork();
    virtual void do_moveToFailedState( xcept::Exception& exception ) const;

  };

  /**
     DrainingQueues state

     $Author: mommsen $
     $Revision: 1.11 $
     $Date: 2011/03/07 15:31:32 $
  */
  class DrainingQueues: public bsc::state<DrainingQueues,Running>, public Operations
  {

  public:

    void logEndRunRequest( const EndRun& request );

    typedef bsc::transition<QueuesEmpty,FinishingDQM> FT;
    typedef bsc::in_state_reaction<EndRun,DrainingQueues,&DrainingQueues::logEndRunRequest> EndRunIR;
    typedef boost::mpl::list<FT,EndRunIR> reactions;

    DrainingQueues( my_context );
    virtual ~DrainingQueues();

  private:
    virtual std::string do_stateName() const;
    virtual void do_noFragmentToProcess() const;
    virtual void do_entryActionWork();
    virtual void do_exitActionWork();
    virtual void do_moveToFailedState( xcept::Exception& exception ) const;

    bool allQueuesAndWorkersAreEmpty() const;
    void processStaleFragments() const;
  };

  /**
     FinishingDQM state

     $Author: mommsen $
     $Revision: 1.11 $
     $Date: 2011/03/07 15:31:32 $
  */
  class FinishingDQM: public bsc::state<FinishingDQM,Running>, public Operations
  {

  public:

    FinishingDQM( my_context );
    virtual ~FinishingDQM();

  private:

    virtual std::string do_stateName() const;
    virtual void do_noFragmentToProcess() const;
    virtual void do_entryActionWork();
    virtual void do_exitActionWork();
    virtual void do_moveToFailedState( xcept::Exception& exception ) const;

    bool endOfRunProcessingIsDone() const;

  };

  typedef boost::shared_ptr<StateMachine> StateMachinePtr;

} // end namespace stor

#endif // EventFilter_StorageManager_StateMachine_h

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
