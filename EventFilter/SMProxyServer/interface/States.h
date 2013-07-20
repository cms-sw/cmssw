// $Id: States.h,v 1.2 2011/03/07 15:41:54 mommsen Exp $
/// @file: States.h 

#ifndef EventFilter_SMProxyServer_States_h
#define EventFilter_SMProxyServer_States_h

#include "EventFilter/SMProxyServer/interface/Exception.h"

#include "xcept/Exception.h"

#include <boost/mpl/list.hpp>
#include <boost/thread/thread.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/statechart/custom_reaction.hpp>
#include <boost/statechart/event.hpp>
#include <boost/statechart/state.hpp>
#include <boost/statechart/transition.hpp>

#include <string>


namespace smproxy
{

  ///////////////////////////////////////////
  // Forward declarations of state classes //
  ///////////////////////////////////////////

  // Outer states:
  class Failed;
  class AllOk;
  // Inner states of AllOk
  class Constructed;
  class Halted;
  class Configuring;
  class Ready;
  class Enabled;
  class Stopping;
  class Halting;
  // Inner states of Enabled
  class Starting;
  class Running;


  ////////////////////////////////
  // Internal transition events //
  ////////////////////////////////

  class ConfiguringDone: public boost::statechart::event<ConfiguringDone> {};
  class StartingDone: public boost::statechart::event<StartingDone> {};
  class StoppingDone: public boost::statechart::event<StoppingDone> {};
  class HaltingDone: public boost::statechart::event<HaltingDone> {};


  ////////////////////////////
  // Wrapper state template //
  ////////////////////////////

  template< class MostDerived,
            class Context,
            class InnerInitial = boost::mpl::list<>,
            boost::statechart::history_mode historyMode = boost::statechart::has_no_history >
  class State : public StateName,
                public boost::statechart::state<MostDerived, Context, InnerInitial, historyMode>
  {
  public:
    std::string stateName() const
    { return stateName_; }

  protected:
    typedef boost::statechart::state<MostDerived, Context, InnerInitial, historyMode> boost_state;
    typedef State my_state;
    
    State(const std::string stateName, typename boost_state::my_context& c) :
    boost_state(c), stateName_(stateName) {};
    virtual ~State() {};

    virtual void entryAction() {};
    virtual void exitAction() {};

    const std::string stateName_;

    void safeEntryAction()
    {
      std::string msg = "Failed to enter " + stateName_ + " state";
      try
      {
        entryAction();
      }
      catch( xcept::Exception& e )
      {
        XCEPT_DECLARE_NESTED(exception::StateTransition,
          sentinelException, msg, e );
        this->post_event( Fail(sentinelException) );
      }
      catch( std::exception& e )
      {
        msg += ": ";
        msg += e.what();
        XCEPT_DECLARE(exception::StateTransition,
          sentinelException, msg );
        this->post_event( Fail(sentinelException) );
      }
      catch(...)
      {
        msg += ": unknown exception";
        XCEPT_DECLARE(exception::StateTransition,
          sentinelException, msg );
        this->post_event( Fail(sentinelException) );
      }
    };

    void safeExitAction()
    {
      std::string msg = "Failed to leave " + stateName_ + " state";
      try
      {
        exitAction();
      }
      catch( xcept::Exception& e )
      {
        XCEPT_DECLARE_NESTED(exception::StateTransition,
          sentinelException, msg, e );
        this->post_event( Fail(sentinelException) );
      }
      catch( std::exception& e )
      {
        msg += ": ";
        msg += e.what();
        XCEPT_DECLARE(exception::StateTransition,
          sentinelException, msg );
        this->post_event( Fail(sentinelException) );
      }
      catch(...)
      {
        msg += ": unknown exception";
        XCEPT_DECLARE(exception::StateTransition,
          sentinelException, msg );
        this->post_event( Fail(sentinelException) );
      }
    };

  };


  ///////////////////
  // State classes //
  ///////////////////

  /**
   * Failed state
   */
  class Failed: public State<Failed,StateMachine>
  {

  public:

    typedef boost::mpl::list<
    boost::statechart::transition<Fail,Failed>
    > reactions;

    Failed(my_context c) : my_state("Failed", c)
    { safeEntryAction(); }
    virtual ~Failed()
    { safeExitAction(); }

  };

  /**
   * The default state AllOk
   */
  class AllOk: public State<AllOk,StateMachine,Constructed>
  {

  public:

    typedef boost::mpl::list<
    boost::statechart::transition<Fail,Failed,StateMachine,&StateMachine::failEvent>
    > reactions;

    AllOk(my_context c) : my_state("AllOk", c)
    { safeEntryAction(); }
    virtual ~AllOk()
    { safeExitAction(); }

  };


  /**
   * The Constructed state. Initial state of outer-state AllOk.
   */
  class Constructed: public State<Constructed,AllOk>
  {

  public:

    typedef boost::mpl::list<
    boost::statechart::transition<Configure,Configuring>
    > reactions;

    Constructed(my_context c) : my_state("Constructed", c)
    { safeEntryAction(); }
    virtual ~Constructed()
    { safeExitAction(); }

  };


  /**
   * The Halted state of outer-state AllOk.
   */
  class Halted: public State<Halted,AllOk>
  {

  public:

    typedef boost::mpl::list<
    boost::statechart::transition<Configure,Configuring>
    > reactions;

    Halted(my_context c) : my_state("Halted", c)
    { safeEntryAction(); }
    virtual ~Halted()
    { safeExitAction(); }

    virtual void entryAction()
    { outermost_context().setExternallyVisibleStateName("Halted"); }

  };


  /**
   * The Configuring state of outer-state AllOk.
   */
  class Configuring: public State<Configuring,AllOk>
  {

  public:

    typedef boost::mpl::list<
    boost::statechart::transition<ConfiguringDone,Ready>
    > reactions;

    Configuring(my_context c) : my_state("Configuring", c)
    { safeEntryAction(); }
    virtual ~Configuring()
    { safeExitAction(); }

    virtual void entryAction();
    virtual void exitAction();
    void activity();
    
  private:
    boost::scoped_ptr<boost::thread> configuringThread_;

  };


  /**
   * The Ready state of the outer-state AllOk.
   */
  class Ready: public State<Ready,AllOk>
  {

  public:

    typedef boost::mpl::list<
    boost::statechart::transition<Enable,Enabled>,
    boost::statechart::transition<Halt,Halted>
    > reactions;

    Ready(my_context c) : my_state("Ready", c)
    { safeEntryAction(); }
    virtual ~Ready()
    { safeExitAction(); }

    virtual void entryAction()
    { outermost_context().setExternallyVisibleStateName("Ready"); }

  };


  /**
   * The Enabled state of the outer-state AllOk.
   */
  class Enabled: public State<Enabled,AllOk,Starting>
  {

  public:

    typedef boost::mpl::list<
    boost::statechart::transition<Stop,Stopping>,
    boost::statechart::transition<Halt,Halting>
    > reactions;

    Enabled(my_context c) : my_state("Enabled", c)
    { safeEntryAction(); }
    virtual ~Enabled()
    { safeExitAction(); }

    // virtual void entryAction();
    // virtual void exitAction();

  };


  /**
   * The Stopping state of the outer-state AllOk.
   */
  class Stopping: public State<Stopping,AllOk>
  {

  public:
    
    typedef boost::mpl::list<
    boost::statechart::transition<StoppingDone,Ready>
    > reactions;
    
    Stopping(my_context c) : my_state("Stopping", c)
    { safeEntryAction(); }
    virtual ~Stopping()
    { safeExitAction(); }

    virtual void entryAction();
    virtual void exitAction();
    void activity();
    
  private:
    boost::scoped_ptr<boost::thread> stoppingThread_;

  };


  /**
   * The Halting state of the outer-state AllOk.
   */
  class Halting: public State<Halting,AllOk>
  {

  public:
    
    typedef boost::mpl::list<
    boost::statechart::transition<HaltingDone,Halted>
    > reactions;
    
    Halting(my_context c) : my_state("Halting", c)
    { safeEntryAction(); }
    virtual ~Halting()
    { safeExitAction(); }

    virtual void entryAction();
    virtual void exitAction();
    void activity();
    
  private:
    boost::scoped_ptr<boost::thread> haltingThread_;

  };


  /**
   * The Running state of the outer-state Enabled.
   */
  class Starting: public State<Starting,Enabled>
  {

  public:
    
    typedef boost::mpl::list<
    boost::statechart::transition<StartingDone,Running>
    > reactions;
    
    Starting(my_context c) : my_state("Starting", c)
    { safeEntryAction(); }
    virtual ~Starting()
    { safeExitAction(); }

    virtual void entryAction();
    virtual void exitAction();
    void activity();
    
  private:
    boost::scoped_ptr<boost::thread> startingThread_;

  };


  /**
   * The Running state of the outer-state Enabled.
   */
  class Running: public State<Running,Enabled>
  {

  public:
    
    Running(my_context c) : my_state("Running", c)
    { safeEntryAction(); }
    virtual ~Running()
    { safeExitAction(); }

    virtual void entryAction()
    { outermost_context().setExternallyVisibleStateName("Enabled"); }

  };

  
} //namespace smproxy

#endif //SMProxyServer_States_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
