////////////////////////////////////////////////////////////////////////////////
//
// RBStateMachine.h
// -------
//
// Finite state machine for the Resource Broker.
//
//  Created on: Dec 9, 2011
//      									Andrei Spataru : aspataru@cern.ch
////////////////////////////////////////////////////////////////////////////////

#ifndef RBBOOSTSTATEMACHINE_H_
#define RBBOOSTSTATEMACHINE_H_

#include "EventFilter/Utilities/interface/Exception.h"
#include "EventFilter/ResourceBroker/interface/FUTypes.h"

#include "xdaq2rc/RcmsStateNotifier.h"
#include "xdata/String.h"
#include "xdata/Bag.h"
#include "xdaq/Application.h"

#include <boost/statechart/event.hpp>
#include <boost/statechart/in_state_reaction.hpp>
#include <boost/statechart/state_machine.hpp>
#include <boost/statechart/state.hpp>
#include <boost/statechart/transition.hpp>
#include <boost/mpl/list.hpp>
#include <boost/shared_ptr.hpp>

#include "toolbox/task/Action.h"
#include "toolbox/task/WorkLoop.h"
#include "toolbox/task/WorkLoopFactory.h"

#include <iostream>
#include <string>
#include <vector>
#include <semaphore.h>

namespace bsc = boost::statechart;

namespace evf {

namespace rb_statemachine {

enum States {
	HALTED,
	CONFIGURING,
	READY,
	STOPPED,
	ENABLING,
	ENABLED,
	RUNNING,
	STOPPING,
	HALTING,
	NORMAL,
	FAILED
};

class SharedResources;
typedef boost::shared_ptr<SharedResources> SharedResourcesPtr_t;

////////////////////////////////////////////////
//// Forward declarations of state classes: ////
////////////////////////////////////////////////

// Outer states:
class Failed;
class Normal;

// Inner states of Normal:
class Halted;
class Halting;
class Configuring;
class Ready;

// Inner states of Ready:
class Stopped;
class Enabled;
class Enabling;
// state hidden from RCMS
class Stopping;

// Inner states of Enabled:
// state hidden from RCMS
class Running;

////////////////////////////
//// Transition events: ////
////////////////////////////

class Configure: public bsc::event<Configure> {
};
class ConfigureDone: public bsc::event<ConfigureDone> {
};
class Enable: public bsc::event<Enable> {
};
class EnableDone: public bsc::event<EnableDone> {
};
class Stop: public bsc::event<Stop> {
};
class StopDone: public bsc::event<StopDone> {
};
class Halt: public bsc::event<Halt> {
};
class HaltDone: public bsc::event<HaltDone> {
};
class Fail: public bsc::event<Fail> {
};

//______________________________________________________________________________
/**
 * Abstract base for state classes
 *
 * $Author: smorovic $
 *
 */

class BaseState {

public:

	BaseState();
	virtual ~BaseState() = 0;
	std::string stateName() const;
	void moveToFailedState(xcept::Exception& exception) const;

	/*
	 * I2O message handling capability of states.
	 * All states with special behavior must override these functions.
	 */
	/**
	 * Base callback on process buffer received via I2O_SM_DATA_DISCARD message
	 */
	virtual bool discardDataEvent(MemRef_t* bufRef) const {
		std::cout
				<< "RBStateMachine: current state does not support operation >>discardDataEvent<<"
				<< std::endl;
		// bool value doesn't matter... return value ignored by caller: FUResourceBroker
		return false;
	}

	/**
	 * Base callback on process buffer received via I2O_SM_DQM_DISCARD message
	 */
	virtual bool discardDqmEvent(MemRef_t* bufRef) const {
		std::cout
				<< "RBStateMachine: current state does not support operation >>discardDqmEvent<<"
				<< std::endl;
		// bool value doesn't matter... return value ignored by caller: FUResourceBroker
		return false;
	}

	/*
	 * DEFAULT implementations of state-dependent actions.
	 * All states with special behavior must override these functions.
	 */

	/**
	 * State entry notifications
	 */
	virtual void do_stateNotify() = 0;

	/**
	 * Return the current state ID
	 */
	virtual int stateID() const = 0;

	/**
	 * State-dependent actions
	 */
	virtual void do_stateAction() const {
		// do nothing if state does not override this function
		/*
		 std::cout << "RBStateMachine: no >>STATE ACTION<< defined for state: "
		 << stateName() << std::endl;
		 */
	}

protected:

	virtual std::string do_stateName() const = 0;

	virtual void do_moveToFailedState(xcept::Exception& exception) const = 0;
	void fail();

	void safeEntryAction();
	virtual void do_entryActionWork() = 0;

	void safeExitAction();
	virtual void do_exitActionWork() = 0;

};

//______________________________________________________________________________
/**
 State machine class

 */

class RBStateMachine: public bsc::state_machine<RBStateMachine, Normal> {

public:

	RBStateMachine(xdaq::Application* app, SharedResourcesPtr_t sr);
	~RBStateMachine();

	/**
	 * Returns the current state of the FSM as const reference.
	 * Throws std::bad_cast if FSM is in transition (new state not constructed).
	 */
	BaseState const& getCurrentState() const throw (std::bad_cast);

	BaseState & getCurrentStateNC() const throw (std::bad_cast);

	inline SharedResourcesPtr_t getSharedResources() const {
		return sharedResources_;
	}
	inline std::string getExternallyVisibleState() {
		return visibleStateName_.value_;
	}
	inline xdata::String* getExternallyVisibleStatePtr() {
		return &visibleStateName_;
	}
	inline std::string getInternalStateName() {
		return internalStateName_;
	}
	inline xdaq::Application* getApp() const {
		return app_;
	}
	void setExternallyVisibleState(const std::string& s);
	void setInternalStateName(const std::string& s);
	/**
	 * Returns true if Halted state was never exited (visited only once).
	 */
	inline bool firstTimeInHalted() const {
		return firstTimeInHalted_;
	}
	inline void setFirstTimeInHaltedFalse() {
		firstTimeInHalted_ = false;
	}

	/**
	 * Get the RCMS StateListener parameter (classname/instance)
	 */
	xdata::Bag<xdaq2rc::ClassnameAndInstance>* rcmsStateListener();
	/**
	 * Report if RCMS StateListener was found
	 */
	xdata::Boolean* foundRcmsStateListener();
	void findRcmsStateListener(xdaq::Application* app);
	/**
	 * Notify RCMS with externally visible state
	 */
	void rcmsStateChangeNotify();

	// state-transition safety
	/**
	 * Write-locks the transition lock.
	 */
	void transitionWriteLock() {
		pthread_rwlock_wrlock(&transitionLock_);
	}
	/**
	 * Read-locks the transition lock.
	 */
	void transitionReadLock() {
		pthread_rwlock_rdlock(&transitionLock_);
	}
	/**
	 * Unlocks the transition lock.
	 */
	void transitionUnlock() {
		pthread_rwlock_unlock(&transitionLock_);
	}

private:
	void updateWebGUIExternalState(std::string newStateName) const;
	void updateWebGUIInternalState(std::string newStateName) const;

private:

	xdaq::Application* app_;
	SharedResourcesPtr_t sharedResources_;
	xdaq2rc::RcmsStateNotifier rcmsStateNotifier_;
	xdata::String visibleStateName_;
	std::string internalStateName_;
	bool firstTimeInHalted_;

	pthread_rwlock_t transitionLock_;
};

////////////////////////
//// State classes: ////
////////////////////////

//______________________________________________________________________________
/**
 Failed state

 */
class Failed: public bsc::state<Failed, RBStateMachine>, public BaseState {

public:

	Failed( my_context);
	virtual ~Failed();

	// state-dependent actions
	virtual void do_stateNotify();

	virtual int stateID() const {
		return rb_statemachine::FAILED;
	}

private:

	virtual std::string do_stateName() const;
	virtual void do_entryActionWork();
	virtual void do_exitActionWork();
	virtual void do_moveToFailedState(xcept::Exception& exception) const;

};

//______________________________________________________________________________
/**
 Normal state

 */

class Normal: public bsc::state<Normal, RBStateMachine, Halted>,
		public BaseState {

public:

	typedef bsc::transition<Fail, Failed> FT;
	typedef boost::mpl::list<FT> reactions;

	// state-dependent actions
	virtual void do_stateNotify();

	virtual int stateID() const {
		return rb_statemachine::NORMAL;
	}

	Normal( my_context);
	virtual ~Normal();

private:

	virtual std::string do_stateName() const;
	virtual void do_entryActionWork();
	virtual void do_exitActionWork();
	virtual void do_moveToFailedState(xcept::Exception& exception) const;
};

//______________________________________________________________________________
/**
 Halted state

 */

class Halted: public bsc::state<Halted, Normal>, public BaseState {

public:

	typedef bsc::transition<Configure, Configuring> RT;
	typedef boost::mpl::list<RT> reactions;

	// state-dependent actions
	virtual void do_stateNotify();

	virtual int stateID() const {
		return rb_statemachine::HALTED;
	}

	Halted( my_context);
	virtual ~Halted();

private:

	virtual std::string do_stateName() const;
	virtual void do_entryActionWork();
	virtual void do_exitActionWork();
	virtual void do_moveToFailedState(xcept::Exception& exception) const;

};

//______________________________________________________________________________
/**
 Configuring state

 */

class Configuring: public bsc::state<Configuring, Normal>, public BaseState {

public:

	typedef bsc::transition<ConfigureDone, Ready> CR;
	typedef boost::mpl::list<CR> reactions;

	Configuring( my_context);
	virtual ~Configuring();

	// state-dependent actions
	virtual void do_stateNotify();
	virtual int stateID() const {
		return rb_statemachine::CONFIGURING;
	}
	virtual void do_stateAction() const;

private:

	/**
	 * Connection to BuilderUnit bu_ and StorageManager sm_
	 */
	void connectToBUandSM() const throw (evf::Exception);

private:

	virtual std::string do_stateName() const;
	virtual void do_entryActionWork();
	virtual void do_exitActionWork();
	virtual void do_moveToFailedState(xcept::Exception& exception) const;

};

//______________________________________________________________________________
/**
 Ready state

 */

class Ready: public bsc::state<Ready, Normal, Stopped>, public BaseState {

public:

	typedef bsc::transition<Halt, Halting> HT;
	typedef boost::mpl::list<HT> reactions;

	// state-dependent actions
	virtual void do_stateNotify();

	virtual int stateID() const {
		return rb_statemachine::READY;
	}

	Ready( my_context);
	virtual ~Ready();

private:

	virtual std::string do_stateName() const;
	virtual void do_entryActionWork();
	virtual void do_exitActionWork();
	virtual void do_moveToFailedState(xcept::Exception& exception) const;

};

//______________________________________________________________________________
/**
 Stopped state

 */

class Stopped: public bsc::state<Stopped, Ready>, public BaseState {

public:

	typedef bsc::transition<Enable, Enabling> ET;
	typedef boost::mpl::list<ET> reactions;

	// state-dependent actions
	virtual void do_stateNotify();

	virtual int stateID() const {
		return rb_statemachine::STOPPED;
	}

	Stopped( my_context);
	virtual ~Stopped();

private:

	virtual std::string do_stateName() const;
	virtual void do_entryActionWork();
	virtual void do_exitActionWork();
	virtual void do_moveToFailedState(xcept::Exception& exception) const;

};

//______________________________________________________________________________
/**
 Enabling state

 */

class Enabling: public bsc::state<Enabling, Ready>,
		public BaseState,
		public toolbox::lang::Class {

public:

	typedef bsc::transition<EnableDone, Enabled> ED;
	typedef boost::mpl::list<ED> reactions;

	Enabling( my_context);
	virtual ~Enabling();

	// state-dependent actions
	virtual void do_stateNotify();
	virtual int stateID() const {
		return rb_statemachine::ENABLING;
	}
	virtual void do_stateAction() const;

private:

	virtual std::string do_stateName() const;
	virtual void do_entryActionWork();
	virtual void do_exitActionWork();
	virtual void do_moveToFailedState(xcept::Exception& exception) const;

};

//______________________________________________________________________________
/**
 Enabled state

 */

class Enabled: public bsc::state<Enabled, Ready, Running>, public BaseState {

public:

	typedef bsc::transition<Stop, Stopping> ST;
	typedef boost::mpl::list<ST> reactions;

	// state-dependent actions
	virtual void do_stateNotify();

	virtual int stateID() const {
		return rb_statemachine::ENABLED;
	}

	Enabled( my_context);
	virtual ~Enabled();

private:

	virtual std::string do_stateName() const;
	virtual void do_entryActionWork();
	virtual void do_exitActionWork();
	virtual void do_moveToFailedState(xcept::Exception& exception) const;

};

//______________________________________________________________________________
/**
 Running state

 */

class Running: public bsc::state<Running, Enabled>,
		public BaseState,
		public toolbox::lang::Class {

public:

	Running( my_context);
	virtual ~Running();

	// I2O message handling capability of state
	virtual bool discardDataEvent(MemRef_t* bufRef) const;
	virtual bool discardDqmEvent(MemRef_t* bufRef) const;

	// state-dependent actions
	virtual void do_stateNotify();
	virtual void do_stateAction() const;
	virtual int stateID() const {
		return rb_statemachine::RUNNING;
	}

private:

	virtual std::string do_stateName() const;
	virtual void do_entryActionWork();
	virtual void do_exitActionWork();
	virtual void do_moveToFailedState(xcept::Exception& exception) const;
};

//______________________________________________________________________________
/**
 Stopping state

 */

class Stopping: public bsc::state<Stopping, Ready>, public BaseState {

public:

	typedef bsc::transition<StopDone, Stopped> SD;
	typedef boost::mpl::list<SD> reactions;

	Stopping( my_context);
	virtual ~Stopping();

	//I2O capability
	virtual bool discardDataEvent(MemRef_t* bufRef) const;
	virtual bool discardDqmEvent(MemRef_t* bufRef) const;

	// state-dependent actions
	virtual void do_stateNotify();
	virtual int stateID() const {
		return rb_statemachine::STOPPING;
	}
	virtual void do_stateAction() const;

private:

	void emergencyStop() const;

private:

	virtual std::string do_stateName() const;
	virtual void do_entryActionWork();
	virtual void do_exitActionWork();
	virtual void do_moveToFailedState(xcept::Exception& exception) const;

	bool destructionIsDone() const;
};

//______________________________________________________________________________
/**
 Halting state

 */

class Halting: public bsc::state<Halting, Normal>, public BaseState {

public:

	typedef bsc::transition<HaltDone, Halted> HD;
	typedef boost::mpl::list<HD> reactions;

	Halting( my_context);
	virtual ~Halting();

	//I2O capability
	virtual bool discardDataEvent(MemRef_t* bufRef) const;
	virtual bool discardDqmEvent(MemRef_t* bufRef) const;

	// state-dependent actions
	virtual void do_stateNotify();
	virtual int stateID() const {
		return rb_statemachine::HALTING;
	}
	virtual void do_stateAction() const;

private:

	virtual std::string do_stateName() const;
	virtual void do_entryActionWork();
	virtual void do_exitActionWork();
	virtual void do_moveToFailedState(xcept::Exception& exception) const;

	bool destructionIsDone() const;

	void doAsync() const;

};

typedef boost::shared_ptr<RBStateMachine> RBStateMachinePtr;

} // end namespace rb_statemachine

} // end namespace evf

#endif /* RBBOOSTSTATEMACHINE_H_ */
