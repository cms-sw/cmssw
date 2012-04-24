////////////////////////////////////////////////////////////////////////////////
//
// SharedResources.h
// -------
//
// Resources shared between FSM states.
//
// Created on: Sep 21, 2011
//      									Andrei Spataru : aspataru@cern.ch
////////////////////////////////////////////////////////////////////////////////

#ifndef RBSHAREDRESOURCES_H_
#define RBSHAREDRESOURCES_H_

#include "EventFilter/ResourceBroker/interface/RBStateMachine.h"
#include "EventFilter/ResourceBroker/interface/BUProxy.h"
#include "EventFilter/ResourceBroker/interface/SMProxy.h"
#include "EventFilter/ResourceBroker/interface/IPCManager.h"
#include "EventFilter/ResourceBroker/interface/FUResource.h"
#include "EventFilter/ResourceBroker/interface/CommandQueue.h"

#include "EventFilter/Utilities/interface/IndependentWebGUI.h"

#include "boost/statechart/event_base.hpp"
#include <boost/shared_ptr.hpp>

#include "xdata/InfoSpace.h"
#include "xdata/UnsignedInteger32.h"
#include "xdata/Double.h"
#include "xdata/Boolean.h"
#include "xdata/String.h"

#include "toolbox/task/Action.h"
#include "toolbox/task/WorkLoop.h"
#include "toolbox/task/WorkLoopFactory.h"

#include <semaphore.h>
#include <iostream>
#include <queue>

namespace evf {

class FUResourceBroker;

namespace rb_statemachine {

typedef boost::shared_ptr<boost::statechart::event_base> EventPtr;

// forward declaration of state classes
class Halted;
class Configuring;
class Ready;
class Stopped;
class Enabling;
class Enabled;
class Running;
class Stopping;
class Halting;
class Normal;
class Failed;

/**
 * Class holding resources shared between FSM states.
 *
 * $Author: aspataru $
 *
 */

class SharedResources: public toolbox::lang::Class {

public:
	SharedResources(Logger log);
	~SharedResources();

	// in SharedResources because it's called by Configuring and Stopping
	/**
	 * Sets-up the type of IPC to be used, and sets the resourceStructure_ pointer.
	 */
	void configureResources(xdaq::Application* app);

	// in SharedResources because it's called by Configuring and Stopping
	/**
	 * Resets counters
	 */
	void reset();
	void updateGUIExternalState(std::string newState) {
		//lock();
		gui_->updateExternalState(newState);
		//unlock();
	}
	void updateGUIInternalState(std::string newState) {
		//lock();
		gui_->updateInternalState(newState);
		//unlock();
	}
	void setFsmPointer(RBStateMachine* const fsm) {
		fsm_ = fsm;
	}

	void cancelAllWorkloops();

	/*
	 * CONCURRENT ACCESS
	 */
	void lock() {
		while (0 != sem_wait(&lock_)) {
			if (errno != EINTR) {
				LOG4CPLUS_ERROR(log_, "Cannot obtain lock on sem LOCK!");
			}
		}
	}
	void unlock() {
		sem_post(&lock_);
	}

	void printWorkLoopStatus();

private:

	/*
	 * Workloop starters and dispatch actions.
	 * Used by friend state classes.
	 */
	/**
	 * Calculate monitoring information in separate thread. Starts the workloop.
	 */
	void startMonitoringWorkLoop() throw (evf::Exception);
	/**
	 * Action for Monitoring workloop.
	 */
	bool monitoring(toolbox::task::WorkLoop* wl);
	/**
	 * Watch the state of the shm buffer in a separate thread. Start the workloop.
	 */
	void startWatchingWorkLoop() throw (evf::Exception);
	/**
	 * Action for Watching workloop.
	 */
	bool watching(toolbox::task::WorkLoop* wl);
	double deltaT(const struct timeval *start, const struct timeval *end);

	/**
	 * Work loop to send data events to Storage Manager
	 */
	void startSendDataWorkLoop() throw (evf::Exception);
	/**
	 * Action for SendData Work loop. The forwards the call to the current state of the FSM.
	 */
	bool sendData(toolbox::task::WorkLoop* wl);

	/**
	 * Work loop to send dqm events to Storage Manager
	 */
	void startSendDqmWorkLoop() throw (evf::Exception);
	/**
	 * Action for SendDqm Work loop. The forwards the call to the current state of the FSM.
	 */
	bool sendDqm(toolbox::task::WorkLoop* wl);

	/**
	 * Work loop to discard events to Builder Unit
	 */
	void startDiscardWorkLoop() throw (evf::Exception);
	/**
	 * Action for Discard Work loop. The forwards the call to the current state of the FSM.
	 */
	bool discard(toolbox::task::WorkLoop* wl);

private:

	typedef toolbox::task::WorkLoop WorkLoop_t;
	typedef toolbox::task::ActionSignature ActionSignature_t;

	/*
	 * Workloops and action signatures
	 */

	// workloop / action signature for monitoring
	WorkLoop_t *wlMonitoring_;
	ActionSignature_t *asMonitoring_;
	// workloop / action signature for watching
	WorkLoop_t *wlWatching_;
	ActionSignature_t *asWatching_;

	WorkLoop_t *wlSendData_;
	ActionSignature_t *asSendData_;
	WorkLoop_t *wlSendDqm_;
	ActionSignature_t *asSendDqm_;
	WorkLoop_t *wlDiscard_;
	ActionSignature_t *asDiscard_;

	//
	// member data
	//

	// pointer to FSM
	RBStateMachine* fsm_;

	// Hyper DAQ Independent web GUI
	IndependentWebGUI* gui_;

	// Command Queue containing state machine events
	CommandQueue commands_;

	// application logger
	Logger log_;

	// BuilderUnit (BU) to receive raw even data from
	BUProxy *bu_;

	// StorageManager (SM) to send selected events to
	SMProxy *sm_;

	// memory pool for bu <-> fu comunication messages
	toolbox::mem::Pool* i2oPool_;

	// managed resources
	IPCManager* ipcManager_;
	IPCMethod* resourceStructure_;

	// application identifier
	std::string sourceId_;

	// monitored parameters
	xdata::UnsignedInteger32 runNumber_;

	xdata::Double deltaT_;
	xdata::UnsignedInteger32 deltaN_;
	xdata::Double deltaSumOfSquares_;
	xdata::UnsignedInteger32 deltaSumOfSizes_;

	xdata::Double throughput_;
	xdata::Double rate_;
	xdata::Double average_;
	xdata::Double rms_;

	// monitored counters
	xdata::UnsignedInteger32 nbAllocatedEvents_;
	xdata::UnsignedInteger32 nbPendingRequests_;
	xdata::UnsignedInteger32 nbReceivedEvents_;
	xdata::UnsignedInteger32 nbProcessedEvents_;
	xdata::UnsignedInteger32 nbSentEvents_;
	xdata::UnsignedInteger32 nbSentDqmEvents_;
	xdata::UnsignedInteger32 nbSentErrorEvents_;
	xdata::UnsignedInteger32 nbPendingSMDiscards_;
	xdata::UnsignedInteger32 nbPendingSMDqmDiscards_;
	xdata::UnsignedInteger32 nbDiscardedEvents_;

	// UPDATED
	xdata::UnsignedInteger32 nbReceivedEol_;
	xdata::UnsignedInteger32 highestEolReceived_;
	xdata::UnsignedInteger32 nbEolPosted_;
	xdata::UnsignedInteger32 nbEolDiscarded_;

	xdata::UnsignedInteger32 nbLostEvents_;
	xdata::UnsignedInteger32 nbDataErrors_;
	xdata::UnsignedInteger32 nbCrcErrors_;
	xdata::UnsignedInteger32 nbTimeoutsWithEvent_;
	xdata::UnsignedInteger32 nbTimeoutsWithoutEvent_;
	xdata::UnsignedInteger32 dataErrorFlag_;

	// standard parameters
	xdata::Boolean segmentationMode_;
	xdata::Boolean useMessageQueueIPC_;
	xdata::UnsignedInteger32 nbClients_;
	xdata::String clientPrcIds_;
	xdata::UnsignedInteger32 nbRawCells_;
	xdata::UnsignedInteger32 nbRecoCells_;
	xdata::UnsignedInteger32 nbDqmCells_;
	xdata::UnsignedInteger32 rawCellSize_;
	xdata::UnsignedInteger32 recoCellSize_;
	xdata::UnsignedInteger32 dqmCellSize_;

	xdata::Boolean doDropEvents_;
	xdata::Boolean doFedIdCheck_;
	xdata::UnsignedInteger32 doCrcCheck_;
	xdata::UnsignedInteger32 doDumpEvents_;

	xdata::String buClassName_;
	xdata::UnsignedInteger32 buInstance_;
	xdata::String smClassName_;
	xdata::UnsignedInteger32 smInstance_;

	xdata::UnsignedInteger32 resourceStructureTimeout_;
	xdata::UnsignedInteger32 monSleepSec_;
	xdata::UnsignedInteger32 watchSleepSec_;
	xdata::UnsignedInteger32 timeOutSec_;
	xdata::Boolean processKillerEnabled_;
	xdata::Boolean useEvmBoard_;

	xdata::String reasonForFailed_;

	// debug parameters
	xdata::UnsignedInteger32 nbAllocateSent_;
	xdata::UnsignedInteger32 nbTakeReceived_;
	xdata::UnsignedInteger32 nbDataDiscardReceived_;
	xdata::UnsignedInteger32 nbDqmDiscardReceived_;

	// helper variables for monitoring
	struct timeval monStartTime_;
	UInt_t nbSentLast_;
	uint64_t sumOfSquaresLast_;
	UInt_t sumOfSizesLast_;

	// lock
	sem_t lock_;
	EvffedFillerRB *frb_;
	bool shmInconsistent_;

	/*
	 * FRIENDS
	 */
	friend class evf::FUResourceBroker;

	friend class Halted;
	friend class Configuring;
	friend class Ready;
	friend class Stopped;
	friend class Enabling;
	friend class Enabled;
	friend class Running;
	friend class Stopping;
	friend class Halting;
	friend class Normal;
	friend class Failed;
};

typedef boost::shared_ptr<SharedResources> SharedResourcesPtr_t;

} // end namespace rb_statemachine

} //end namespace evf

#endif /* RBSHAREDRESOURCES_H_ */
