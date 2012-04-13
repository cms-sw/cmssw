////////////////////////////////////////////////////////////////////////////////
//
// SMEventScheduler.h
// -------
//
// Holds and executes a list FIFO of FSM transition events.
//
//  Created on: Dec 13, 2011
//      									Andrei Spataru : aspataru@cern.ch
////////////////////////////////////////////////////////////////////////////////

#include "EventFilter/ResourceBroker/interface/SMEventScheduler.h"

#include <string>
#include <iostream>

using namespace evf::rb_statemachine;
using namespace evf;

using std::string;
using std::cout;
using std::endl;

SMEventScheduler::SMEventScheduler(RBStateMachinePtr fsm, CommandQueue& comms) :
	fsm_(fsm), commands_(comms), continueWorkloop_(true) {

	startSchedulerWorkloop();
}

SMEventScheduler::~SMEventScheduler() {
	stopScheduling();
}

//______________________________________________________________________________
void SMEventScheduler::startSchedulerWorkloop() throw (evf::Exception) {
	try {
		//improve log instead of cout
		cout << "Start 'SCHEDULER EVENT PROCESSING' workloop" << endl;
		wlProcessingEvents_ = toolbox::task::getWorkLoopFactory()->getWorkLoop(
				"Scheduler Processing Events", "waiting");
		if (!wlProcessingEvents_->isActive())
			wlProcessingEvents_->activate();
		asProcessingEvents_ = toolbox::task::bind(this,
				&SMEventScheduler::processing, "SchedulerProcessing");
		wlProcessingEvents_->submit(asProcessingEvents_);
	} catch (xcept::Exception& e) {
		string msg = "Failed to start workloop 'SCHEDULER EVENT PROCESSING'.";
		//improve log instead of cout
		cout << msg << endl;
	}
}

//______________________________________________________________________________
bool SMEventScheduler::processing(toolbox::task::WorkLoop* wl) {
	// deqEvent() blocks until a command is present in the queue
	EventPtr topEvent = commands_.deqEvent();
	string type(typeid(*topEvent).name());

	// 0. lock state transition
	fsm_->transitionWriteLock();
	// 1. process top event from the queue
	fsm_->process_event(*topEvent);
	// 1.5 unlock state transition
	fsm_->transitionUnlock();

	// 2. update state of the FSM, also notifying RCMS
	/*
	 *  XXX casting away constness for the state Notification call
	 *  done because state stateChanged in rcmsStateListener is not const
	 *  stateNotify does not change BaseState& so operation is safe
	 */
	const_cast<BaseState&> (fsm_->getCurrentState()).do_stateNotify();

	// 3. perform state-specific action
	fsm_->getCurrentState().do_stateAction();

	return continueWorkloop_;
}
