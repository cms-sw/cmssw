////////////////////////////////////////////////////////////////////////////////
//
// SMEventScheduler.h
// -------
//
// Holds and executes a FIFO of FSM transition events.
//
//  Created on: Dec 13, 2011
//      									Andrei Spataru : aspataru@cern.ch
////////////////////////////////////////////////////////////////////////////////

#ifndef SMEVENTSCHEDULER_H_
#define SMEVENTSCHEDULER_H_

#include "EventFilter/ResourceBroker/interface/RBStateMachine.h"
#include "EventFilter/ResourceBroker/interface/CommandQueue.h"

#include "toolbox/task/WorkLoopFactory.h"
#include "toolbox/task/WaitingWorkLoop.h"
#include "toolbox/task/Action.h"

namespace evf {

namespace rb_statemachine {

/**
 * Holds and executes a FIFO of FSM transition events.
 *
 * $Author: aspataru $
 *
 */

class SMEventScheduler: public toolbox::lang::Class {

public:
	SMEventScheduler(RBStateMachinePtr fsm, CommandQueue& commands_);
	~SMEventScheduler();

	void startSchedulerWorkloop() throw (evf::Exception);
	bool processing(toolbox::task::WorkLoop* wl);

private:
	void stopScheduling() {
		continueWorkloop_ = false;
	}

private:
	RBStateMachinePtr fsm_;
	CommandQueue& commands_;
	// workloop / action signature for event processing
	toolbox::task::WorkLoop *wlProcessingEvents_;
	toolbox::task::ActionSignature *asProcessingEvents_;
	bool continueWorkloop_;
};

}
}

#endif /* SMEVENTSCHEDULER_H_ */
