////////////////////////////////////////////////////////////////////////////////
//
// CommandQueue.h
// -------
//
//  Created on: Dec 13, 2011
//      									Andrei Spataru : aspataru@cern.ch
////////////////////////////////////////////////////////////////////////////////

#ifndef COMMANDQUEUE_H_
#define COMMANDQUEUE_H_

#include "boost/statechart/event_base.hpp"
#include <boost/shared_ptr.hpp>

#include <queue>
#include <semaphore.h>

namespace evf {

namespace rb_statemachine {

typedef boost::shared_ptr<boost::statechart::event_base> EventPtr;

/**
 * Thread-safe queue containing state machine commands
 *
 * $Author: aspataru $
 *
 */

class CommandQueue {

public:
	CommandQueue();
	~CommandQueue();

	/**
	 * Enqueues a new state machine event
	 */
	void enqEvent(EventPtr event);

	/**
	 * Waits for an event to be in the queue and returns it
	 */
	EventPtr deqEvent();

private:
	void lock();
	void unlock();

private:
	std::queue<EventPtr> eventQueue_;
	sem_t lock_;
	// counting semaphore used to reflect the state of the event queue
	sem_t commandsSem_;
};

}
}

#endif /* COMMANDQUEUE_H_ */
