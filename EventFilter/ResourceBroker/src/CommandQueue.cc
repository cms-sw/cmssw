////////////////////////////////////////////////////////////////////////////////
//
// CommandQueue.cc
// -------
//
//  Created on: Dec 13, 2011
//      									Andrei Spataru : aspataru@cern.ch
////////////////////////////////////////////////////////////////////////////////

#include "EventFilter/ResourceBroker/interface/CommandQueue.h"

#include <iostream>

using namespace evf::rb_statemachine;

using std::cout;
using std::endl;

CommandQueue::CommandQueue() {
	sem_init(&lock_, 0, 1);
	// commands semaphore initialized to 0
	// no commands on the queue initially
	sem_init(&commandsSem_, 0, 0);
}

CommandQueue::~CommandQueue() {
}

void CommandQueue::enqEvent(EventPtr evType) {
	lock();
	eventQueue_.push(evType);
	sem_post(&commandsSem_);
	unlock();
}

EventPtr CommandQueue::deqEvent() {
	sem_wait(&commandsSem_);
	EventPtr eventPtr;

	lock();
	eventPtr = eventQueue_.front();
	eventQueue_.pop();
	unlock();

	return eventPtr;
}

void CommandQueue::lock() {
	while (0 != sem_wait(&lock_)) {
		cout << "SMEventScheduler: cannot obtain lock!" << endl;
		sleep(1);
	}
}

void CommandQueue::unlock() {
	sem_post(&lock_);
}
