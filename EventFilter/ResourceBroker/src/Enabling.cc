/** \class Enabling
 *
 *  \author A. Spataru - andrei.cristian.spataru@cern.ch
 */

#include "EventFilter/ResourceBroker/interface/SharedResources.h"
#include "EventFilter/ResourceBroker/interface/RBStateMachine.h"

#include <vector>

using std::string;
using std::vector;
using namespace evf::rb_statemachine;

// entry action, state notification, state action
//______________________________________________________________________________
void Enabling::do_entryActionWork() {
}

void Enabling::do_stateNotify() {
	SharedResourcesPtr_t res = outermost_context().getSharedResources();
	LOG4CPLUS_INFO(res->log_, "--> ResourceBroker: NEW STATE: " << stateName());
	outermost_context().setExternallyVisibleState(stateName());
	outermost_context().setInternalStateName(stateName());
	// RCMS notification no longer required here
	// this is done in FUResourceBroker in SOAP reply
	//outermost_context().rcmsStateChangeNotify();
}

void Enabling::do_stateAction() const {
	SharedResourcesPtr_t res = outermost_context().getSharedResources();
	IPCMethod* resourceStructure = res->resourceStructure_;

	try {
		LOG4CPLUS_INFO(res->log_, "Start enabling ...");

		// set current run number and reset GUI counters
		// UPDATED
		res->reset();
		resourceStructure->setRunNumber(res->runNumber_);
		res->lock();

		/*
		 *  UPDATE:
		 *  releaseResources
		 *  resetPendingAllocates
		 *  resetIPC
		 *
		 *  after stopping
		 */

		resourceStructure->resetCounters();
		res->unlock();

		LOG4CPLUS_INFO(res->log_, "Starting monitoring / watching workloops.");
		// starting monitoring workloop
		res->startMonitoringWorkLoop();

		// Watching Workloop is used only for Shared Memory IPC
		if (!res->useMessageQueueIPC_)
			res->startWatchingWorkLoop();

		// starting main workloops
		// 1. Discard
		res->startDiscardWorkLoop();
		// 2. Send Data
		res->startSendDataWorkLoop();
		// 3. Send DQM
		res->startSendDqmWorkLoop();

		resourceStructure->sendAllocate();

		res->nbTimeoutsWithEvent_ = 0;
		res->nbTimeoutsWithoutEvent_ = 0;
		res->dataErrorFlag_ = 0;

		// make sure access is granted to the resource structure
		// (re-enable access after an emergency stop)
		res->allowAccessToResourceStructure_ = true;

		LOG4CPLUS_INFO(res->log_, "Finished enabling!");
		EventPtr enableDone(new EnableDone());
		res->commands_.enqEvent(enableDone);

	} catch (xcept::Exception &e) {
		moveToFailedState(e);
	}
}

// construction / destruction
//______________________________________________________________________________
Enabling::Enabling(my_context c) :
	my_base(c) {
	safeEntryAction();
}

Enabling::~Enabling() {
	safeExitAction();
}

// exit action, state name, move to failed state
//______________________________________________________________________________
void Enabling::do_exitActionWork() {
}

string Enabling::do_stateName() const {
	return string("Enabling");
}

void Enabling::do_moveToFailedState(xcept::Exception& exception) const {
	SharedResourcesPtr_t res = outermost_context().getSharedResources();
	res->reasonForFailed_ = exception.what();
	LOG4CPLUS_ERROR(res->log_,
			"Moving to FAILED state! Reason: " << exception.what());
	EventPtr fail(new Fail());
	res->commands_.enqEvent(fail);
}
