/** \class Running
 *
 *  \author A. Spataru - andrei.cristian.spataru@cern.ch
 */

#include "EventFilter/ResourceBroker/interface/RBStateMachine.h"
#include "EventFilter/ResourceBroker/interface/SharedResources.h"

#include "interface/evb/i2oEVBMsgs.h"

#include <iostream>

using std::string;
using std::cout;
using std::endl;
using namespace evf::rb_statemachine;

// entry action, state notification, state action
//______________________________________________________________________________
void Running::do_entryActionWork() {
}

void Running::do_stateNotify() {
	SharedResourcesPtr_t res = outermost_context().getSharedResources();
	LOG4CPLUS_INFO(res->log_, "--> ResourceBroker: NEW STATE: " << stateName());
	outermost_context().setExternallyVisibleState("Enabled");
	outermost_context().setInternalStateName(stateName());
	// notify RCMS of the new state
	outermost_context().rcmsStateChangeNotify();
}

void Running::do_stateAction() const {
	outermost_context().getSharedResources()->resourceStructure_->setStopFlag(false);
}
	
/*
 * Supported I2O operations
 */

bool Running::discardDataEvent(MemRef_t* bufRef) const {
	SharedResourcesPtr_t res = outermost_context().getSharedResources();
	bool returnValue = false;
	try {
		returnValue = res->resourceStructure_->discardDataEvent(bufRef);
	} catch (evf::Exception& e) {
		moveToFailedState(e);
	}
	return returnValue;
}
bool Running::discardDqmEvent(MemRef_t* bufRef) const {
	SharedResourcesPtr_t res = outermost_context().getSharedResources();
	bool returnValue = false;
	try {
		returnValue = res->resourceStructure_->discardDqmEvent(bufRef);
	} catch (evf::Exception& e) {
		moveToFailedState(e);
	}
	return returnValue;
}

// construction / destruction
//______________________________________________________________________________
Running::Running(my_context c) :
	my_base(c) {
	safeEntryAction();
}

Running::~Running() {
	safeExitAction();
}

// exit action, state name, move to failed state
//______________________________________________________________________________
void Running::do_exitActionWork() {
}

string Running::do_stateName() const {
	return string("Running");
}

void Running::do_moveToFailedState(xcept::Exception& exception) const {
	SharedResourcesPtr_t res = outermost_context().getSharedResources();
	res->reasonForFailed_ = exception.what();
	LOG4CPLUS_FATAL(res->log_,
			"Moving to FAILED state! Reason: " << exception.what());
	EventPtr fail(new Fail());
	res->commands_.enqEvent(fail);

}
