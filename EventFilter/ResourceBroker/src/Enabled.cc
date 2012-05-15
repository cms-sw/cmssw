/** \class Enabled
 *
 *  \author A. Spataru - andrei.cristian.spataru@cern.ch
 */

#include "EventFilter/ResourceBroker/interface/SharedResources.h"
#include "EventFilter/ResourceBroker/interface/RBStateMachine.h"

#include <iostream>

using std::string;
using namespace evf::rb_statemachine;

// entry action, state notification
//______________________________________________________________________________
void Enabled::do_entryActionWork() {
}

void Enabled::do_stateNotify() {
	SharedResourcesPtr_t res = outermost_context().getSharedResources();
	LOG4CPLUS_INFO(res->log_, "--> ResourceBroker: NEW STATE: " << stateName());
	outermost_context().setExternallyVisibleState(stateName());
	outermost_context().setInternalStateName(stateName());
	// notify RCMS of the new state
	outermost_context().rcmsStateChangeNotify();
}

// construction / destruction
//______________________________________________________________________________
Enabled::Enabled(my_context c) :
	my_base(c) {
	safeEntryAction();
}

Enabled::~Enabled() {
	safeExitAction();
}

// exit action, state name, move to failed state
//______________________________________________________________________________
void Enabled::do_exitActionWork() {
}

string Enabled::do_stateName() const {
	return string("Enabled");
}

void Enabled::do_moveToFailedState(xcept::Exception& exception) const {
	SharedResourcesPtr_t res = outermost_context().getSharedResources();
	res->reasonForFailed_ = exception.what();
	LOG4CPLUS_ERROR(res->log_,
			"Moving to FAILED state! Reason: " << exception.what());
	EventPtr fail(new Fail());
	res->commands_.enqEvent(fail);
}
