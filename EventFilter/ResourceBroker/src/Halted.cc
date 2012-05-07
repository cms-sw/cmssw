/** \class Halted
 *
 *  \author A. Spataru - andrei.cristian.spataru@cern.ch
 */

#include "EventFilter/ResourceBroker/interface/SharedResources.h"
#include "EventFilter/ResourceBroker/interface/RBStateMachine.h"

using std::string;
using namespace evf::rb_statemachine;

// entry action, state notification, state action
//______________________________________________________________________________
void Halted::do_entryActionWork() {
	if (outermost_context().firstTimeInHalted())
		// set states, not RCMS notification
		//do_stateNotify();
		outermost_context().setExternallyVisibleState(do_stateName());
	outermost_context().setInternalStateName(do_stateName());
}

void Halted::do_stateNotify() {
	SharedResourcesPtr_t res = outermost_context().getSharedResources();
	LOG4CPLUS_INFO(res->log_, "--> ResourceBroker: NEW STATE: " << stateName());
	outermost_context().setExternallyVisibleState(stateName());
	outermost_context().setInternalStateName(stateName());
	// notify RCMS of the new state
	outermost_context().rcmsStateChangeNotify();
}

// construction / destruction
//______________________________________________________________________________
Halted::Halted(my_context c) :
	my_base(c) {
	safeEntryAction();
}

Halted::~Halted() {
	safeExitAction();
}

// exit action, state name, move to failed state
//______________________________________________________________________________
void Halted::do_exitActionWork() {
	outermost_context().setFirstTimeInHaltedFalse();
}

string Halted::do_stateName() const {
	return string("Halted");
}

void Halted::do_moveToFailedState(xcept::Exception& exception) const {
	SharedResourcesPtr_t res = outermost_context().getSharedResources();
	res->reasonForFailed_ = exception.what();
	LOG4CPLUS_ERROR(res->log_,
			"Moving to FAILED state! Reason: " << exception.what());
	EventPtr fail(new Fail());
	res->commands_.enqEvent(fail);
}
