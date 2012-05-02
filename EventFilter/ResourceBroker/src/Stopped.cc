/** \class Stopped
 *
 *  \author A. Spataru - andrei.cristian.spataru@cern.ch
 */

#include "EventFilter/ResourceBroker/interface/RBStateMachine.h"
#include "EventFilter/ResourceBroker/interface/SharedResources.h"

#include <iostream>

using std::string;
using namespace evf::rb_statemachine;

// entry action, state notification, state action
//______________________________________________________________________________
void Stopped::do_entryActionWork() {
}

void Stopped::do_stateNotify() {
	/*
	 * Stopped will set the externally visible state to ready.
	 * (useful when re-entering Stopped after a _stop_ event)
	 */
	SharedResourcesPtr_t res = outermost_context().getSharedResources();
	LOG4CPLUS_INFO(res->log_, "--> ResourceBroker: NEW STATE: " << stateName());
	outermost_context().setExternallyVisibleState("Ready");
	outermost_context().setInternalStateName(stateName());
	// notify RCMS of the new state
	outermost_context().rcmsStateChangeNotify();
}

// construction / destruction
//______________________________________________________________________________
Stopped::Stopped(my_context c) :
	my_base(c) {
	safeEntryAction();
}

Stopped::~Stopped() {
	safeExitAction();
}

// exit action, state name, move to failed state
//______________________________________________________________________________
void Stopped::do_exitActionWork() {
}

string Stopped::do_stateName() const {
	return string("Stopped");
}

void Stopped::do_moveToFailedState(xcept::Exception& exception) const {
	// handled by super-state
}
