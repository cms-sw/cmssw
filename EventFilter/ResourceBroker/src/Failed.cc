/** \class Failed
 *
 *  \author A. Spataru - andrei.cristian.spataru@cern.ch
 */

#include "EventFilter/ResourceBroker/interface/SharedResources.h"
#include "EventFilter/ResourceBroker/interface/RBStateMachine.h"

using std::string;
using namespace evf::rb_statemachine;

// entry action, state notification, state action
//______________________________________________________________________________
void Failed::do_entryActionWork() {
}

void Failed::do_stateNotify() {
	SharedResourcesPtr_t res = outermost_context().getSharedResources();
	LOG4CPLUS_INFO(res->log_, "--> ResourceBroker: NEW STATE: " << stateName());
	outermost_context().setExternallyVisibleState(stateName());
	outermost_context().setInternalStateName(stateName());
	// notify RCMS of the new state
	outermost_context().rcmsStateChangeNotify();
}

// construction / destruction
//______________________________________________________________________________
Failed::Failed(my_context c) :
	my_base(c) {
	safeEntryAction();
}

Failed::~Failed() {
	safeExitAction();
}

// exit action, state name, move to failed state
//______________________________________________________________________________
void Failed::do_exitActionWork() {
}

string Failed::do_stateName() const {
	return string("Failed");
}

void Failed::do_moveToFailedState(xcept::Exception& exception) const {
	// nothing to do here
}
