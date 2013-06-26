/** \class Ready
 *
 *  \author A. Spataru - andrei.cristian.spataru@cern.ch
 */

#include "EventFilter/ResourceBroker/interface/SharedResources.h"
#include "EventFilter/ResourceBroker/interface/RBStateMachine.h"

using std::string;
using namespace evf::rb_statemachine;

// entry action, state notification, state action
//______________________________________________________________________________
void Ready::do_entryActionWork() {
}

void Ready::do_stateNotify() {
	SharedResourcesPtr_t res = outermost_context().getSharedResources();
	LOG4CPLUS_INFO(res->log_, "--> ResourceBroker: NEW STATE: " << stateName());
	outermost_context().setExternallyVisibleState(stateName());
	outermost_context().setInternalStateName(stateName());
	// notify RCMS of the new state
	outermost_context().rcmsStateChangeNotify();
}

// construction / destruction
//______________________________________________________________________________
Ready::Ready(my_context c) :
	my_base(c) {
	safeEntryAction();
}

Ready::~Ready() {
	safeExitAction();
}

// exit action, state name, move to failed state
//______________________________________________________________________________
void Ready::do_exitActionWork() {
}

string Ready::do_stateName() const {
	return string("Ready");
}

void Ready::do_moveToFailedState(xcept::Exception& exception) const {
}
