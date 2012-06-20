/** \class Normal
 *
 *  \author A. Spataru - andrei.cristian.spataru@cern.ch
 */

#include "EventFilter/ResourceBroker/interface/RBStateMachine.h"

using std::string;
using namespace evf::rb_statemachine;

// entry action, state notification, state action
//______________________________________________________________________________
void Normal::do_entryActionWork() {
}

void Normal::do_stateNotify() {
}

// construction / destruction
//______________________________________________________________________________
Normal::Normal(my_context c) :
	my_base(c) {
	safeEntryAction();
}

Normal::~Normal() {
	safeExitAction();
}

// exit action, state name, move to failed state
//______________________________________________________________________________
void Normal::do_exitActionWork() {
}

string Normal::do_stateName() const {
	return string("Normal");
}

void Normal::do_moveToFailedState(xcept::Exception& exception) const {
}
