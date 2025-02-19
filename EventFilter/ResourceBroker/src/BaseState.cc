////////////////////////////////////////////////////////////////////////////////
//
// BaseState
// -------
//
// Superclass of all FSM states.
//
//            20/01/2012 Andrei Spataru <aspataru@cern.ch>
////////////////////////////////////////////////////////////////////////////////

#include "EventFilter/ResourceBroker/interface/RBStateMachine.h"
#include "EventFilter/ResourceBroker/interface/SharedResources.h"
#include "EventFilter/Utilities/interface/Exception.h"

#include <boost/statechart/event_base.hpp>

#include <iostream>

using std::string;
using std::cout;
using std::endl;
using namespace boost::statechart;
using namespace evf::rb_statemachine;

BaseState::BaseState() {
}

BaseState::~BaseState() {
}

string BaseState::stateName() const {
	return do_stateName();
}

void BaseState::moveToFailedState(xcept::Exception& exception) const {
	do_moveToFailedState(exception);
}

////////////////////////////////////////////////////////////////////
// Default implementation for entry / exit action virtual functions.
////////////////////////////////////////////////////////////////////

void BaseState::safeEntryAction() {
	string errmsg = "Error going into requested state!";
	try {
		do_entryActionWork();
	} catch (xcept::Exception& e) {
		XCEPT_RETHROW(evf::Exception, errmsg, e);
	}
}

void BaseState::safeExitAction() {
	string errmsg = "Error leaving current state!";
	try {
		do_exitActionWork();
	} catch (xcept::Exception& e) {
		XCEPT_RETHROW(evf::Exception, errmsg, e);
	}
}
