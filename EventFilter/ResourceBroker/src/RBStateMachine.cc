////////////////////////////////////////////////////////////////////////////////
//
// RBStateMachine.cc
// -------
//
// Finite state machine for the Resource Broker.
//
//  Created on: Dec 9, 2011
//      									Andrei Spataru : aspataru@cern.ch
////////////////////////////////////////////////////////////////////////////////

#include "EventFilter/ResourceBroker/interface/RBStateMachine.h"
#include "EventFilter/ResourceBroker/interface/SharedResources.h"

#include <typeinfo>
#include <fstream>

using std::cout;
using std::endl;
using std::string;
using namespace evf::rb_statemachine;

RBStateMachine::RBStateMachine(xdaq::Application* app, SharedResourcesPtr_t sr) :
			app_(app),
			sharedResources_(sr),
			rcmsStateNotifier_(app_->getApplicationLogger(),
					app_->getApplicationDescriptor(),
					app_->getApplicationContext()), visibleStateName_("N/A"),
			internalStateName_("N/A"), firstTimeInHalted_(true) {

	// pass pointer of FSM to shared resources
	sharedResources_->setFsmPointer(this);

	pthread_rwlockattr_t attr;
	pthread_rwlockattr_init(&attr);
	pthread_rwlockattr_setkind_np(&attr,
			PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP);
	pthread_rwlock_init(&transitionLock_, &attr);

}

RBStateMachine::~RBStateMachine() {
	pthread_rwlock_destroy(&transitionLock_);
	cout << "State machine DESTROYED!" << endl;
}

BaseState const& RBStateMachine::getCurrentState() const throw (std::bad_cast) {
	return state_cast<BaseState const&> ();
}

void RBStateMachine::setExternallyVisibleState(const std::string& s) {
	visibleStateName_ = s;
	updateWebGUIExternalState(visibleStateName_);
}

void RBStateMachine::setInternalStateName(const std::string& s) {
	internalStateName_ = s;
	updateWebGUIInternalState(internalStateName_);
}

// get the RCMS StateListener parameter (classname/instance)
xdata::Bag<xdaq2rc::ClassnameAndInstance>* RBStateMachine::rcmsStateListener() {
	return rcmsStateNotifier_.getRcmsStateListenerParameter();
}

// report if RCMS StateListener was found
xdata::Boolean* RBStateMachine::foundRcmsStateListener() {
	return rcmsStateNotifier_.getFoundRcmsStateListenerParameter();
}

void RBStateMachine::findRcmsStateListener(xdaq::Application* app) {
	rcmsStateNotifier_.findRcmsStateListener(); //might not be needed
	rcmsStateNotifier_.subscribeToChangesInRcmsStateListener(
			app->getApplicationInfoSpace());
}

void RBStateMachine::rcmsStateChangeNotify() {
	string state = getExternallyVisibleState();
	try {
		cout << "-->RCMS state change notify: " << state << endl;
		rcmsStateNotifier_.stateChanged(state,
				"ResourceBroker has reached target state " + state);
	} catch (...) {
		cout << "Failed to notify state change: " << state << endl;
	}
}

void RBStateMachine::updateWebGUIExternalState(string newStateName) const {
	sharedResources_->updateGUIExternalState(newStateName);
}

void RBStateMachine::updateWebGUIInternalState(string newStateName) const {
	sharedResources_->updateGUIInternalState(newStateName);
}
