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

/*
 * Supported I2O operations
 */
bool Running::take(toolbox::mem::Reference* bufRef) const {
	SharedResourcesPtr_t res = outermost_context().getSharedResources();
	bool eventComplete = res->resourceStructure_->buildResource(bufRef);
	if (eventComplete && res->doDropEvents_) {
		cout << "dropping event" << endl;
		res->resourceStructure_->dropEvent();
	}
	return true;
}
bool Running::evmLumisection(toolbox::mem::Reference* bufRef) const {
	SharedResourcesPtr_t res = outermost_context().getSharedResources();
	I2O_EVM_END_OF_LUMISECTION_MESSAGE_FRAME
			*msg =
					(I2O_EVM_END_OF_LUMISECTION_MESSAGE_FRAME *) bufRef->getDataLocation();
	if (msg->lumiSection == 0) {
		LOG4CPLUS_ERROR(res->log_, "EOL message received for ls=0!!! ");
		EventPtr fail(new Fail());
		res->commands_.enqEvent(fail);
		return false;
	}
	res->nbReceivedEol_++;
	if (res->highestEolReceived_.value_ + 100 < msg->lumiSection) {
		LOG4CPLUS_ERROR(
				res->log_,
				"EOL message not in sequence, expected "
						<< res->highestEolReceived_.value_ + 1 << " received "
						<< msg->lumiSection);
		EventPtr fail(new Fail());
		res->commands_.enqEvent(fail);
		return false;
	}
	if (res->highestEolReceived_.value_ + 1 != msg->lumiSection)
		LOG4CPLUS_WARN(
				res->log_,
				"EOL message not in sequence, expected "
						<< res->highestEolReceived_.value_ + 1 << " received "
						<< msg->lumiSection);

	if (res->highestEolReceived_.value_ < msg->lumiSection)
		res->highestEolReceived_.value_ = msg->lumiSection;

	res->resourceStructure_->postEndOfLumiSection(bufRef);
	return true;
}

bool Running::discardDataEvent(MemRef_t* bufRef) const {
	SharedResourcesPtr_t res = outermost_context().getSharedResources();
	return res->resourceStructure_->discardDataEvent(bufRef);
}
bool Running::discardDqmEvent(MemRef_t* bufRef) const {
	SharedResourcesPtr_t res = outermost_context().getSharedResources();
	return res->resourceStructure_->discardDqmEvent(bufRef);
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
	LOG4CPLUS_ERROR(res->log_,
			"Moving to FAILED state! Reason: " << exception.what());
	EventPtr fail(new Fail());
	res->commands_.enqEvent(fail);

}
