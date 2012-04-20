/** \class Halting
 *
 *  \author A. Spataru - andrei.cristian.spataru@cern.ch
 */

#include "EventFilter/ResourceBroker/interface/RBStateMachine.h"
#include "EvffedFillerRB.h"

#include <iostream>

using std::string;
using std::cout;
using std::endl;
using namespace evf::rb_statemachine;

// entry action, state notification, state action
//______________________________________________________________________________
void Halting::do_entryActionWork() {
}

void Halting::do_stateNotify() {
	SharedResourcesPtr_t res = outermost_context().getSharedResources();
	LOG4CPLUS_INFO(res->log_, "--> ResourceBroker: NEW STATE: " << stateName());
	outermost_context().setExternallyVisibleState(stateName());
	outermost_context().setInternalStateName(stateName());
	// RCMS notification no longer required here
	// this is done in FUResourceBroker in SOAP reply
	//outermost_context().rcmsStateChangeNotify();
}

void Halting::do_stateAction() const {
	SharedResourcesPtr_t res = outermost_context().getSharedResources();
	IPCMethod* resourceStructure = res->resourceStructure_;
	try {
		LOG4CPLUS_INFO(res->log_, "Start halting ...");
		if (resourceStructure->isActive()) {
			resourceStructure->shutDownClients();
			UInt_t count = 0;
			while (count < 10) {
				if (resourceStructure->isReadyToShutDown()) {
					res->lock();

					delete res->resourceStructure_;
					res->resourceStructure_ = 0;

					res->unlock();
					LOG4CPLUS_INFO(
							res->log_,
							count + 1
									<< ". try to destroy resource table succeeded!");
					break;
				} else {
					count++;

					LOG4CPLUS_DEBUG(
							res->log_,
							count
									<< ". try to destroy resource table failed ...");

					::sleep(1);
				}
			}
		} else {
			res->lock();

			delete res->resourceStructure_;
			res->resourceStructure_ = 0;

			res->unlock();
		}

		if (0 == res->resourceStructure_) {
			LOG4CPLUS_INFO(res->log_, "Finished halting!");
			EventPtr haltDone(new HaltDone());
			res->commands_.enqEvent(haltDone);
		} else {
			res->reasonForFailed_
					= "halting FAILED: ResourceTable shutdown timed out.";
			EventPtr failTimeOut(new Fail());
			res->commands_.enqEvent(failTimeOut);
		}
	} catch (xcept::Exception &e) {

		res->reasonForFailed_ = e.what();
		moveToFailedState(e);
	}

	if (res->frb_)
		delete res->frb_;
}

/*
 * I2O capability
 */
bool Halting::discardDataEvent(MemRef_t* bufRef) const {
	SharedResourcesPtr_t res = outermost_context().getSharedResources();
	return res->resourceStructure_->discardDataEventWhileHalting(bufRef);
}
bool Halting::discardDqmEvent(MemRef_t* bufRef) const {
	SharedResourcesPtr_t res = outermost_context().getSharedResources();
	return res->resourceStructure_->discardDqmEvent/*WhileHalting*/(bufRef);
}

// construction / destruction
//______________________________________________________________________________
Halting::Halting(my_context c) :
	my_base(c) {
	safeEntryAction();
}

Halting::~Halting() {
	safeExitAction();
}

// exit action, state name, move to failed state
//______________________________________________________________________________
void Halting::do_exitActionWork() {
}

string Halting::do_stateName() const {
	return string("Halting");
}

void Halting::do_moveToFailedState(xcept::Exception& exception) const {
	SharedResourcesPtr_t res = outermost_context().getSharedResources();
	LOG4CPLUS_ERROR(res->log_,
			"Moving to FAILED state! Reason: " << exception.what());
	EventPtr fail(new Fail());
	res->commands_.enqEvent(fail);
}
