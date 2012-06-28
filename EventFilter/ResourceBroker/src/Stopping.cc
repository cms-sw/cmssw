/** \class Stopping
 *
 *  \author A. Spataru - andrei.cristian.spataru@cern.ch
 */

#include "EventFilter/ResourceBroker/interface/RBStateMachine.h"
//#include "EventFilter/ResourceBroker/interface/IPCMethod.h"
#include "EventFilter/ResourceBroker/interface/SharedResources.h"

#include <iostream>
#include <vector>
#include <sstream>

using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::ostringstream;
using namespace evf::rb_statemachine;

// entry action, state notification, state action
//______________________________________________________________________________
void Stopping::do_entryActionWork() {
}

void Stopping::do_stateNotify() {
	SharedResourcesPtr_t res = outermost_context().getSharedResources();
	LOG4CPLUS_INFO(res->log_, "--> ResourceBroker: NEW STATE: " << stateName());
	outermost_context().setExternallyVisibleState(stateName());
	outermost_context().setInternalStateName(stateName());
	// RCMS notification no longer required here
	// this is done in FUResourceBroker in SOAP reply
	//outermost_context().rcmsStateChangeNotify();
}

void Stopping::do_stateAction() const {
	SharedResourcesPtr_t res = outermost_context().getSharedResources();

	try {
		LOG4CPLUS_INFO(res->log_, "Start stopping :) ...");
		res->resourceStructure_->setStopFlag(true);
		res->resourceStructure_->shutDownClients();
		timeval now;
		timeval then;
		gettimeofday(&then, 0);
		while (!res->resourceStructure_->isReadyToShutDown()) {
			::usleep(res->resourceStructureTimeout_.value_ * 10);
			gettimeofday(&now, 0);
			if ((unsigned int) (now.tv_sec - then.tv_sec)
					> res->resourceStructureTimeout_.value_ / 10000) {
				cout << "times: " << now.tv_sec << " " << then.tv_sec << " "
						<< res->resourceStructureTimeout_.value_ / 10000
						<< endl;
				LOG4CPLUS_WARN(res->log_,
						"Some Process did not detach - going to Emergency stop!");

				/**
				 * EMERGENCY STOP IS TRIGGERED
				 */
				res->lockRSAccess();
				emergencyStop();
				res->unlockRSAccess();

				break;
			}
		}

		if (res->resourceStructure_->isReadyToShutDown()) {
			// lock access to I2O discards (data & dqm)
			res->lockRSAccess();

			// if emergency stop was not triggered
			if (res->allowI2ODiscards_) {
				// any I2O discards after this point are ignored
				res->allowI2ODiscards_ = false;
				// UPDATED: release resources
				res->resourceStructure_->releaseResources();
				// UPDATED: forget pending allocates to BU
				res->resourceStructure_->resetPendingAllocates();
				// UPDATE: reset the underlying IPC method
				res->resourceStructure_->resetIPC();
			}

			res->unlockRSAccess();

			LOG4CPLUS_INFO(res->log_, "Finished stopping!");
			EventPtr stopDone(new StopDone());
			res->commands_.enqEvent(stopDone);
		}
	} catch (xcept::Exception &e) {
		moveToFailedState(e);
	}
}

/*
 * I2O capability
 */
bool Stopping::discardDataEvent(MemRef_t* bufRef) const {
	SharedResourcesPtr_t res = outermost_context().getSharedResources();
	bool returnValue = false;
	try {
		returnValue = res->resourceStructure_->discardDataEvent(bufRef);
	} catch (evf::Exception& e) {
		moveToFailedState(e);
	}
	return returnValue;
}
bool Stopping::discardDqmEvent(MemRef_t* bufRef) const {
	SharedResourcesPtr_t res = outermost_context().getSharedResources();
	bool returnValue = false;
	try {
		returnValue = res->resourceStructure_->discardDqmEvent(bufRef);
		//returnValue = res->resourceStructure_->discardDqmEventWhileHalting(bufRef);
	} catch (evf::Exception& e) {
		moveToFailedState(e);
	}
	return returnValue;
}

// construction / destruction
//______________________________________________________________________________
Stopping::Stopping(my_context c) :
	my_base(c) {
	safeEntryAction();
}

Stopping::~Stopping() {
	safeExitAction();
}

void Stopping::emergencyStop() const {
	SharedResourcesPtr_t res = outermost_context().getSharedResources();
	IPCMethod* resourceStructure = res->resourceStructure_;

	LOG4CPLUS_WARN(res->log_, "in Emergency stop - handle non-clean stops");

	// UPDATE: while in emergency stop I2O discards from SM are not allowed
	// they are re-allowed after a new enable
	res->allowI2ODiscards_ = false;
	{
		auto lk = resourceStructure->lockCrashHandlerTimed(10);
		if (lk) { 
			vector < pid_t > client_prc_ids = resourceStructure->clientPrcIds();
			for (UInt_t i = 0; i < client_prc_ids.size(); i++) {
				pid_t pid = client_prc_ids[i];
				cout << "B: killing process " << i << " pid= " << pid << endl;
				if (pid != 0) {
					//assume processes are dead by now
					if (!resourceStructure->handleCrashedEP(res->runNumber_, pid))
						res->nbTimeoutsWithoutEvent_++;
					else
						res->nbTimeoutsWithEvent_++;
				}
			}
		}
		else {
		  XCEPT_RAISE(evf::Exception, 
			"Timed out accessing the EP Crash Handler in emergency stop. SM discards not arriving?");
		}
	}
	resourceStructure->lastResort();
	::sleep(1);
	if (!resourceStructure->isReadyToShutDown()) {
		UInt_t shutdownStatus = resourceStructure->shutdownStatus();
		std::ostringstream ostr;
		ostr << "EmergencyStop: failed to shut down ResourceTable. Debug info mask:" << std::hex <<  shutdownStatus;
		res->reasonForFailed_ = ostr.str();
		XCEPT_RAISE(evf::Exception, res->reasonForFailed_);
	}

	res->printWorkLoopStatus();
	res->lock();

	LOG4CPLUS_WARN(res->log_, "Deleting the resource structure!");
	delete res->resourceStructure_;
	res->resourceStructure_ = 0;

	cout << "cycle through resourcetable config " << endl;
	res->configureResources(outermost_context().getApp());
	res->unlock();
	if (res->shmInconsistent_)
		XCEPT_RAISE(evf::Exception, "Inconsistent shm state");
	cout << "done with emergency stop" << endl;
}

// exit action, state name, move to failed state
//______________________________________________________________________________
void Stopping::do_exitActionWork() {
}

string Stopping::do_stateName() const {
	return std::string("Stopping");
}

void Stopping::do_moveToFailedState(xcept::Exception& exception) const {
	SharedResourcesPtr_t res = outermost_context().getSharedResources();
	res->reasonForFailed_ = exception.what();
	LOG4CPLUS_FATAL(res->log_,
			"Moving to FAILED state! Reason: " << exception.what());
	EventPtr fail(new Fail());
	res->commands_.enqEvent(fail);
}
