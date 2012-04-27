/** \class Configuring
 *
 *  \author A. Spataru - andrei.cristian.spataru@cern.ch
 */

#include "EventFilter/ResourceBroker/interface/RBStateMachine.h"
#include "EvffedFillerRB.h"

#include <iostream>

using namespace evf::rb_statemachine;
using namespace evf;
using std::set;
using std::string;

// entry action, state notification, state action
//______________________________________________________________________________
void Configuring::do_entryActionWork() {
}

void Configuring::do_stateNotify() {
	SharedResourcesPtr_t res = outermost_context().getSharedResources();
	LOG4CPLUS_INFO(res->log_, "--> ResourceBroker: NEW STATE: " << stateName());
	outermost_context().setExternallyVisibleState(stateName());
	outermost_context().setInternalStateName(stateName());
	// RCMS notification no longer required here
	// this is done in FUResourceBroker in SOAP reply
	//outermost_context().rcmsStateChangeNotify();
}

void Configuring::do_stateAction() const {
	SharedResourcesPtr_t res = outermost_context().getSharedResources();

	try {
		LOG4CPLUS_INFO(res->log_, "Start configuring ...");

		connectToBUandSM();

		res->frb_ = new EvffedFillerRB(
				(FUResourceBroker*) outermost_context().getApp());
		// IPC choice & init
		res->configureResources(outermost_context().getApp());

		if (res->shmInconsistent_) {
			std::ostringstream ost;
			ost
					<< "configuring FAILED: Inconsistency in ResourceTable - nbRaw="
					<< res->nbRawCells_.value_ << " but nbResources="
					<< res->resourceStructure_->nbResources()
					<< " and nbFreeSlots="
					<< res->resourceStructure_->nbFreeSlots();
			XCEPT_RAISE(evf::Exception, ost.str());
		}

		LOG4CPLUS_INFO(res->log_, "Finished configuring!");

		EventPtr configureDone(new ConfigureDone());
		res->commands_.enqEvent(configureDone);

	} catch (xcept::Exception &e) {
		res->reasonForFailed_ = e.what();
		moveToFailedState(e);
	}
}

// construction / destruction
//______________________________________________________________________________
Configuring::Configuring(my_context c) :
	my_base(c) {
	safeEntryAction();
}

Configuring::~Configuring() {
	safeExitAction();
}

// exit action, state name, move to failed state
//______________________________________________________________________________
void Configuring::do_exitActionWork() {
}

string Configuring::do_stateName() const {
	return string("Configuring");
}

void Configuring::do_moveToFailedState(xcept::Exception& exception) const {
	SharedResourcesPtr_t res = outermost_context().getSharedResources();
	LOG4CPLUS_ERROR(res->log_,
			"Moving to FAILED state! Reason: " << exception.what());
	EventPtr fail(new Fail());
	res->commands_.enqEvent(fail);
}

// others
//______________________________________________________________________________
void Configuring::connectToBUandSM() const throw (evf::Exception) {

	SharedResourcesPtr_t res = outermost_context().getSharedResources();
	xdaq::Application* app = outermost_context().getApp();

	typedef set<xdaq::ApplicationDescriptor*> AppDescSet_t;
	typedef AppDescSet_t::iterator AppDescIter_t;

	// locate input BU
	AppDescSet_t
			setOfBUs =
					app->getApplicationContext()->getDefaultZone()-> getApplicationDescriptors(
							res->buClassName_.toString());

	if (0 != res->bu_) {
		delete res->bu_;
		res->bu_ = 0;
	}

	for (AppDescIter_t it = setOfBUs.begin(); it != setOfBUs.end(); ++it)

		if ((*it)->getInstance() == res->buInstance_)

			res->bu_ = new BUProxy(app->getApplicationDescriptor(), *it,
					app->getApplicationContext(), res->i2oPool_);

	if (0 == res->bu_) {
		string msg = res->sourceId_ + " failed to locate input BU!";
		XCEPT_RAISE(evf::Exception, msg);
	}

	// locate output SM
	AppDescSet_t
			setOfSMs =
					app->getApplicationContext()->getDefaultZone()-> getApplicationDescriptors(
							res->smClassName_.toString());

	if (0 != res->sm_) {
		delete res->sm_;
		res->sm_ = 0;
	}

	for (AppDescIter_t it = setOfSMs.begin(); it != setOfSMs.end(); ++it)
		if ((*it)->getInstance() == res->smInstance_)
			res->sm_ = new SMProxy(app->getApplicationDescriptor(), *it,
					app->getApplicationContext(), res->i2oPool_);

	if (0 == res->sm_)
		LOG4CPLUS_WARN(res->log_,
				res->sourceId_ << " failed to locate output SM!");
}
