////////////////////////////////////////////////////////////////////////////////
//
// FUResourceBroker
// ----------------
//
//            10/20/2006 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
//            20/01/2012 Andrei Spataru <aspataru@cern.ch>
////////////////////////////////////////////////////////////////////////////////

#include "EventFilter/ResourceBroker/interface/FUResourceBroker.h"
#include "EventFilter/ResourceBroker/interface/SMEventScheduler.h"

#include "EventFilter/ResourceBroker/interface/FUResource.h"
#include "EventFilter/ResourceBroker/interface/BUProxy.h"
#include "EventFilter/ResourceBroker/interface/SMProxy.h"
#include "EventFilter/ResourceBroker/interface/SoapUtils.h"

#include "i2o/Method.h"
#include "interface/shared/i2oXFunctionCodes.h"
#include "interface/evb/i2oEVBMsgs.h"
#include "xcept/tools.h"

#include "toolbox/mem/HeapAllocator.h"
#include "toolbox/mem/Reference.h"
#include "toolbox/mem/MemoryPoolFactory.h"
#include "toolbox/mem/exception/Exception.h"

#include "xoap/MessageReference.h"
#include "xoap/MessageFactory.h"
#include "xoap/SOAPEnvelope.h"
#include "xoap/SOAPBody.h"
#include "xoap/domutils.h"
#include "xoap/Method.h"

#include "cgicc/CgiDefs.h"
#include "cgicc/Cgicc.h"
#include "cgicc/FormEntry.h"
#include "cgicc/HTMLClasses.h"

#include <signal.h>
#include <iostream>
#include <sstream>

using std::string;
using std::cout;
using std::endl;
using std::stringstream;
using std::vector;
using namespace evf;
using namespace evf::rb_statemachine;

////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
FUResourceBroker::FUResourceBroker(xdaq::ApplicationStub *s) :
	xdaq::Application(s),
			res_(new SharedResources(/*this, */getApplicationLogger())) {

	bindStateMachineCallbacks();

	res_->gui_ = new IndependentWebGUI(this);
	res_->gui_->setVersionString("Changeset:   *** 05.07.2012 - V1.22 ***");

	// create state machine with shared resources
	fsm_.reset(new RBStateMachine(this, res_));

	// initialise state machine
	fsm_->initiate();

	// start the event scheduler
	eventScheduler_ = new SMEventScheduler(fsm_, res_->commands_);

	// set url, class, instance, and sourceId (=class_instance)
	url_ = getApplicationDescriptor()->getContextDescriptor()->getURL() + "/"
			+ getApplicationDescriptor()->getURN();
	class_ = getApplicationDescriptor()->getClassName();
	instance_ = getApplicationDescriptor()->getInstance();
	res_->sourceId_ = class_.toString() + "_" + instance_.toString();

	// bind i2o callbacks
	i2o::bind(this, &FUResourceBroker::I2O_FU_TAKE_Callback, I2O_FU_TAKE,
			XDAQ_ORGANIZATION_ID);
	i2o::bind(this, &FUResourceBroker::I2O_FU_DATA_DISCARD_Callback,
			I2O_FU_DATA_DISCARD, XDAQ_ORGANIZATION_ID);
	i2o::bind(this, &FUResourceBroker::I2O_FU_DQM_DISCARD_Callback,
			I2O_FU_DQM_DISCARD, XDAQ_ORGANIZATION_ID);
	i2o::bind(this, &FUResourceBroker::I2O_EVM_LUMISECTION_Callback,
			I2O_EVM_LUMISECTION, XDAQ_ORGANIZATION_ID);

	// bind HyperDAQ web pages
	xgi::bind(this, &evf::FUResourceBroker::webPageRequest, "Default");

	vector<toolbox::lang::Method*> methods = res_->gui_->getMethods();
	vector<toolbox::lang::Method*>::iterator it;
	for (it = methods.begin(); it != methods.end(); ++it) {
		if ((*it)->type() == "cgi") {
			string name = static_cast<xgi::MethodSignature*> (*it)->name();
			xgi::bind(this, &evf::FUResourceBroker::webPageRequest, name);
		}
	}
	xgi::bind(this, &evf::FUResourceBroker::customWebPage, "customWebPage");

	// allocate i2o memory pool
	string i2oPoolName = res_->sourceId_ + "_i2oPool";
	try {
		toolbox::mem::HeapAllocator *allocator =
				new toolbox::mem::HeapAllocator();
		toolbox::net::URN urn("toolbox-mem-pool", i2oPoolName);
		toolbox::mem::MemoryPoolFactory* poolFactory =
				toolbox::mem::getMemoryPoolFactory();
		res_->i2oPool_ = poolFactory->createPool(urn, allocator);
	} catch (toolbox::mem::exception::Exception& e) {
		string s = "Failed to create pool: " + i2oPoolName;
		LOG4CPLUS_FATAL(res_->log_, s);
		XCEPT_RETHROW(xcept::Exception, s, e);
	}

	// publish all parameters to app info space
	exportParameters();

	// findRcmsStateListener
	fsm_->findRcmsStateListener(this);

	// set application icon for hyperdaq
	getApplicationDescriptor()->setAttribute("icon", "/evf/images/rbicon.jpg");
	//FUResource::useEvmBoard_ = useEvmBoard_;

}

//______________________________________________________________________________
FUResourceBroker::~FUResourceBroker() {
	delete eventScheduler_;
}

////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void FUResourceBroker::bindStateMachineCallbacks() {
	xoap::bind(this, &FUResourceBroker::handleFSMSoapMessage, "Configure",
			XDAQ_NS_URI);
	xoap::bind(this, &FUResourceBroker::handleFSMSoapMessage, "Enable",
			XDAQ_NS_URI);
	xoap::bind(this, &FUResourceBroker::handleFSMSoapMessage, "Stop",
			XDAQ_NS_URI);
	xoap::bind(this, &FUResourceBroker::handleFSMSoapMessage, "Halt",
			XDAQ_NS_URI);
}

//______________________________________________________________________________
bool FUResourceBroker::waitForStateChange(string initialState,
		int timeoutMicroSec) {
	timeval start;
	timeval now;

	gettimeofday(&start, 0);

	while (fsm_->getExternallyVisibleState().compare(initialState) == 0) {
		gettimeofday(&now, 0);
		if (now.tv_usec <= start.tv_usec + timeoutMicroSec)
			::usleep(50000);
		else
			return false;
	}
	return true;
}

///////////////////////////////////////
// State Machine call back functions //
///////////////////////////////////////
//______________________________________________________________________________
xoap::MessageReference FUResourceBroker::handleFSMSoapMessage(
		xoap::MessageReference msg) throw (xoap::exception::Exception) {

	string errorMsg;
	xoap::MessageReference returnMsg;

	// register the state of the FSM before processing SOAP command
	string initialState = fsm_->getExternallyVisibleState();

	try {
		errorMsg
				= "Failed to extract FSM event and parameters from SOAP message: ";
		string command = soaputils::extractParameters(msg, this);

		errorMsg = "Failed to put a '" + command
				+ "' state machine event into command queue: ";

		if (command == "Configure") {

			EventPtr stMachEvent(new Configure());
			res_->commands_.enqEvent(stMachEvent);

		} else if (command == "Enable") {

			EventPtr stMachEvent(new Enable());
			res_->commands_.enqEvent(stMachEvent);

		} else if (command == "Stop") {

			EventPtr stMachEvent(new Stop());
			res_->commands_.enqEvent(stMachEvent);

		} else if (command == "Halt") {

			EventPtr stMachEvent(new Halt());
			res_->commands_.enqEvent(stMachEvent);
		}

		else {
			XCEPT_RAISE(
					xcept::Exception,
					"Received an unknown state machine event '" + command
							+ "'.");
			// send fail event to FSM
			EventPtr stMachEvent(new Fail());
			res_->commands_.enqEvent(stMachEvent);
		}

		errorMsg = "Failed to create FSM SOAP reply message: ";

		// wait until 'initialState' is changed
		// the SOAP response will be issued only when the state has changed
		if (waitForStateChange(initialState, 2000000)) {
			returnMsg = soaputils::createFsmSoapResponseMsg(command,
					fsm_->getExternallyVisibleState());
		} else {
			XCEPT_RAISE(xcept::Exception,
					"FAILED TO REACH TARGET STATE FROM SOAP COMMAND WITHIN TIMEOUT!");
			// send fail event to FSM
			EventPtr stMachEvent(new Fail());
			res_->commands_.enqEvent(stMachEvent);
		}

	} catch (xcept::Exception& e) {
		string s = "Exception on FSM Callback!";
		LOG4CPLUS_FATAL(res_->log_, s);
		XCEPT_RETHROW(xcept::Exception, s, e);
	}

	return returnMsg;
}

//______________________________________________________________________________
void FUResourceBroker::I2O_FU_TAKE_Callback(toolbox::mem::Reference* bufRef) throw(i2o::exception::Exception) {

	int currentStateID = -1;
	fsm_->transitionReadLock();
	currentStateID = fsm_->getCurrentState().stateID();
	fsm_->transitionUnlock();

	if (currentStateID==rb_statemachine::RUNNING) {
		try {
			bool eventComplete = res_->resourceStructure_->buildResource(bufRef);
			if (eventComplete && res_->doDropEvents_)
			{
				cout << "dropping event" << endl;
				res_->resourceStructure_->dropEvent();
			}
		}
		catch (evf::Exception& e) {
			fsm_->getCurrentState().moveToFailedState(e);
		}
	}
	else {

		stringstream details;
		details << " More details -> allocated events: "
				<< res_->nbAllocatedEvents_ << ", pending requests to BU: "
				<< res_->nbPendingRequests_ << ", received events: "
				<< res_->nbReceivedEvents_;
		LOG4CPLUS_ERROR(
				res_->log_,
				"TAKE i2o frame received in state "
						<< fsm_->getExternallyVisibleState()
						<< " is being lost! THIS MEANS LOST EVENT DATA!"
						<< details.str());

		bufRef->release();
	}
	res_->nbTakeReceived_.value_++;
}

//______________________________________________________________________________
void FUResourceBroker::I2O_EVM_LUMISECTION_Callback(
		toolbox::mem::Reference* bufRef) throw(i2o::exception::Exception){

	int currentStateID = -1;
	fsm_->transitionReadLock();
	currentStateID = fsm_->getCurrentState().stateID();
	fsm_->transitionUnlock();

	bool success = true;
	if (currentStateID==rb_statemachine::RUNNING) {

		I2O_EVM_END_OF_LUMISECTION_MESSAGE_FRAME *msg =
			(I2O_EVM_END_OF_LUMISECTION_MESSAGE_FRAME *) bufRef->getDataLocation();
		if (msg->lumiSection == 0) {
			LOG4CPLUS_ERROR(res_->log_, "EOL message received for ls=0!!! ");
			EventPtr fail(new Fail());
			res_->commands_.enqEvent(fail);
			success=false;
		}
		if (success) {
			res_->nbReceivedEol_++;
			if (res_->highestEolReceived_.value_ + 100 < msg->lumiSection) {
				LOG4CPLUS_ERROR( res_->log_, "EOL message not in sequence, expected "
					<< res_->highestEolReceived_.value_ + 1 << " received "
					<< msg->lumiSection);

				EventPtr fail(new Fail());
				res_->commands_.enqEvent(fail);
				success=false;
			}
		}
		if (success) {
			if (res_->highestEolReceived_.value_ + 1 != msg->lumiSection)
				LOG4CPLUS_WARN( res_->log_, "EOL message not in sequence, expected "
						<< res_->highestEolReceived_.value_ + 1 << " received "
						<< msg->lumiSection);

			if (res_->highestEolReceived_.value_ < msg->lumiSection)
				res_->highestEolReceived_.value_ = msg->lumiSection;

			try {
				res_->resourceStructure_->postEndOfLumiSection(bufRef);
			} catch (evf::Exception& e) {
				fsm_->getCurrentState().moveToFailedState(e);
			}
		}
	}
	else success=false;

	if (!success)  LOG4CPLUS_ERROR(res_->log_,"EOL i2o frame received in state "
				<< fsm_->getExternallyVisibleState() << " is being lost");
	bufRef->release();
}

//______________________________________________________________________________
void FUResourceBroker::I2O_FU_DATA_DISCARD_Callback(
		toolbox::mem::Reference* bufRef) throw(i2o::exception::Exception){

	// obtain lock on Resource Structure for discard
	res_->lockRSAccess();

	fsm_->transitionReadLock();
	const BaseState& currentState = fsm_->getCurrentState();

	if (res_->allowI2ODiscards_)
		/*bool success = */
		currentState.discardDataEvent(bufRef);
	else {
		LOG4CPLUS_WARN(
				res_->log_,
				"Data Discard I2O message received from SM is being ignored! ShmBuffer was reinitialized!");
		bufRef->release();
	}
	fsm_->transitionUnlock();
	res_->unlockRSAccess();

	res_->nbDataDiscardReceived_.value_++;
}

//______________________________________________________________________________
void FUResourceBroker::I2O_FU_DQM_DISCARD_Callback(
		toolbox::mem::Reference* bufRef) throw(i2o::exception::Exception){

	// obtain lock on Resource Structure for discard
	res_->lockRSAccess();

	fsm_->transitionReadLock();
	const BaseState& currentState = fsm_->getCurrentState();

	if (res_->allowI2ODiscards_)
		/*bool success = */
		currentState.discardDqmEvent(bufRef);
	else {
		LOG4CPLUS_WARN(
				res_->log_,
				"DQM Discard I2O message received from SM is being ignored! ShmBuffer was reinitialized!");
		bufRef->release();
	}
	fsm_->transitionUnlock();
	res_->unlockRSAccess();

	res_->nbDqmDiscardReceived_.value_++;
}

//______________________________________________________________________________
void FUResourceBroker::webPageRequest(xgi::Input *in, xgi::Output *out)
		throw (xgi::exception::Exception) {
	string name = in->getenv("PATH_INFO");
	if (name.empty())
		name = "defaultWebPage";
	static_cast<xgi::MethodSignature*> (res_->gui_->getMethod(name))->invoke(
			in, out);
}

//______________________________________________________________________________
void FUResourceBroker::actionPerformed(xdata::Event& e) {
	res_->lock();

	if (0 != res_->resourceStructure_) {

		//gui_->monInfoSpace()->lock();

		if (e.type() == "urn:xdata-event:ItemGroupRetrieveEvent") {
			res_->nbClients_ = res_->resourceStructure_->nbClients();
			res_->clientPrcIds_
					= res_->resourceStructure_->clientPrcIdsAsString();
			res_->nbAllocatedEvents_ = res_->resourceStructure_->nbAllocated();
			res_->nbPendingRequests_ = res_->resourceStructure_->nbPending();
			res_->nbReceivedEvents_ = res_->resourceStructure_->nbCompleted();
			res_->nbSentEvents_ = res_->resourceStructure_->nbSent();
			res_->nbSentDqmEvents_ = res_->resourceStructure_->nbSentDqm();
			res_->nbSentErrorEvents_ = res_->resourceStructure_->nbSentError();

			int nbPSMD = res_->resourceStructure_->nbPendingSMDiscards();
			if (nbPSMD>=0) res_->nbPendingSMDiscards_=(unsigned int)nbPSMD;
			else res_->nbPendingSMDiscards_=0;

			res_->nbPendingSMDqmDiscards_
					= res_->resourceStructure_->nbPendingSMDqmDiscards();
			res_->nbDiscardedEvents_ = res_->resourceStructure_->nbDiscarded();
			res_->nbLostEvents_ = res_->resourceStructure_->nbLost();
			// UPDATED
			res_->nbEolPosted_ = res_->resourceStructure_->nbEolPosted();
			res_->nbEolDiscarded_ = res_->resourceStructure_->nbEolDiscarded();
			res_->nbDataErrors_ = res_->resourceStructure_->nbErrors();
			res_->nbCrcErrors_ = res_->resourceStructure_->nbCrcErrors();
			res_->nbAllocateSent_ = res_->resourceStructure_->nbAllocSent();
			res_->dataErrorFlag_.value_ = (res_->nbCrcErrors_.value_ != 0u
					+ ((res_->nbDataErrors_.value_ != 0u) << 1)
					+ ((res_->nbLostEvents_.value_ != 0u) << 2)
					+ ((res_->nbTimeoutsWithEvent_.value_ != 0u) << 3)
					+ ((res_->nbTimeoutsWithoutEvent_.value_ != 0u) << 4)
					+ ((res_->nbSentErrorEvents_.value_ != 0u) << 5));

		} else if (e.type() == "ItemChangedEvent") {

			string item = dynamic_cast<xdata::ItemChangedEvent&> (e).itemName();

			if (item == "doFedIdCheck")
				FUResource::doFedIdCheck(res_->doFedIdCheck_);
			if (item == "useEvmBoard")
				FUResource::useEvmBoard(res_->useEvmBoard_);
			if (item == "doCrcCheck")
				res_->resourceStructure_->setDoCrcCheck(res_->doCrcCheck_);
			if (item == "doDumpEvents")
				res_->resourceStructure_->setDoDumpEvents(res_->doDumpEvents_);
		}

		//gui_->monInfoSpace()->unlock();
	} else {
		res_->nbClients_ = 0;
		res_->clientPrcIds_ = "";
		res_->nbAllocatedEvents_ = 0;
		res_->nbPendingRequests_ = 0;
		res_->nbReceivedEvents_ = 0;
		res_->nbSentEvents_ = 0;
		res_->nbSentDqmEvents_ = 0;
		res_->nbSentErrorEvents_ = 0;
		res_->nbPendingSMDiscards_ = 0;
		res_->nbPendingSMDqmDiscards_ = 0;
		res_->nbDiscardedEvents_ = 0;
		res_->nbLostEvents_ = 0;
		res_->nbDataErrors_ = 0;
		res_->nbCrcErrors_ = 0;
		res_->nbAllocateSent_ = 0;
	}
	res_->unlock();
}

//______________________________________________________________________________
void FUResourceBroker::exportParameters() {
	assert(0 != res_->gui_);

	res_->gui_->addMonitorParam("url", &url_);
	res_->gui_->addMonitorParam("class", &class_);
	res_->gui_->addMonitorParam("instance", &instance_);
	res_->gui_->addMonitorParam("runNumber", &res_->runNumber_);
	res_->gui_->addMonitorParam("stateName",
			fsm_->getExternallyVisibleStatePtr());

	res_->gui_->addMonitorParam("deltaT", &res_->deltaT_);
	res_->gui_->addMonitorParam("deltaN", &res_->deltaN_);
	res_->gui_->addMonitorParam("deltaSumOfSquares", &res_->deltaSumOfSquares_);
	res_->gui_->addMonitorParam("deltaSumOfSizes", &res_->deltaSumOfSizes_);

	res_->gui_->addMonitorParam("throughput", &res_->throughput_);
	res_->gui_->addMonitorParam("rate", &res_->rate_);
	res_->gui_->addMonitorParam("average", &res_->average_);
	res_->gui_->addMonitorParam("rms", &res_->rms_);
	res_->gui_->addMonitorParam("dataErrorFlag", &res_->dataErrorFlag_);

	res_->gui_->addMonitorCounter("nbAllocatedEvents",
			&res_->nbAllocatedEvents_);
	res_->gui_->addMonitorCounter("nbPendingRequests",
			&res_->nbPendingRequests_);
	res_->gui_->addMonitorCounter("nbReceivedEvents", &res_->nbReceivedEvents_);
	res_->gui_->addMonitorCounter("nbSentEvents", &res_->nbSentEvents_);
	res_->gui_->addMonitorCounter("nbSentErrorEvents",
			&res_->nbSentErrorEvents_);
	res_->gui_->addMonitorCounter("nbDiscardedEvents",
			&res_->nbDiscardedEvents_);
	// UPDATED
	res_->gui_->addMonitorCounter("nbReceivedEol", &res_->nbReceivedEol_);
	res_->gui_->addMonitorCounter("highestEolReceived",
			&res_->highestEolReceived_);
	res_->gui_->addMonitorCounter("nbEolPosted", &res_->nbEolPosted_);
	res_->gui_->addMonitorCounter("nbEolDiscarded", &res_->nbEolDiscarded_);

	res_->gui_->addMonitorCounter("nbPendingSMDiscards",
			&res_->nbPendingSMDiscards_);

	res_->gui_->addMonitorCounter("nbSentDqmEvents", &res_->nbSentDqmEvents_);
	res_->gui_->addMonitorCounter("nbDqmDiscardReceived",
			&res_->nbDqmDiscardReceived_);
	res_->gui_->addMonitorCounter("nbPendingSMDqmDiscards",
			&res_->nbPendingSMDqmDiscards_);

	res_->gui_->addMonitorCounter("nbLostEvents", &res_->nbLostEvents_);
	res_->gui_->addMonitorCounter("nbDataErrors", &res_->nbDataErrors_);
	res_->gui_->addMonitorCounter("nbCrcErrors", &res_->nbCrcErrors_);
	res_->gui_->addMonitorCounter("nbTimeoutsWithEvent",
			&res_->nbTimeoutsWithEvent_);
	res_->gui_->addMonitorCounter("nbTimeoutsWithoutEvent",
			&res_->nbTimeoutsWithoutEvent_);

	res_->gui_->addStandardParam("segmentationMode", &res_->segmentationMode_);
	res_->gui_->addStandardParam("useMessageQueueIPC",
			&res_->useMessageQueueIPC_);
	res_->gui_->addStandardParam("nbClients", &res_->nbClients_);
	res_->gui_->addStandardParam("clientPrcIds", &res_->clientPrcIds_);
	res_->gui_->addStandardParam("nbRawCells", &res_->nbRawCells_);
	res_->gui_->addStandardParam("nbRecoCells", &res_->nbRecoCells_);
	res_->gui_->addStandardParam("nbDqmCells", &res_->nbDqmCells_);
	res_->gui_->addStandardParam("rawCellSize", &res_->rawCellSize_);
	res_->gui_->addStandardParam("recoCellSize", &res_->recoCellSize_);
	res_->gui_->addStandardParam("dqmCellSize", &res_->dqmCellSize_);
	res_->gui_->addStandardParam("nbFreeResRequiredForAllocate",
			&res_->freeResRequiredForAllocate_);

	res_->gui_->addStandardParam("doDropEvents", &res_->doDropEvents_);
	res_->gui_->addStandardParam("doFedIdCheck", &res_->doFedIdCheck_);
	res_->gui_->addStandardParam("doCrcCheck", &res_->doCrcCheck_);
	res_->gui_->addStandardParam("doDumpEvents", &res_->doDumpEvents_);
	res_->gui_->addStandardParam("buClassName", &res_->buClassName_);
	res_->gui_->addStandardParam("buInstance", &res_->buInstance_);
	res_->gui_->addStandardParam("smClassName", &res_->smClassName_);
	res_->gui_->addStandardParam("smInstance", &res_->smInstance_);
	res_->gui_->addStandardParam("resourceStructureTimeout_",
			&res_->resourceStructureTimeout_);
	res_->gui_->addStandardParam("monSleepSec", &res_->monSleepSec_);
	res_->gui_->addStandardParam("watchSleepSec", &res_->watchSleepSec_);
	res_->gui_->addStandardParam("timeOutSec", &res_->timeOutSec_);
	res_->gui_->addStandardParam("processKillerEnabled",
			&res_->processKillerEnabled_);
	res_->gui_->addStandardParam("useEvmBoard", &res_->useEvmBoard_);
	res_->gui_->addStandardParam("rcmsStateListener", fsm_->rcmsStateListener());

	res_->gui_->addStandardParam("foundRcmsStateListener",
			fsm_->foundRcmsStateListener());

	res_->gui_->addStandardParam("reasonForFailed", &res_->reasonForFailed_);

	res_->gui_->addDebugCounter("nbAllocateSent", &res_->nbAllocateSent_);
	res_->gui_->addDebugCounter("nbTakeReceived", &res_->nbTakeReceived_);
	res_->gui_->addDebugCounter("nbDataDiscardReceived",
			&res_->nbDataDiscardReceived_);

	res_->gui_->exportParameters();

	res_->gui_->addItemChangedListener("doFedIdCheck", this);
	res_->gui_->addItemChangedListener("useEvmBoard", this);
	res_->gui_->addItemChangedListener("doCrcCheck", this);
	res_->gui_->addItemChangedListener("doDumpEvents", this);

}

//______________________________________________________________________________
void FUResourceBroker::customWebPage(xgi::Input*in, xgi::Output*out)
		throw (xgi::exception::Exception) {
	using namespace cgicc;
	Cgicc cgi(in);
	std::vector < FormEntry > els = cgi.getElements();
	for (std::vector<FormEntry>::iterator it = els.begin(); it != els.end(); it++)
		cout << "form entry " << (*it).getValue() << endl;

	std::vector < FormEntry > el1;
	cgi.getElement("crcError", el1);
	*out << "<html>" << endl;
	res_->gui_->htmlHead(in, out, res_->sourceId_);
	*out << "<body>" << endl;
	res_->gui_->htmlHeadline(in, out);

	res_->lock();

	if (0 != res_->resourceStructure_) {
		if (el1.size() != 0) {
			res_->resourceStructure_->injectCRCError();
		}
		*out << "<form method=\"GET\" action=\"customWebPage\" >";
		*out
				<< "<button name=\"crcError\" type=\"submit\" value=\"injCRC\">Inject CRC</button>"
				<< endl;
		*out << "</form>" << endl;
		*out << "<hr/>" << endl;
		vector < pid_t > client_prc_ids
				= res_->resourceStructure_->clientPrcIds();
		*out << table().set("frame", "void").set("rules", "rows") .set("class",
				"modules").set("width", "250") << endl << tr() << th(
				"Client Processes").set("colspan", "3") << tr() << endl << tr()
				<< th("client").set("align", "left") << th("process id").set(
				"align", "center") << th("status").set("align", "center")
				<< tr() << endl;
		for (UInt_t i = 0; i < client_prc_ids.size(); i++) {

			pid_t pid = client_prc_ids[i];
			int status = kill(pid, 0);

			stringstream ssi;
			ssi << i + 1;
			stringstream sspid;
			sspid << pid;
			stringstream ssstatus;
			ssstatus << status;

			string bg_status = (status == 0) ? "#00ff00" : "ff0000";
			*out << tr() << td(ssi.str()).set("align", "left") << td(
					sspid.str()).set("align", "center")
					<< td(ssstatus.str()).set("align", "center").set("bgcolor",
							bg_status) << tr() << endl;
		}
		*out << table() << endl;
		*out << "<br><br>" << endl;

		vector < string > states = res_->resourceStructure_->cellStates();
		vector < UInt_t > evt_numbers
				= res_->resourceStructure_->cellEvtNumbers();
		vector < pid_t > prc_ids = res_->resourceStructure_->cellPrcIds();
		vector < time_t > time_stamps
				= res_->resourceStructure_->cellTimeStamps();

		*out << table().set("frame", "void").set("rules", "rows") .set("class",
				"modules").set("width", "500") << endl << tr() << th(
				"Shared Memory Cells").set("colspan", "6") << tr() << endl
				<< tr() << th("cell").set("align", "left") << th("state").set(
				"align", "center") << th("event").set("align", "center") << th(
				"process id").set("align", "center") << th("timestamp").set(
				"align", "center") << th("time").set("align", "center") << tr()
				<< endl;
		for (UInt_t i = 0; i < states.size(); i++) {
			string state = states[i];
			UInt_t evt = evt_numbers[i];
			pid_t pid = prc_ids[i];
			time_t tstamp = time_stamps[i];
			double tdiff = difftime(time(0), tstamp);

			stringstream ssi;
			ssi << i;
			stringstream ssevt;
			if (evt != 0xffffffff)
				ssevt << evt;
			else
				ssevt << " - ";
			stringstream sspid;
			if (pid != 0)
				sspid << pid;
			else
				sspid << " - ";
			stringstream sststamp;
			if (tstamp != 0)
				sststamp << tstamp;
			else
				sststamp << " - ";
			stringstream sstdiff;
			if (tstamp != 0)
				sstdiff << tdiff;
			else
				sstdiff << " - ";

			string bg_state = "#ffffff";
			if (state == "RAWWRITING" || state == "RAWWRITTEN" || state
					== "RAWREADING" || state == "RAWREAD")
				bg_state = "#99CCff";
			else if (state == "PROCESSING")
				bg_state = "#ff0000";
			else if (state == "PROCESSED" || state == "RECOWRITING" || state
					== "RECOWRITTEN")
				bg_state = "#CCff99";
			else if (state == "SENDING")
				bg_state = "#00FF33";
			else if (state == "SENT")
				bg_state = "#006633";
			else if (state == "DISCARDING")
				bg_state = "#FFFF00";
			else if (state == "LUMISECTION")
				bg_state = "#0000FF";

			*out << tr() << td(ssi.str()).set("align", "left")
					<< td(state).set("align", "center").set("bgcolor", bg_state)
					<< td(ssevt.str()).set("align", "center")
					<< td(sspid.str()).set("align", "center") << td(
					sststamp.str()).set("align", "center")
					<< td(sstdiff.str()).set("align", "center") << tr() << endl;
		}
		*out << table() << endl;
		*out << "<br><br>" << endl;

		vector < string > dqmstates = res_->resourceStructure_->dqmCellStates();

		*out << table().set("frame", "void").set("rules", "rows") .set("class",
				"modules").set("width", "500") << endl << tr() << th(
				"Shared Memory DQM Cells").set("colspan", "6") << tr() << endl
				<< tr() << th("cell").set("align", "left") << th("state").set(
				"align", "center") << tr() << endl;
		for (UInt_t i = 0; i < dqmstates.size(); i++) {
			string state = dqmstates[i];

			string bg_state = "#ffffff";
			if (state == "WRITING" || state == "WRITTEN")
				bg_state = "#99CCff";
			else if (state == "SENDING")
				bg_state = "#00FF33";
			else if (state == "SENT")
				bg_state = "#006633";
			else if (state == "DISCARDING")
				bg_state = "#FFFF00";

			*out << tr() << "<td>" << i << "</td>" << td(state).set("align",
					"center").set("bgcolor", bg_state) << tr() << endl;
		}
		*out << table() << endl;

	}
	*out << "</body>" << endl << "</html>" << endl;

	res_->unlock();
}

////////////////////////////////////////////////////////////////////////////////
// XDAQ instantiator implementation macro
////////////////////////////////////////////////////////////////////////////////

XDAQ_INSTANTIATOR_IMPL(FUResourceBroker)
