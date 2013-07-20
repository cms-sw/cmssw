////////////////////////////////////////////////////////////////////////////////
//
// FUResourceBroker
// ----------------
//
//            10/20/2006 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
//            20/01/2012 Andrei Spataru <aspataru@cern.ch>
////////////////////////////////////////////////////////////////////////////////

#ifndef FURESOURCEBROKER_H
#define FURESOURCEBROKER_H 1

#include "EventFilter/ResourceBroker/interface/FUTypes.h"
#include "EventFilter/ResourceBroker/interface/IPCManager.h"
#include "EventFilter/ResourceBroker/interface/RBStateMachine.h"
#include "EventFilter/ResourceBroker/interface/SharedResources.h"
#include "EventFilter/ResourceBroker/interface/SMEventScheduler.h"

#include "xdaq/Application.h"
#include "xdaq/NamespaceURI.h"
#include "xdata/InfoSpace.h"
#include "xdata/String.h"
#include "xdata/Boolean.h"
#include "xdata/UnsignedInteger32.h"
#include "xdata/Double.h"
#include "i2o/exception/Exception.h"

#include "interface/shared/frl_header.h"
#include "interface/shared/fed_header.h"
#include "interface/shared/fed_trailer.h"

#include <vector>
#include <string>
#include <semaphore.h>
#include <sys/time.h>

namespace evf {

class BUProxy;
class SMProxy;
class EvffedFillerRB;

/**
 * Main class of the Resource Broker XDAQ application.
 *
 * $Author: eulisse $
 *
 */

class FUResourceBroker: public xdaq::Application, public xdata::ActionListener {

public:
	//
	// xdaq instantiator macro
	//
	XDAQ_INSTANTIATOR();

	//
	// construction/destruction
	//
	FUResourceBroker(xdaq::ApplicationStub *s);
	virtual ~FUResourceBroker();

	//
	// public member functions
	//

	/**
	 *  FSM SOAP command callback
	 */
	xoap::MessageReference handleFSMSoapMessage(xoap::MessageReference msg)
			throw (xoap::exception::Exception);

	/**
	 * I2O TAKE callback, received from BU
	 */
	void I2O_FU_TAKE_Callback(toolbox::mem::Reference *bufRef) throw(i2o::exception::Exception);

	/**
	 * I2O DATA DISCARD callback, received from SM
	 */
	void I2O_FU_DATA_DISCARD_Callback(toolbox::mem::Reference *bufRef) throw(i2o::exception::Exception);

	/**
	 * I2O DQM DISCARD callback, received from SM
	 */
	void I2O_FU_DQM_DISCARD_Callback(toolbox::mem::Reference *bufRef) throw(i2o::exception::Exception);

	/**
	 * I2O End Of Lumisection callback, received from EVM
	 */
	void I2O_EVM_LUMISECTION_Callback(toolbox::mem::Reference *bufRef) throw(i2o::exception::Exception);

	// Hyper DAQ web page(s) [see Utilities/WebGUI]
	void webPageRequest(xgi::Input *in, xgi::Output *out)
			throw (xgi::exception::Exception);
	void customWebPage(xgi::Input *in, xgi::Output *out)
			throw (xgi::exception::Exception);

	/**
	 * xdata::ActionListener callback(s)
	 */
	void actionPerformed(xdata::Event& e);

	unsigned int instanceNumber() const {
		return instance_.value_;
	}

public:
	static const int CRC_ERROR_SHIFT = 0x0;
	static const int DATA_ERROR_SHIFT = 0x1;
	static const int LOST_ERROR_SHIFT = 0x2;
	static const int TIMEOUT_NOEVENT_ERROR_SHIFT = 0x3;
	static const int TIMEOUT_EVENT_ERROR_SHIFT = 0x4;
	static const int SENT_ERREVENT_ERROR_SHIFT = 0x5;

private:
	//
	// private member functions
	//
	void exportParameters();
	void bindStateMachineCallbacks();
	bool waitForStateChange(std::string name, int timeoutMicroSec);

private:
	//
	// member data
	//
	// Event scheduler, owned by FUResourceBroker
	rb_statemachine::SMEventScheduler* eventScheduler_;

	// Shared Resources
	rb_statemachine::SharedResourcesPtr_t res_;

	// Finite state machine
	rb_statemachine::RBStateMachinePtr fsm_;

	// monitored parameters
	xdata::String url_;
	xdata::String class_;
	xdata::UnsignedInteger32 instance_;
};

} // namespace evf


#endif
