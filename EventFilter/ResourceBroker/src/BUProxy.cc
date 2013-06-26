////////////////////////////////////////////////////////////////////////////////
//
// BUProxy
// -------
//
////////////////////////////////////////////////////////////////////////////////


#include "EventFilter/ResourceBroker/interface/BUProxy.h"

#include "toolbox/mem/Reference.h"
#include "toolbox/mem/MemoryPoolFactory.h"
#include "toolbox/mem/exception/Exception.h"

#include "i2o/Method.h"
#include "i2o/utils/AddressMap.h"

#include "xcept/tools.h"

#include "interface/evb/i2oEVBMsgs.h" 
#include "interface/shared/i2oXFunctionCodes.h"

#include <iostream>

using namespace std;
using namespace evf;

////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
BUProxy::BUProxy(xdaq::ApplicationDescriptor *fuAppDesc,
		xdaq::ApplicationDescriptor *buAppDesc,
		xdaq::ApplicationContext *fuAppContext, toolbox::mem::Pool *i2oPool) :
	fuAppDesc_(fuAppDesc), buAppDesc_(buAppDesc), fuAppContext_(fuAppContext),
			i2oPool_(i2oPool) {

}

//______________________________________________________________________________
BUProxy::~BUProxy() {

}

////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void BUProxy::sendAllocate(const UIntVec_t& fuResourceIds)
		throw (evf::Exception) {
	Logger log = fuAppContext_->getLogger();

	try {
		size_t msgSize = sizeof(I2O_BU_ALLOCATE_MESSAGE_FRAME)
				+ sizeof(BU_ALLOCATE) * (fuResourceIds.size() - 1);

		toolbox::mem::Reference *bufRef =
				toolbox::mem::getMemoryPoolFactory()->getFrame(i2oPool_,
						msgSize);

		I2O_MESSAGE_FRAME *stdMsg;
		I2O_PRIVATE_MESSAGE_FRAME *pvtMsg;
		I2O_BU_ALLOCATE_MESSAGE_FRAME *msg;

		stdMsg = (I2O_MESSAGE_FRAME*) bufRef->getDataLocation();
		pvtMsg = (I2O_PRIVATE_MESSAGE_FRAME*) stdMsg;
		msg = (I2O_BU_ALLOCATE_MESSAGE_FRAME*) stdMsg;

		stdMsg->MessageSize = msgSize >> 2;
		stdMsg->InitiatorAddress = i2o::utils::getAddressMap()->getTid(
				fuAppDesc_);
		stdMsg->TargetAddress = i2o::utils::getAddressMap()->getTid(buAppDesc_);
		stdMsg->Function = I2O_PRIVATE_MESSAGE;
		stdMsg->VersionOffset = 0;
		stdMsg->MsgFlags = 0; // Point-to-point

		pvtMsg->XFunctionCode = I2O_BU_ALLOCATE;
		pvtMsg->OrganizationID = XDAQ_ORGANIZATION_ID;

		msg->n = fuResourceIds.size();

		for (UInt_t i = 0; i < fuResourceIds.size(); i++) {
			msg->allocate[i].fuTransactionId = fuResourceIds[i];
			msg->allocate[i].fset = 1; // IGNORED!!!
		}

		bufRef->setDataSize(msgSize);

		fuAppContext_->postFrame(bufRef, fuAppDesc_, buAppDesc_);
	} catch (toolbox::mem::exception::Exception& e) {
		string errmsg = "Failed to allocate buffer reference.";
		LOG4CPLUS_ERROR(log, errmsg);
		XCEPT_RETHROW(evf::Exception, errmsg, e);
	} catch (xdaq::exception::ApplicationDescriptorNotFound& e) {
		string errmsg = "Failed to get tid.";
		LOG4CPLUS_ERROR(log, errmsg);
		XCEPT_RETHROW(evf::Exception, errmsg, e);
	} catch (xdaq::exception::Exception& e) {
		string errmsg = "Failed to post 'Allocate' message.";
		LOG4CPLUS_ERROR(log, errmsg);
		XCEPT_RETHROW(evf::Exception, errmsg, e);
	}
}

//______________________________________________________________________________
void BUProxy::sendDiscard(UInt_t buResourceId) throw (evf::Exception) {
	Logger log = fuAppContext_->getLogger();

	try {
		size_t msgSize = sizeof(I2O_BU_DISCARD_MESSAGE_FRAME);

		toolbox::mem::Reference *bufRef =
				toolbox::mem::getMemoryPoolFactory()->getFrame(i2oPool_,
						msgSize);

		I2O_MESSAGE_FRAME *stdMsg =
				(I2O_MESSAGE_FRAME*) bufRef->getDataLocation();
		I2O_PRIVATE_MESSAGE_FRAME *pvtMsg = (I2O_PRIVATE_MESSAGE_FRAME*) stdMsg;
		I2O_BU_DISCARD_MESSAGE_FRAME *msg =
				(I2O_BU_DISCARD_MESSAGE_FRAME*) stdMsg;

		stdMsg->MessageSize = msgSize >> 2;
		stdMsg->InitiatorAddress = i2o::utils::getAddressMap()->getTid(
				fuAppDesc_);
		stdMsg->TargetAddress = i2o::utils::getAddressMap()->getTid(buAppDesc_);
		stdMsg->Function = I2O_PRIVATE_MESSAGE;
		stdMsg->VersionOffset = 0;
		stdMsg->MsgFlags = 0; // Point-to-point

		pvtMsg->XFunctionCode = I2O_BU_DISCARD;
		pvtMsg->OrganizationID = XDAQ_ORGANIZATION_ID;

		msg->n = 1;
		msg->buResourceId[0] = buResourceId;

		bufRef->setDataSize(msgSize);

		fuAppContext_->postFrame(bufRef, fuAppDesc_, buAppDesc_);
	} catch (toolbox::mem::exception::Exception& e) {
		string errmsg = "Failed to allocate buffer reference.";
		LOG4CPLUS_ERROR(log, errmsg);
		XCEPT_RETHROW(evf::Exception, errmsg, e);
	} catch (xdaq::exception::ApplicationDescriptorNotFound& e) {
		string errmsg = "Failed to get tid.";
		LOG4CPLUS_ERROR(log, errmsg);
		XCEPT_RETHROW(evf::Exception, errmsg, e);
	} catch (xdaq::exception::Exception &e) {
		string errmsg = "Failed to post 'Discard' message.";
		LOG4CPLUS_ERROR(log, errmsg);
		XCEPT_RETHROW(evf::Exception, errmsg, e);
	}

}
