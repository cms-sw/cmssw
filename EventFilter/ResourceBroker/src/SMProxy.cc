////////////////////////////////////////////////////////////////////////////////
//
// SMProxy
// -------
//
//            03/20/2007 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////


#include "EventFilter/ResourceBroker/interface/SMProxy.h"

#include "xdaq/Application.h"

#include "toolbox/mem/Reference.h"
#include "toolbox/mem/MemoryPoolFactory.h"
#include "toolbox/mem/exception/Exception.h"

#include "i2o/Method.h"
#include "i2o/utils/AddressMap.h"

#include "xcept/tools.h"

#include <iostream>
#include <cmath>

using namespace std;
using namespace evf;

////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
SMProxy::SMProxy(xdaq::ApplicationDescriptor *fuAppDesc,
		xdaq::ApplicationDescriptor *smAppDesc,
		xdaq::ApplicationContext *fuAppContext, toolbox::mem::Pool *i2oPool) :
	log_(fuAppContext->getLogger()), fuAppDesc_(fuAppDesc),
			smAppDesc_(smAppDesc), fuAppContext_(fuAppContext),
			i2oPool_(i2oPool),
			initHeaderSize_(sizeof(I2O_SM_PREAMBLE_MESSAGE_FRAME)),
			dataHeaderSize_(sizeof(I2O_SM_DATA_MESSAGE_FRAME)),
			dqmHeaderSize_(sizeof(I2O_SM_DQM_MESSAGE_FRAME)) {
	fuUrl_ = fuAppDesc_->getContextDescriptor()->getURL();
	if (fuUrl_.size() >= MAX_I2O_SM_URLCHARS)
		fuUrl_ = fuUrl_.substr(0, MAX_I2O_SM_URLCHARS - 1);

	fuClassName_ = fuAppDesc_->getClassName();
	if (fuClassName_.size() >= MAX_I2O_SM_URLCHARS)
		fuClassName_ = fuClassName_.substr(0, MAX_I2O_SM_URLCHARS - 1);
}

//______________________________________________________________________________
SMProxy::~SMProxy() {

}

////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
UInt_t SMProxy::sendInitMessage(UInt_t fuResourceId, UInt_t outModId,
		UInt_t fuProcessId, UInt_t fuGuid, UChar_t*data, UInt_t dataSize,
		UInt_t nExpectedEPs) throw (evf::Exception) {
	UInt_t totalSize = 0;
	MemRef_t* bufRef = createFragmentChain(I2O_SM_PREAMBLE, initHeaderSize_,
			data, dataSize, totalSize);

	I2O_SM_PREAMBLE_MESSAGE_FRAME *msg;
	MemRef_t* next = bufRef;
	do {
		msg = (I2O_SM_PREAMBLE_MESSAGE_FRAME*) next->getDataLocation();
		msg->rbBufferID = fuResourceId;
		msg->outModID = outModId;
		msg->fuProcID = fuProcessId;
		msg->fuGUID = fuGuid;
		msg->nExpectedEPs = nExpectedEPs;
	} while ((next = next->getNextReference()));

	try {
		fuAppContext_->postFrame(bufRef, fuAppDesc_, smAppDesc_);
	} catch (xdaq::exception::Exception &e) {
		string msg = "Failed to post INIT Message.";
		XCEPT_RETHROW(evf::Exception, msg, e);
	}

	return totalSize;
}

//______________________________________________________________________________
UInt_t SMProxy::sendDataEvent(UInt_t fuResourceId, UInt_t runNumber,
		UInt_t evtNumber, UInt_t outModId, UInt_t fuProcessId, UInt_t fuGuid,
		UChar_t *data, UInt_t dataSize) throw (evf::Exception) {
	UInt_t totalSize = 0;
	MemRef_t* bufRef = createFragmentChain(I2O_SM_DATA, dataHeaderSize_, data,
			dataSize, totalSize);

	I2O_SM_DATA_MESSAGE_FRAME *msg;
	MemRef_t* next = bufRef;
	do {
		msg = (I2O_SM_DATA_MESSAGE_FRAME*) next->getDataLocation();
		msg->rbBufferID = fuResourceId;
		msg->runID = runNumber;
		msg->eventID = evtNumber;
		msg->outModID = outModId;
		msg->fuProcID = fuProcessId;
		msg->fuGUID = fuGuid;
	} while ((next = next->getNextReference()));

	try {
		fuAppContext_->postFrame(bufRef, fuAppDesc_, smAppDesc_);
	} catch (xdaq::exception::Exception &e) {
		string errmsg = "Failed to post DATA Message.";
		LOG4CPLUS_FATAL(log_, errmsg);
		XCEPT_RETHROW(evf::Exception, errmsg, e);
	}

	return totalSize;
}

//______________________________________________________________________________
UInt_t SMProxy::sendErrorEvent(UInt_t fuResourceId, UInt_t runNumber,
		UInt_t evtNumber, UInt_t fuProcessId, UInt_t fuGuid, UChar_t *data,
		UInt_t dataSize) throw (evf::Exception) {
	UInt_t totalSize = 0;
	MemRef_t* bufRef = createFragmentChain(I2O_SM_ERROR, dataHeaderSize_, data,
			dataSize, totalSize);

	I2O_SM_DATA_MESSAGE_FRAME *msg;
	MemRef_t* next = bufRef;
	do {
		msg = (I2O_SM_DATA_MESSAGE_FRAME*) next->getDataLocation();
		msg->rbBufferID = fuResourceId;
		msg->runID = runNumber;
		msg->eventID = evtNumber;
		msg->outModID = 0xffffffff;
		msg->fuProcID = fuProcessId;
		msg->fuGUID = fuGuid;
	} while ((next = next->getNextReference()));

	try {
		fuAppContext_->postFrame(bufRef, fuAppDesc_, smAppDesc_);
	} catch (xdaq::exception::Exception &e) {
		string errmsg = "Failed to post ERROR Message.";
		LOG4CPLUS_FATAL(log_, errmsg);
		XCEPT_RETHROW(evf::Exception, errmsg, e);
	}

	return totalSize;
}

//______________________________________________________________________________
UInt_t SMProxy::sendDqmEvent(UInt_t fuDqmId, UInt_t runNumber,
		UInt_t evtAtUpdate, UInt_t folderId, UInt_t fuProcessId, UInt_t fuGuid,
		UChar_t*data, UInt_t dataSize) throw (evf::Exception) {
	UInt_t totalSize = 0;
	MemRef_t* bufRef = createFragmentChain(I2O_SM_DQM, dqmHeaderSize_, data,
			dataSize, totalSize);

	I2O_SM_DQM_MESSAGE_FRAME *msg;
	MemRef_t* next = bufRef;
	do {
		msg = (I2O_SM_DQM_MESSAGE_FRAME*) next->getDataLocation();
		msg->rbBufferID = fuDqmId;
		msg->runID = runNumber;
		msg->eventAtUpdateID = evtAtUpdate;
		msg->folderID = folderId;
		msg->fuProcID = fuProcessId;
		msg->fuGUID = fuGuid;
	} while ((next = next->getNextReference()));

	try {
		fuAppContext_->postFrame(bufRef, fuAppDesc_, smAppDesc_);
	} catch (xdaq::exception::Exception &e) {
		string errmsg = "Failed to post DQM Message.";
		LOG4CPLUS_FATAL(log_, errmsg);
		XCEPT_RETHROW(evf::Exception, errmsg, e);
	}

	return totalSize;
}

////////////////////////////////////////////////////////////////////////////////
// implementation of private member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
MemRef_t* SMProxy::createFragmentChain(UShort_t i2oFunctionCode,
		UInt_t headerSize, UChar_t *data, UInt_t dataSize, UInt_t &totalSize)
		throw (evf::Exception) {
	totalSize = 0;

	UInt_t fragmentDataSizeMax = I2O_MAX_SIZE - headerSize;
	UInt_t fragmentCount = (dataSize / fragmentDataSizeMax);
	if (dataSize % fragmentDataSizeMax)
		++fragmentCount;

	UInt_t currentPosition = 0;
	UInt_t remainingDataSize = dataSize;

	MemRef_t *head(0);
	MemRef_t *tail(0);

	try {

		for (UInt_t iFragment = 0; iFragment < fragmentCount; iFragment++) {

			UInt_t fragmentDataSize = fragmentDataSizeMax;
			UInt_t fragmentSize = fragmentDataSize + headerSize;

			if (remainingDataSize < fragmentDataSizeMax) {
				fragmentDataSize = remainingDataSize;
				fragmentSize = fragmentDataSize + headerSize;
				if (fragmentSize & 0x7)
					fragmentSize = ((fragmentSize >> 3) + 1) << 3;
			}

			// allocate the fragment buffer from the pool
			toolbox::mem::Reference *bufRef =
					toolbox::mem::getMemoryPoolFactory()->getFrame(i2oPool_,
							fragmentSize);

			// set up pointers to the allocated message buffer
			I2O_MESSAGE_FRAME *stdMsg;
			I2O_PRIVATE_MESSAGE_FRAME *pvtMsg;
			I2O_SM_MULTIPART_MESSAGE_FRAME *msg;

			stdMsg = (I2O_MESSAGE_FRAME*) bufRef->getDataLocation();
			pvtMsg = (I2O_PRIVATE_MESSAGE_FRAME*) stdMsg;
			msg = (I2O_SM_MULTIPART_MESSAGE_FRAME*) stdMsg;

			stdMsg->VersionOffset = 0;
			stdMsg->MsgFlags = 0; // normal message (not multicast)
			stdMsg->MessageSize = fragmentSize >> 2;
			stdMsg->Function = I2O_PRIVATE_MESSAGE;
			stdMsg->InitiatorAddress = i2o::utils::getAddressMap()->getTid(
					fuAppDesc_);
			stdMsg->TargetAddress = i2o::utils::getAddressMap()->getTid(
					smAppDesc_);

			pvtMsg->XFunctionCode = i2oFunctionCode;
			pvtMsg->OrganizationID = XDAQ_ORGANIZATION_ID;

			msg->dataSize = fragmentDataSize;
			msg->hltLocalId = fuAppDesc_->getLocalId();
			msg->hltInstance = fuAppDesc_->getInstance();
			msg->hltTid = i2o::utils::getAddressMap()->getTid(fuAppDesc_);
			msg->numFrames = fragmentCount;
			msg->frameCount = iFragment;
			msg->originalSize = dataSize;

			for (UInt_t i = 0; i < fuUrl_.size(); i++)
				msg->hltURL[i] = fuUrl_[i];
			msg->hltURL[fuUrl_.size()] = '\0';

			for (UInt_t i = 0; i < fuClassName_.size(); i++)
				msg->hltClassName[i] = fuClassName_[i];
			msg->hltClassName[fuClassName_.size()] = '\0';

			if (iFragment == 0) {
				head = bufRef;
				tail = bufRef;
			} else {
				tail->setNextReference(bufRef);
				tail = bufRef;
			}

			if (fragmentDataSize != 0) {
				UChar_t* targetAddr = (UChar_t*) msg + headerSize;
				std::copy(data + currentPosition,
						data + currentPosition + fragmentDataSize, targetAddr);
			}

			bufRef->setDataSize(fragmentSize);
			remainingDataSize -= fragmentDataSize;
			currentPosition += fragmentDataSize;
			totalSize += fragmentSize;

		} // for (iFragment ...)
	} catch (toolbox::mem::exception::Exception& e) {
		if (0 != head)
			head->release();
		totalSize = 0;
		string errmsg = "Failed to allocate buffer reference.";
		LOG4CPLUS_FATAL(log_, errmsg);
		XCEPT_RETHROW(evf::Exception, errmsg, e);
	} catch (xdaq::exception::ApplicationDescriptorNotFound& e) {
		if (0 != head)
			head->release();
		totalSize = 0;
		string errmsg = "Failed to get tid.";
		LOG4CPLUS_FATAL(log_, errmsg);
		XCEPT_RETHROW(evf::Exception, errmsg, e);
	}

	return head;
}
