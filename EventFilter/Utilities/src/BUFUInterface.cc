/*
 * BUFUInterface.cc
 *
 *  Created on: Aug 16, 2011
 *      Author: aspataru
 */

#include "EventFilter/Utilities/interface/BUFUInterface.h"

#define BUFU_VERBOSE
//#define BUFU_DEBUG

using namespace evf;

// initialize instance_ to 0
BUFUInterface* evf::BUFUInterface::instance_ = 0;

//______________________________________________________________________________
BUFUInterface::~BUFUInterface() {
	// delete instance_;
	instance_ = 0;
}

//______________________________________________________________________________
BUFUInterface* BUFUInterface::instance() {
	if (instance_ == 0) {
		instance_ = new BUFUInterface();
	}
	return instance_;

}

//______________________________________________________________________________
BUFUInterface* BUFUInterface::forceNewInstance() {
	return new BUFUInterface();
}

//______________________________________________________________________________
bool BUFUInterface::registerBU(BaseBU* bu, Logger* log) {
	bool succeeded = false;
	if (bu != 0) {
		bu_ = bu;
		buLogger_ = log;
		buConn_ = true;
		succeeded = true;

#ifdef BUFU_VERBOSE
		LOG4CPLUS_INFO(*buLogger_, "BU registered to BUFU interface!");
		if (directConnection()) {
			LOG4CPLUS_INFO(*buLogger_, "BU and FU : DIRECTLY CONNECTED!");
		}
#endif

	}
	return succeeded;
}

//______________________________________________________________________________
bool BUFUInterface::registerFU(BaseFU* fu, Logger* log) {
	bool succeeded = false;
	if (fu != 0) {
		fu_ = fu;
		fuLogger_ = log;
		fuConn_ = true;
		succeeded = true;

#ifdef BUFU_VERBOSE
		LOG4CPLUS_INFO(*fuLogger_, "FU registered to BUFU interface!");
		if (directConnection()) {
			LOG4CPLUS_INFO(*buLogger_, "BU and FU : DIRECTLY CONNECTED!");
		}
#endif

	}
	return succeeded;
}

//______________________________________________________________________________
void BUFUInterface::allocate(const UIntVec_t& fuResourceIds,
		xdaq::ApplicationDescriptor* fuAppDesc) {

	// direct connection -> method call
	if (directConnection()) {

		bu_->DIRECT_BU_ALLOCATE(fuResourceIds, fuAppDesc);

#ifdef BUFU_DEBUG
		std::cout << "ALLOCATE FU->BU message sent directly!" << std::endl;
#endif
	}

	// no connection -> line protocol
	else {
		// call FU to build I2O allocate message and send it
		fu_->buildAndSendAllocate(fuResourceIds);

#ifdef BUFU_DEBUG
		std::cout << "ALLOCATE FU->BU message sent through I2O!" << std::endl;
#endif
	}

}

//______________________________________________________________________________
void BUFUInterface::discard(UInt_t buResourceId) {

	// direct connection -> method call
	if (directConnection()) {

		bu_->DIRECT_BU_DISCARD(buResourceId);

#ifdef BUFU_DEBUG
		std::cout << "DISCARD FU->BU message sent directly!" << std::endl;
#endif
	}

	// no connection -> line protocol
	else {
		// call FU to build I2O discard and send it
		fu_->buildAndSendDiscard(buResourceId);

#ifdef BUFU_DEBUG
		std::cout << "DISCARD FU->BU message sent through I2O!" << std::endl;
#endif
	}

}

//______________________________________________________________________________
void BUFUInterface::take(xdaq::ApplicationDescriptor* fuAppDesc,
		toolbox::mem::Reference* bufRef) {

	// direct connection -> method call
	if (directConnection()) {

		fu_->I2O_FU_TAKE_Callback(bufRef);

#ifdef BUFU_DEBUG
		std::cout << "TAKE BU->FU message sent directly!" << std::endl;
#endif
	}

	// no connection -> line protocol
	else {
		bu_->postI2OFrame(fuAppDesc, bufRef);

#ifdef BUFU_DEBUG
		std::cout << "TAKE BU->FU message sent through I2O!" << std::endl;
#endif
	}

}

//______________________________________________________________________________
BUFUInterface::BUFUInterface() :
	bu_(0), fu_(0), buLogger_(0), fuLogger_(0), buConn_(false), fuConn_(false) {

}
