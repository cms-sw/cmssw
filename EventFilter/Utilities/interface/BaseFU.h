/*
 * BaseFU.h
 *
 *  Created on: Aug 19, 2011
 *      Author: aspataru
 */

#ifndef BASEFU_H_
#define BASEFU_H_

#include "xdaq/Application.h"
#include "EventFilter/Utilities/interface/BUFUTypes.h"

namespace evf {

class BaseFU {

public:
	/// function to be performed upon receiving a TAKE from the BU
	virtual void I2O_FU_TAKE_Callback(toolbox::mem::Reference* bufRef) = 0;

	/// called by the BUFU Interface to trigger an I2O ALLOCATE message being built and sent
	virtual void buildAndSendAllocate(const UIntVec_t& fuResourceIds) = 0;
	/// called by the BUFU Interface to trigger an I2O DISCARD message being built and sent
	virtual void buildAndSendDiscard(UInt_t buResourceId) = 0;

};
}

#endif /* BASEFU_H_ */
