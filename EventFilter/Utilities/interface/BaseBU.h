/*
 * BaseBU.h
 *
 *  Created on: Aug 18, 2011
 *      Author: aspataru
 */

#ifndef BASEBU_H_
#define BASEBU_H_

#include "xdaq/Application.h"
#include "EventFilter/Utilities/interface/BUFUTypes.h"

namespace evf {

class BaseBU {

public:
	/*
	 * functions to be performed when receiving an ALLOCATE or DISCARD through direct call
	 */
	/// ALLOCATE called by the BUFU Interface when BU and FU are in the same process
	virtual void DIRECT_BU_ALLOCATE(const UIntVec_t& fuResourceIds, xdaq::ApplicationDescriptor* fuAppDesc) = 0;
	/// DISCARD called by the BUFU Interface when BU and FU are in the same process
	virtual void DIRECT_BU_DISCARD(UInt_t buResourceId) = 0;

	/// to be called by the BUFUInterface
	/// the child class BU implements this function, posting a frame
	virtual void postI2OFrame(xdaq::ApplicationDescriptor* fuAppDesc,
			toolbox::mem::Reference* bufRef) = 0;
};
}

#endif /* BASEBU_H_ */
