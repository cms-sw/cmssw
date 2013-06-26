/*
 * BUFUInterface.h
 *
 *  Created on: Aug 16, 2011
 *      Author: aspataru
 */

#ifndef BUFUINTERFACE_H_
#define BUFUINTERFACE_H_

#include "BaseBU.h"
#include "BaseFU.h"
#include "xdaq/Application.h"

namespace evf {

class BUFUInterface {

public:
	~BUFUInterface();

	/// returns a pointer to a singleton instance of the interface
	static BUFUInterface* instance();
	/**
	 * always returns a new instance.
	 * used to force I2O communication
	 * between BU and FU in the same process
	 */
	static BUFUInterface* forceNewInstance();

	/// register the BU to the interface
	bool registerBU(BaseBU* bu, Logger log);
	/// register the FU to the interface
	bool registerFU(BaseFU* fu, Logger log);

	/// FU->BU ALLOCATE
	/// the FU application description is required by the BU when registering
	/// the requesting FU
	void allocate(const UIntVec_t& fuResourceIds,
			xdaq::ApplicationDescriptor* fuAppDesc);
	/// FU->BU DISCARD
	void discard(UInt_t buResourceId);
	/// BU->FU TAKE
	void take(xdaq::ApplicationDescriptor* fuAppDesc,
			toolbox::mem::Reference* bufRef);

private:
	/// pointer to unique instance
	static BUFUInterface* instance_;

	/// private constructor
	BUFUInterface();

	/*
	 * private functions
	 */
	/// checks if both BU and FU are connected to the interface
	inline bool directConnection() const {
		return buConn_ && fuConn_;
	}

	/*
	 * private fields
	 */
	/// pointer to connected BU, using BaseBU interface
	BaseBU* bu_;
	/// pointer to connected ResourceBroker(FU), using BaseFU interface
	BaseFU* fu_;
	/// loggers
	Logger buLogger_, fuLogger_;
	/// flags for BU, FU connection
	bool buConn_, fuConn_;

};
}

#endif /* BUFUINTERFACE_H_ */
