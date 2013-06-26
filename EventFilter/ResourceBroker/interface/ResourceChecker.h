/*
 * ResourceChecker.h
 *
 *  Created on: Nov 23, 2011
 *
 *      Author:
 *      Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
 *      Andrei Spataru <aspataru@cern.ch>
 */

#ifndef RESOURCECHECKER_H_
#define RESOURCECHECKER_H_

#include "EventFilter/ResourceBroker/interface/FUResource.h"

namespace evf {

/**
 * Checks resource data.
 *
 * $Author: aspataru $
 *
 */

class ResourceChecker {

public:
	ResourceChecker(FUResource* const resToCheck);

	/**
	 * Performs checks on Data Blocks received from the BU.
	 */
	void processDataBlock(MemRef_t* bufRef) throw (evf::Exception);

private:
	/**
	 * Performs checks on the Data Block payload.
	 */
	void checkDataBlockPayload(MemRef_t* bufRef) throw (evf::Exception);
	/**
	 * Finds individual FED frames in the data block payload by using
	 * FED header information.
	 */
	void findFEDs() throw (evf::Exception);

private:
	FUResource* const res_;

};

}

#endif /* RESOURCECHECKER_H_ */
