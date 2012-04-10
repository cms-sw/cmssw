////////////////////////////////////////////////////////////////////////////////
//
// IPCManager.h
// -------
//
//  Created on: Oct 26, 2011
//      									Andrei Spataru : aspataru@cern.ch
////////////////////////////////////////////////////////////////////////////////

#ifndef IPCMANAGER_H_
#define IPCMANAGER_H_

#include "EventFilter/ResourceBroker/interface/IPCMethod.h"
#include "xdaq/Application.h"
#include "FWCore/Utilities/interface/CRC16.h"

namespace evf {

/**
 * Class used to create and initialize an IPC method of the required type.
 * If constructed with argument <false>, the manager will create and initialize a FUResourceTable object.
 * If constructed with argument <true>, the manager will create and initialize a FUResourceQueue object.
 *
 * $Author: aspataru $
 *
 */
class IPCManager {

public:
	IPCManager(bool useMQ);
	virtual ~IPCManager();

	/**
	 * Initialize the IPCMethod object with the given parameters.
	 */
	void initialise(bool segmentationMode, UInt_t nbRawCells,
			UInt_t nbRecoCells, UInt_t nbDqmCells, UInt_t rawCellSize,
			UInt_t recoCellSize, UInt_t dqmCellSize, BUProxy* bu, SMProxy* sm,
			log4cplus::Logger logger, unsigned int resourceStructureTimeout,
			EvffedFillerRB*frb, xdaq::Application*) throw (evf::Exception);

	/**
	 * Returns a const pointer to the IPCMethod object created by the manager.
	 */
	IPCMethod* const ipc();

private:
	// selected IPC method
	//	true - Message Queue
	//	false - Shared Memory
	bool useMessageQueueIPC_;
	// pointer to IPC communication type
	IPCMethod* ipc_;

};

}

#endif /* IPCMANAGER_H_ */
