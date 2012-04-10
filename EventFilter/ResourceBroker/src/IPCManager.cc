////////////////////////////////////////////////////////////////////////////////
//
// IPCManager.cc
// -------
//
//  Created on: Oct 26, 2011
//      									Andrei Spataru : aspataru@cern.ch
////////////////////////////////////////////////////////////////////////////////

#include "EventFilter/ResourceBroker/interface/IPCManager.h"
#include "EventFilter/ResourceBroker/interface/FUResourceTable.h"
#include "EventFilter/ResourceBroker/interface/FUResourceQueue.h"

using std::cout;
using std::endl;
using namespace evf;

//______________________________________________________________________________
IPCManager::IPCManager(bool useMQ) :
	useMessageQueueIPC_(useMQ) {
}

//______________________________________________________________________________
IPCManager::~IPCManager() {
	delete ipc_;
}

//______________________________________________________________________________
void IPCManager::initialise(bool segmentationMode, UInt_t nbRawCells,
		UInt_t nbRecoCells, UInt_t nbDqmCells, UInt_t rawCellSize,
		UInt_t recoCellSize, UInt_t dqmCellSize, BUProxy* bu, SMProxy* sm,
		log4cplus::Logger logger, unsigned int resourceStructureTimeout,
		EvffedFillerRB* frb, xdaq::Application* app) throw (evf::Exception) {

	// XXX: MQ IPC is disabled until it is ready
	useMessageQueueIPC_ = false;

	if (!useMessageQueueIPC_) {
		// IPC communication type is SharedMemory
		//improve replace with logging
		cout << "IPCMANAGER:: ----> IPC communication type is Shared Memory!"
				<< endl;

		ipc_ = new FUResourceTable(segmentationMode, nbRawCells, nbRecoCells,
				nbDqmCells, rawCellSize, recoCellSize, dqmCellSize, bu, sm,
				logger, resourceStructureTimeout, frb, app);
	} else {
		// IPC communication type is MessageQueue
		cout << "IPCMANAGER:: ----> IPC communication type is Message Queue!"
				<< endl;

		ipc_ = new FUResourceQueue(segmentationMode, nbRawCells, nbRecoCells,
				nbDqmCells, rawCellSize, recoCellSize, dqmCellSize, bu, sm,
				logger, resourceStructureTimeout, frb, app);
	}
}

//______________________________________________________________________________
IPCMethod* const IPCManager::ipc() {
	return ipc_;
}
