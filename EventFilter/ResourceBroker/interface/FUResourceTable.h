////////////////////////////////////////////////////////////////////////////////
//
// FUResourceTable
// ---------------
//
//            12/10/2006 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
//            20/01/2012 Andrei Spataru <aspataru@cern.ch>
////////////////////////////////////////////////////////////////////////////////

#ifndef FURESOURCETABLE_H
#define FURESOURCETABLE_H 1

#include "EventFilter/ResourceBroker/interface/BUProxy.h"
#include "EventFilter/ResourceBroker/interface/SMProxy.h"
#include "EventFilter/ResourceBroker/interface/FUTypes.h"
#include "EventFilter/ShmBuffer/interface/FUShmBuffer.h"
#include "EventFilter/Utilities/interface/Exception.h"

#include "log4cplus/logger.h"
#include "toolbox/lang/Class.h"
#include "toolbox/task/Action.h"
#include "toolbox/task/WorkLoop.h"

#include <sys/types.h>
#include <string>
#include <vector>
#include <queue>

#include "IPCMethod.h"

namespace evf {

/**
 * Table of resources linked with the Shared Memory Buffer
 *
 * $Author: aspataru $
 *
 */

class FUResourceTable: public IPCMethod {
public:
	//
	// construction/destruction
	//
	FUResourceTable(bool segmentationMode, UInt_t nbRawCells,
			UInt_t nbRecoCells, UInt_t nbDqmCells, UInt_t rawCellSize,
			UInt_t recoCellSize, UInt_t dqmCellSize, int freeResReq, BUProxy* bu, SMProxy* sm,
			log4cplus::Logger logger, unsigned int, EvffedFillerRB* frb,
			xdaq::Application*) throw (evf::Exception);
	virtual ~FUResourceTable();

	//
	// member functions
	//

	/**
	 * Initialization of the Resource Table with the required resources
	 */
	void initialize(bool segmentationMode, UInt_t nbRawCells,
			UInt_t nbRecoCells, UInt_t nbDqmCells, UInt_t rawCellSize,
			UInt_t recoCellSize, UInt_t dqmCellSize) throw (evf::Exception);

	/**
	 * Corresponds to the SendData workloop, to be called in normal operation.
	 */
	bool sendData();
	/**
	 * Function called when FSM is in state stopping / halting, in SendData workloop.
	 */
	bool sendDataWhileHalting();

	/**
	 * Corresponds to the SendDqm workloop, to be called in normal operation.
	 */
	bool sendDqm();
	/**
	 * Function called when FSM is in state stopping / halting, in SendDqm workloop.
	 */
	bool sendDqmWhileHalting();

	/**
	 * Corresponds to the Discard workloop, to be called in normal operation.
	 */
	bool discard();
	/**
	 * Function called when FSM is in state stopping / halting, in Discard workloop.
	 */
	bool discardWhileHalting(bool sendDiscards);

	/**
	 * Process buffer received via I2O_FU_TAKE message
	 */
	bool buildResource(MemRef_t* bufRef);

	/**
	 * Process buffer received via I2O_SM_DATA_DISCARD message in normal operation.
	 */
	bool discardDataEvent(MemRef_t* bufRef);

	/**
	 * Process buffer received via I2O_SM_DATA_DISCARD message while halting.
	 */
	bool discardDataEventWhileHalting(MemRef_t* bufRef);

	/**
	 * Process buffer received via I2O_SM_DQM_DISCARD message in normal operation.
	 */
	bool discardDqmEvent(MemRef_t* bufRef);
	/**
	 * Process buffer received via I2O_SM_DQM_DISCARD message while halting.
	 */
	bool discardDqmEventWhileHalting(MemRef_t* bufRef);

	/**
	 * Post End-of-LumiSection to Shared Memory.
	 */
	void postEndOfLumiSection(MemRef_t* bufRef);

	/**
	 * Drop next available event.
	 */
	void dropEvent();

	/**
	 * Send event belonging to crashed process to error stream.
	 * Return false if no event is found
	 */
	bool handleCrashedEP(UInt_t runNumber, pid_t pid);

	/**
	 * Send empty events to notify clients to shutdown
	 */
	void shutDownClients();

	/**
	 * Clear contents of resource table. (empty all containers)
	 */
	void clear();

	/**
	 * Reset event & error counters
	 */
	void resetCounters();

	UInt_t nbResources() const {
		return resources_.size();
	}

	// information about (raw) shared memory cells
	UInt_t nbClients() const;
	std::vector<pid_t> clientPrcIds() const;
	std::string clientPrcIdsAsString() const;
	std::vector<std::string> cellStates() const;
	std::vector<std::string> dqmCellStates() const;
	std::vector<UInt_t> cellEvtNumbers() const;
	std::vector<pid_t> cellPrcIds() const;
	std::vector<time_t> cellTimeStamps() const;

	//
	// helpers
	//
	unsigned int nbResources() {
		return resources_.size();
	}
	void lastResort();
	/// reset the ShmBuffer to the initial state
	void resetIPC();

private:

	/**
	 * Called when entering the shutdown cycle.
	 * The function sets readyToShutDown to true, allowing the Resource Table to be safely shut down.
	 */
	void discardNoReschedule();

	/**
	 * Rethrows an exception from the ShmBuffer including details.
	 */
	void rethrowShmBufferException(evf::Exception& e, std::string where) const throw (evf::Exception);

	FUShmBuffer *shmBuffer_;

};

} // namespace evf


#endif
