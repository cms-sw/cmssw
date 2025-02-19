////////////////////////////////////////////////////////////////////////////////
//
// FUResourceQueue
// ---------------
//
//            28/10/2011 Andrei Spataru <aspataru@cern.ch>
////////////////////////////////////////////////////////////////////////////////

#ifndef FURESOURCEQUEUE_H
#define FURESOURCEQUEUE_H 1

#include "EventFilter/ResourceBroker/interface/RawCache.h"
#include "EventFilter/ResourceBroker/interface/FUTypes.h"
#include "EventFilter/Utilities/interface/MasterQueue.h"
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

class FUResourceQueue: public IPCMethod {
public:
	//
	// construction/destruction
	//
	FUResourceQueue(bool segmentationMode, UInt_t nbRawCells,
			UInt_t nbRecoCells, UInt_t nbDqmCells, UInt_t rawCellSize,
			UInt_t recoCellSize, UInt_t dqmCellSize, int freeResReq,
			BUProxy* bu, SMProxy* sm, log4cplus::Logger logger, unsigned int,
			EvffedFillerRB*frb, xdaq::Application*) throw (evf::Exception);
	virtual ~FUResourceQueue();

	//
	// member functions
	//
	// initialization of the resource queue
	void initialize(bool segmentationMode, UInt_t nbRawCells,
			UInt_t nbRecoCells, UInt_t nbDqmCells, UInt_t rawCellSize,
			UInt_t recoCellSize, UInt_t dqmCellSize) throw (evf::Exception);

	// work loop to send data events to storage manager
	bool sendData();
	//improvement: make simpler
	bool sendDataWhileHalting();

	// work loop to send dqm events to storage manager
	bool sendDqm();
	bool sendDqmWhileHalting();

	// work loop to discard events to builder unit
	bool discard();
	bool discardWhileHalting(bool sendDiscards);

	// process buffer received via I2O_FU_TAKE message
	bool buildResource(MemRef_t* bufRef);

	// process buffer received via I2O_SM_DATA_DISCARD message
	bool discardDataEvent(MemRef_t* bufRef);
	bool discardDataEventWhileHalting(MemRef_t* bufRef);

	// process buffer received via I2O_SM_DQM_DISCARD message
	bool discardDqmEvent(MemRef_t* bufRef);
	bool discardDqmEventWhileHalting(MemRef_t* bufRef);

	// post end-of-ls event to shmem
	void postEndOfLumiSection(MemRef_t* bufRef);

	// drop next available event
	void dropEvent();

	// send event belonging to crashed process to error stream (return false
	// if no event is found)
	bool handleCrashedEP(UInt_t runNumber, pid_t pid);

	// dump event to ascii file
	void dumpEvent(evf::FUShmRawCell* cell);

	// send empty events to notify clients to shutdown
	void shutDownClients();

	void clear();

	// reset event & error counters
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
	void lastResort();
	void resetIPC() {
		// nothing here
	}

private:
	//
	// member data
	//

	MasterQueue msq_;
	RawCache* cache_;

	// XXX move?
	UInt_t rawCellSize_, recoCellSize_, dqmCellSize_;

};

} // namespace evf


#endif // FURESOURCEQUEUE_H
