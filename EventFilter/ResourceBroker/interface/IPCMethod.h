////////////////////////////////////////////////////////////////////////////////
//
// IPCMethod.h
// -------
//
// Contains common functionality for FUResourceTable and FUResourceQueue.
//
//  Created on: Oct 26, 2011
//      									Andrei Spataru : aspataru@cern.ch
////////////////////////////////////////////////////////////////////////////////


#ifndef IPCMETHOD_H_
#define IPCMETHOD_H_

#include "EventFilter/ResourceBroker/interface/FUResource.h"
#include "EventFilter/ResourceBroker/interface/BUProxy.h"
#include "EventFilter/ResourceBroker/interface/SMProxy.h"
#include "EventFilter/ResourceBroker/interface/FUTypes.h"
#include "EventFilter/Utilities/interface/Exception.h"

#include "log4cplus/logger.h"

#include <sys/types.h>
#include <string>
#include <vector>
#include <queue>
#include <semaphore.h>

namespace evf {

/**
 * Base class for methods (types) of IPC. Subclasses: FUResourceTable, FUResourceQueue.
 *
 * $Author: aspataru $
 *
 */

class IPCMethod: public toolbox::lang::Class {

public:
	//
	// construction/destruction
	//
	IPCMethod(bool segmentationMode, UInt_t nbRawCells, UInt_t nbRecoCells,
			UInt_t nbDqmCells, UInt_t rawCellSize, UInt_t recoCellSize,
			UInt_t dqmCellSize, int freeResReq, BUProxy *bu, SMProxy *sm,
			log4cplus::Logger logger, unsigned int timeout,
			EvffedFillerRB *frb, xdaq::Application*app) throw (evf::Exception);

	virtual ~IPCMethod();

	//
	// member functions
	//

	// set the run number
	void setRunNumber(UInt_t runNumber) {
		runNumber_ = runNumber;
	}

	/**
	 * Has to be implemented by subclasses, according to IPC type.
	 */
	virtual bool sendData() = 0;
	/**
	 * Has to be implemented by subclasses, according to IPC type.
	 */
	virtual bool sendDataWhileHalting() = 0;
	/**
	 * Has to be implemented by subclasses, according to IPC type.
	 */
	virtual bool sendDqm() = 0;
	/**
	 * Has to be implemented by subclasses, according to IPC type.
	 */
	virtual bool sendDqmWhileHalting() = 0;
	/**
	 * Has to be implemented by subclasses, according to IPC type.
	 */
	virtual bool discard() = 0;
	/**
	 * Has to be implemented by subclasses, according to IPC type.
	 */
	virtual bool discardWhileHalting(bool sendDiscards) = 0;

	/**
	 * Returns the fuResourceId of the allocated resource
	 */
	UInt_t allocateResource();

	/**
	 * Has to be implemented by subclasses, according to IPC type.
	 */
	virtual bool buildResource(MemRef_t* bufRef) = 0;

	/**
	 * Has to be implemented by subclasses, according to IPC type.
	 */
	virtual bool discardDataEvent(MemRef_t* bufRef) = 0;
	/**
	 * Has to be implemented by subclasses, according to IPC type.
	 */
	virtual bool discardDataEventWhileHalting(MemRef_t* bufRef) = 0;

	/**
	 * Has to be implemented by subclasses, according to IPC type.
	 */
	virtual bool discardDqmEvent(MemRef_t* bufRef) = 0;
	/**
	 * Has to be implemented by subclasses, according to IPC type.
	 */
	virtual bool discardDqmEventWhileHalting(MemRef_t* bufRef) = 0;

	/**
	 * Has to be implemented by subclasses, according to IPC type.
	 */
	virtual void postEndOfLumiSection(MemRef_t* bufRef) = 0;

	/**
	 * Has to be implemented by subclasses, according to IPC type.
	 */
	virtual void dropEvent() = 0;

	/**
	 * Has to be implemented by subclasses, according to IPC type.
	 */
	virtual bool handleCrashedEP(UInt_t runNumber, pid_t pid) = 0;

	/**
	 * Dump event to ASCII file.
	 */
	void dumpEvent(evf::FUShmRawCell* cell);

	/**
	 * Has to be implemented by subclasses, according to IPC type.
	 */
	virtual void shutDownClients() = 0;

	/**
	 * Has to be implemented by subclasses, according to IPC type.
	 */
	virtual void clear() = 0;

	/**
	 * Has to be implemented by subclasses, according to IPC type.
	 */
	virtual void resetCounters() = 0;

	/**
	 * Tell resources whether to check the CRC
	 */
	void setDoCrcCheck(UInt_t doCrcCheck) {
		doCrcCheck_ = doCrcCheck;
	}

	/**
	 * Tell resources whether to dump events to an ASCII file.
	 */
	void setDoDumpEvents(UInt_t doDumpEvents) {
		doDumpEvents_ = doDumpEvents;
	}

	/**
	 * Check if resource table is active (enabled).
	 */
	bool isActive() const {
		return isActive_;
	}

	void setActive(bool activeValue) {
		isActive_ = activeValue;
	}

	/**
	 * Check if resource table can be safely destroyed.
	 */
	bool isReadyToShutDown() const {
		return isReadyToShutDown_;
	}

	void setReadyToShutDown(bool readyValue) {
		isReadyToShutDown_ = readyValue;
	}

	// various counters
	virtual UInt_t nbResources() const = 0; /* Implemented in subclass */
	UInt_t nbFreeSlots() const {
		return freeResourceIds_.size();
	}
	UInt_t nbAllocated() const {
		return nbAllocated_;
	}
	UInt_t nbPending() const {
		return nbPending_;
	}
	UInt_t nbCompleted() const {
		return nbCompleted_;
	}
	UInt_t nbSent() const {
		return nbSent_;
	}
	UInt_t nbSentError() const {
		return nbSentError_;
	}
	UInt_t nbSentDqm() const {
		return nbSentDqm_;
	}
	UInt_t nbPendingSMDiscards() const {
		return nbPendingSMDiscards_;
	}
	UInt_t nbPendingSMDqmDiscards() const {
		return nbPendingSMDqmDiscards_;
	}
	UInt_t nbDiscarded() const {
		return nbDiscarded_;
	}
	UInt_t nbLost() const {
		return nbLost_;
	}

	// UPDATED
	UInt_t nbEolPosted() const {
		return nbEolPosted_;
	}
	UInt_t nbEolDiscarded() const {
		return nbEolDiscarded_;
	}

	UInt_t nbErrors() const {
		return nbErrors_;
	}
	UInt_t nbCrcErrors() const {
		return nbCrcErrors_;
	}
	UInt_t nbAllocSent() const {
		return nbAllocSent_;
	}

	uint64_t sumOfSquares() const {
		return sumOfSquares_;
	}
	UInt_t sumOfSizes() const {
		return sumOfSizes_;
	}

	// information about (raw) shared memory cells
	virtual UInt_t nbClients() const = 0;
	virtual std::vector<pid_t> clientPrcIds() const = 0;
	virtual std::string clientPrcIdsAsString() const = 0;
	virtual std::vector<std::string> cellStates() const = 0;
	virtual std::vector<std::string> dqmCellStates() const = 0;
	virtual std::vector<UInt_t> cellEvtNumbers() const = 0;
	virtual std::vector<pid_t> cellPrcIds() const = 0;
	virtual std::vector<time_t> cellTimeStamps() const = 0;

	//
	// helpers
	//
	void sendAllocate();
	/// resets free resources to the maximum number
	void resetPendingAllocates();
	/// releases all FUResource's
	void releaseResources();
	/// resets the underlying IPC method to the initial state
	virtual void resetIPC() = 0;

	void sendDiscard(UInt_t buResourceId);

	void sendInitMessage(UInt_t fuResourceId, UInt_t outModId,
			UInt_t fuProcessId, UInt_t fuGuid, UChar_t*data, UInt_t dataSize,
			UInt_t nExpectedEPs);

	void sendDataEvent(UInt_t fuResourceId, UInt_t runNumber, UInt_t evtNumber,
			UInt_t outModId, UInt_t fuProcessId, UInt_t fuGuid, UChar_t*data,
			UInt_t dataSize);

	void sendErrorEvent(UInt_t fuResourceId, UInt_t runNumber,
			UInt_t evtNumber, UInt_t fuProcessId, UInt_t fuGuid, UChar_t*data,
			UInt_t dataSize);

	void sendDqmEvent(UInt_t fuDqmId, UInt_t runNumber, UInt_t evtAtUpdate,
			UInt_t folderId, UInt_t fuProcessId, UInt_t fuGuid, UChar_t*data,
			UInt_t dataSize);

	bool isLastMessageOfEvent(MemRef_t* bufRef);

	void injectCRCError();

	void lock() {
		//lock_.take();
		while (0 != sem_wait(&lock_)) {
			if (errno != EINTR) {
				LOG4CPLUS_ERROR(log_, "Cannot obtain lock on sem LOCK!");
			}
		}
	}
	void unlock() {
		//lock_.give();
		sem_post(&lock_);
	}

	/**
	 * Has to be implemented by subclasses, according to IPC type.
	 */
	virtual void lastResort() = 0;

protected:
	//
	// member data
	//

	BUProxy *bu_;
	SMProxy *sm_;

	log4cplus::Logger log_;

	UInt_t nbDqmCells_;
	UInt_t nbRawCells_;
	UInt_t nbRecoCells_;

	std::queue<UInt_t> freeResourceIds_;
	// number of free resources required to ask BU for more events
	unsigned int freeResRequiredForAllocate_;

	bool *acceptSMDataDiscard_;
	int *acceptSMDqmDiscard_;

	UInt_t doCrcCheck_;
	UInt_t doDumpEvents_;
	unsigned int shutdownTimeout_;

	UInt_t nbAllocated_;
	UInt_t nbPending_;
	UInt_t nbCompleted_;
	UInt_t nbSent_;
	UInt_t nbSentError_;
	UInt_t nbSentDqm_;
	UInt_t nbPendingSMDiscards_;
	UInt_t nbPendingSMDqmDiscards_;
	UInt_t nbDiscarded_;
	UInt_t nbLost_;
	// UPDATED
	UInt_t nbEolPosted_;
	UInt_t nbEolDiscarded_;

	UInt_t nbClientsToShutDown_;
	bool isReadyToShutDown_;
	bool isActive_;

	UInt_t nbErrors_;
	UInt_t nbCrcErrors_;
	UInt_t nbAllocSent_;

	uint64_t sumOfSquares_;
	UInt_t sumOfSizes_;

	UInt_t runNumber_;

	sem_t lock_;
	EvffedFillerRB *frb_;
	xdaq::Application *app_;

	FUResourceVec_t resources_;

};

}

#endif /* IPCMETHOD_H_ */
