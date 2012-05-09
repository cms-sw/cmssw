////////////////////////////////////////////////////////////////////////////////
//
// FUResourceTable
// ---------------
//
//            12/10/2006 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
//            20/01/2012 Andrei Spataru <aspataru@cern.ch>
////////////////////////////////////////////////////////////////////////////////

#include "EventFilter/ResourceBroker/interface/FUResourceTable.h"
#include "EvffedFillerRB.h"

#include "interface/evb/i2oEVBMsgs.h"
#include "xcept/tools.h"

#include <sys/types.h>
#include <signal.h>

//#define DEBUG_RES_TAB

using namespace evf;
using namespace std;

////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
FUResourceTable::FUResourceTable(bool segmentationMode, UInt_t nbRawCells,
		UInt_t nbRecoCells, UInt_t nbDqmCells, UInt_t rawCellSize,
		UInt_t recoCellSize, UInt_t dqmCellSize, int freeResReq, BUProxy *bu,
		SMProxy *sm, log4cplus::Logger logger, unsigned int timeout,
		EvffedFillerRB *frb, xdaq::Application*app) throw (evf::Exception) :

	// call super constructor
			IPCMethod(segmentationMode, nbRawCells, nbRecoCells, nbDqmCells,
					rawCellSize, recoCellSize, dqmCellSize, freeResReq, bu, sm,
					logger, timeout, frb, app), shmBuffer_(0)

{
	initialize(segmentationMode, nbRawCells, nbRecoCells, nbDqmCells,
			rawCellSize, recoCellSize, dqmCellSize);
}

//______________________________________________________________________________
FUResourceTable::~FUResourceTable() {
	clear();
	//workloop cancels used to be here in the previous version
	shmdt( shmBuffer_);
	if (FUShmBuffer::releaseSharedMemory())
		LOG4CPLUS_INFO(log_, "SHARED MEMORY SUCCESSFULLY RELEASED.");
	if (0 != acceptSMDataDiscard_)
		delete[] acceptSMDataDiscard_;
	if (0 != acceptSMDqmDiscard_)
		delete[] acceptSMDqmDiscard_;
}

////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void FUResourceTable::initialize(bool segmentationMode, UInt_t nbRawCells,
		UInt_t nbRecoCells, UInt_t nbDqmCells, UInt_t rawCellSize,
		UInt_t recoCellSize, UInt_t dqmCellSize) throw (evf::Exception) {
	clear();

	shmBuffer_ = FUShmBuffer::createShmBuffer(segmentationMode, nbRawCells,
			nbRecoCells, nbDqmCells, rawCellSize, recoCellSize, dqmCellSize);
	if (0 == shmBuffer_) {
		string msg = "CREATION OF SHARED MEMORY SEGMENT FAILED!";
		LOG4CPLUS_FATAL(log_, msg);
		XCEPT_RAISE(evf::Exception, msg);
	}

	for (UInt_t i = 0; i < nbRawCells_; i++) {
		FUResource* newResource = new FUResource(i, log_, frb_, app_);
		newResource->release(true);
		resources_.push_back(newResource);
		freeResourceIds_.push(i);
	}

	acceptSMDataDiscard_ = new bool[nbRecoCells];
	acceptSMDqmDiscard_ = new int[nbDqmCells];

	resetCounters();
}

//______________________________________________________________________________
bool FUResourceTable::sendData() {
	bool reschedule = true;
	FUShmRecoCell* cell = 0;
	try {
		cell = shmBuffer_->recoCellToRead();
	} catch (evf::Exception& e) {
		rethrowShmBufferException(e, "FUResourceTable:sendData:recoCellToRead");
	}

	if (0 == cell->eventSize()) {
		LOG4CPLUS_INFO(log_, "Don't reschedule sendData workloop.");
		UInt_t cellIndex = cell->index();
		try {
			shmBuffer_->finishReadingRecoCell(cell);
			shmBuffer_->discardRecoCell(cellIndex);
		} catch (evf::Exception& e) {
			rethrowShmBufferException(e,
					"FUResourceTable:sendData:finishReadingRecoCell/discardRecoCell");
		}
		reschedule = false;
	} else {
		try {
			if (cell->type() == 0) {
				UInt_t cellIndex = cell->index();
				UInt_t cellOutModId = cell->outModId();
				UInt_t cellFUProcId = cell->fuProcessId();
				UInt_t cellFUGuid = cell->fuGuid();
				UChar_t* cellPayloadAddr = cell->payloadAddr();
				UInt_t cellEventSize = cell->eventSize();
				UInt_t cellExpectedEPs = cell->nExpectedEPs();
				try {
					shmBuffer_->finishReadingRecoCell(cell);
				} catch (evf::Exception& e) {
					rethrowShmBufferException(e,
							"FUResourceTable:sendData:finishReadingRecoCell");
				}

				lock();
				nbPendingSMDiscards_++;
				unlock();

				sendInitMessage(cellIndex, cellOutModId, cellFUProcId,
						cellFUGuid, cellPayloadAddr, cellEventSize,
						cellExpectedEPs);
			} else if (cell->type() == 1) {
				UInt_t cellIndex = cell->index();
				UInt_t cellRawIndex = cell->rawCellIndex();
				UInt_t cellRunNumber = cell->runNumber();
				UInt_t cellEvtNumber = cell->evtNumber();
				UInt_t cellOutModId = cell->outModId();
				UInt_t cellFUProcId = cell->fuProcessId();
				UInt_t cellFUGuid = cell->fuGuid();
				UChar_t *cellPayloadAddr = cell->payloadAddr();
				UInt_t cellEventSize = cell->eventSize();
				try {
					shmBuffer_->finishReadingRecoCell(cell);
				} catch (evf::Exception& e) {
					rethrowShmBufferException(e,
							"FUResourceTable:sendData:finishReadingRecoCell");
				}

				lock();
				nbPendingSMDiscards_++;
				resources_[cellRawIndex]->incNbSent();
				if (resources_[cellRawIndex]->nbSent() == 1)
					nbSent_++;
				unlock();

				sendDataEvent(cellIndex, cellRunNumber, cellEvtNumber,
						cellOutModId, cellFUProcId, cellFUGuid,
						cellPayloadAddr, cellEventSize);
			} else if (cell->type() == 2) {
				UInt_t cellIndex = cell->index();
				UInt_t cellRawIndex = cell->rawCellIndex();
				//UInt_t   cellRunNumber   = cell->runNumber();
				UInt_t cellEvtNumber = cell->evtNumber();
				UInt_t cellFUProcId = cell->fuProcessId();
				UInt_t cellFUGuid = cell->fuGuid();
				UChar_t *cellPayloadAddr = cell->payloadAddr();
				UInt_t cellEventSize = cell->eventSize();
				try {
					shmBuffer_->finishReadingRecoCell(cell);
				} catch (evf::Exception& e) {
					rethrowShmBufferException(e,
							"FUResourceTable:sendData:recoCellToRead");
				}

				lock();
				nbPendingSMDiscards_++;
				resources_[cellRawIndex]->incNbSent();
				if (resources_[cellRawIndex]->nbSent() == 1) {
					nbSent_++;
					nbSentError_++;
				}
				unlock();

				sendErrorEvent(cellIndex, runNumber_, cellEvtNumber,
						cellFUProcId, cellFUGuid, cellPayloadAddr,
						cellEventSize);
			} else {
				string errmsg =
						"Unknown RecoCell type (neither INIT/DATA/ERROR).";
				XCEPT_RAISE(evf::Exception, errmsg);
			}
		} catch (xcept::Exception& e) {
			LOG4CPLUS_FATAL(
					log_,
					"Failed to send EVENT DATA to StorageManager: "
							<< xcept::stdformat_exception_history(e));
			reschedule = false;
		}
	}

	return reschedule;
}

//______________________________________________________________________________
bool FUResourceTable::sendDataWhileHalting() {
	bool reschedule = true;
	FUShmRecoCell* cell = 0;
	try {
		cell = shmBuffer_->recoCellToRead();
	} catch (evf::Exception& e) {
		rethrowShmBufferException(e,
				"FUResourceTable:sendDataWhileHalting:recoCellToRead");
	}

	if (0 == cell->eventSize()) {
		LOG4CPLUS_INFO(log_, "Don't reschedule sendData workloop.");
		UInt_t cellIndex = cell->index();
		try {
			shmBuffer_->finishReadingRecoCell(cell);
			shmBuffer_->discardRecoCell(cellIndex);
		} catch (evf::Exception& e) {
			rethrowShmBufferException(e,
					"FUResourceTable:sendDataWhileHalting:finishReadingRecoCell/discardRecoCell");
		}
		reschedule = false;
	} else {
		LOG4CPLUS_INFO(log_, "sendData: isHalting, discard recoCell.");
		UInt_t cellIndex = cell->index();
		try {
			shmBuffer_->finishReadingRecoCell(cell);
			shmBuffer_->discardRecoCell(cellIndex);
		} catch (evf::Exception& e) {
			rethrowShmBufferException(e,
					"FUResourceTable:sendDataWhileHalting:finishReadingRecoCell/discardRecoCell");
		}
	}

	return reschedule;
}

//______________________________________________________________________________
bool FUResourceTable::sendDqm() {
	bool reschedule = true;
	FUShmDqmCell* cell = 0;
	// initialize to a value to avoid warnings
	dqm::State_t state = dqm::EMPTY;
	try {
		cell = shmBuffer_->dqmCellToRead();
		state = shmBuffer_->dqmState(cell->index());
	} catch (evf::Exception& e) {
		rethrowShmBufferException(e,
				"FUResourceTable:sendDqm:dqmCellToRead/dqmState");
	}

	if (state == dqm::EMPTY) {
		LOG4CPLUS_INFO(log_, "Don't reschedule sendDqm workloop.");
		std::cout << "shut down dqm workloop " << std::endl;
		UInt_t cellIndex = cell->index();
		try {
			shmBuffer_->finishReadingDqmCell(cell);
			shmBuffer_->discardDqmCell(cellIndex);
		} catch (evf::Exception& e) {
			rethrowShmBufferException(e,
					"FUResourceTable:sendDqm:finishReadingDqmCell/discardDqmCell");
		}
		reschedule = false;
	} else {
		try {
			UInt_t cellIndex = cell->index();
			UInt_t cellRunNumber = cell->runNumber();
			UInt_t cellEvtAtUpdate = cell->evtAtUpdate();
			UInt_t cellFolderId = cell->folderId();
			UInt_t cellFUProcId = cell->fuProcessId();
			UInt_t cellFUGuid = cell->fuGuid();
			UChar_t *cellPayloadAddr = cell->payloadAddr();
			UInt_t cellEventSize = cell->eventSize();
			sendDqmEvent(cellIndex, cellRunNumber, cellEvtAtUpdate,
					cellFolderId, cellFUProcId, cellFUGuid, cellPayloadAddr,
					cellEventSize);
			try {
				shmBuffer_->finishReadingDqmCell(cell);
			} catch (evf::Exception& e) {
				rethrowShmBufferException(e,
						"FUResourceTable:sendDqm:finishReadingDqmCell");
			}
		} catch (xcept::Exception& e) {
			LOG4CPLUS_FATAL(
					log_,
					"Failed to send DQM DATA to StorageManager: "
							<< xcept::stdformat_exception_history(e));
			reschedule = false;
		}
	}

	return reschedule;
}

//______________________________________________________________________________
bool FUResourceTable::sendDqmWhileHalting() {
	bool reschedule = true;
	FUShmDqmCell* cell = 0;
	// initialize to a value to avoid warnings
	dqm::State_t state = dqm::EMPTY;
	try {
		cell = shmBuffer_->dqmCellToRead();
		state = shmBuffer_->dqmState(cell->index());
	} catch (evf::Exception& e) {
		rethrowShmBufferException(e,
				"FUResourceTable:sendDqmWhileHalting:dqmCellToRead/dqmState");
	}

	if (state == dqm::EMPTY) {
		LOG4CPLUS_INFO(log_, "Don't reschedule sendDqm workloop.");
		std::cout << "shut down dqm workloop " << std::endl;
		UInt_t cellIndex = cell->index();
		try {
			shmBuffer_->finishReadingDqmCell(cell);
			shmBuffer_->discardDqmCell(cellIndex);
		} catch (evf::Exception& e) {
			rethrowShmBufferException(e,
					"FUResourceTable:sendDqmWhileHalting:finishReadingDqmCell/discardDqmCell");
		}
		reschedule = false;
	} else {
		UInt_t cellIndex = cell->index();
		try {
			shmBuffer_->finishReadingDqmCell(cell);
			shmBuffer_->discardDqmCell(cellIndex);
		} catch (evf::Exception& e) {
			rethrowShmBufferException(e,
					"FUResourceTable:sendDqmWhileHalting:finishReadingDqmCell/discardDqmCell");
		}
	}

	return reschedule;
}

// common procedure for discard() and discardWhileHalting()
// when the workloop should not be rescheduled
//______________________________________________________________________________
void FUResourceTable::discardNoReschedule() {
	std::cout << " entered shutdown cycle " << std::endl;
	try {
		shmBuffer_->writeRecoEmptyEvent();
	} catch (evf::Exception& e) {
		rethrowShmBufferException(e,
				"FUResourceTable:discardNoReschedule:writeRecoEmptyEvent");
	}
	UInt_t count = 0;
	while (count < 100) {
		std::cout << " shutdown cycle " << shmBuffer_->nClients() << " "
				<< FUShmBuffer::shm_nattch(shmBuffer_->shmid()) << std::endl;
		if (shmBuffer_->nClients() == 0 && FUShmBuffer::shm_nattch(
				shmBuffer_->shmid()) == 1) {
			//isReadyToShutDown_ = true;
			break;
		} else {
			count++;
			std::cout << " shutdown cycle attempt " << count << std::endl;
			LOG4CPLUS_DEBUG(
					log_,
					"FUResourceTable: Wait for all clients to detach,"
							<< " nClients=" << shmBuffer_->nClients()
							<< " nattch=" << FUShmBuffer::shm_nattch(
							shmBuffer_->shmid()) << " (" << count << ")");
			::usleep( shutdownTimeout_);
			if (count * shutdownTimeout_ > 10000000)
				LOG4CPLUS_WARN(
						log_,
						"FUResourceTable:LONG Wait (>10s) for all clients to detach,"
								<< " nClients=" << shmBuffer_->nClients()
								<< " nattch=" << FUShmBuffer::shm_nattch(
								shmBuffer_->shmid()) << " (" << count << ")");

		}
	}
	bool allEmpty = false;
	std::cout << "Checking if all dqm cells are empty " << std::endl;
	while (!allEmpty) {
		UInt_t n = nbDqmCells_;
		allEmpty = true;
		shmBuffer_->lock();
		for (UInt_t i = 0; i < n; i++) {
			// initialize to a value to avoid warnings
			dqm::State_t state = dqm::EMPTY;
			try {
				state = shmBuffer_->dqmState(i);
			} catch (evf::Exception& e) {
				rethrowShmBufferException(e,
						"FUResourceTable:discardNoReschedule:dqmState");
			}
			if (state != dqm::EMPTY)
				allEmpty = false;
		}
		shmBuffer_->unlock();
	}
	std::cout << "Making sure there are no dqm pending discards " << std::endl;
	if (nbPendingSMDqmDiscards_ != 0) {
		LOG4CPLUS_WARN(
				log_,
				"FUResourceTable: pending DQM discards not zero: ="
						<< nbPendingSMDqmDiscards_
						<< " while cells are all empty. This may cause problems at next start ");

	}
	try {
		shmBuffer_->writeDqmEmptyEvent();
	} catch (evf::Exception& e) {
		rethrowShmBufferException(e,
				"FUResourceTable:discardNoReschedule:writeDqmEmptyEvent");
	}
	isReadyToShutDown_ = true; // moved here from within the first while loop to make sure the
	// sendDqm loop has been shut down as well
}

//______________________________________________________________________________
bool FUResourceTable::discard() {
	FUShmRawCell* cell = 0;
	// initialize to a value to avoid warnings
	evt::State_t state = evt::EMPTY;
	try {
		cell = shmBuffer_->rawCellToDiscard();
		state = shmBuffer_->evtState(cell->index());
	} catch (evf::Exception& e) {
		rethrowShmBufferException(e,
				"FUResourceTable:discard:rawCellToRead/evtState");
	}

	bool reschedule = true;
	bool shutDown = (state == evt::STOP);
	bool isLumi = (state == evt::USEDLS);
	UInt_t fuResourceId = cell->fuResourceId();
	UInt_t buResourceId = cell->buResourceId();

	if (state == evt::EMPTY) {
		LOG4CPLUS_ERROR(log_, "WARNING! ATTEMPTING TO DISCARD EMPTY CELL!!!");
		return true;
	}

	if (shutDown) {
		LOG4CPLUS_INFO(log_, "nbClientsToShutDown = " << nbClientsToShutDown_);
		if (nbClientsToShutDown_ > 0)
			--nbClientsToShutDown_;
		if (nbClientsToShutDown_ == 0) {
			LOG4CPLUS_INFO(log_, "Don't reschedule discard-workloop.");
			isActive_ = false;
			reschedule = false;
		}
	}

	try {
		shmBuffer_->discardRawCell(cell);
	} catch (evf::Exception& e) {
		rethrowShmBufferException(e, "FUResourceTable:discard:discardRawCell");
	}
	// UPDATED
	if (isLumi)
		nbEolDiscarded_++;

	if (!shutDown && !isLumi) {
		if (fuResourceId >= nbResources()) {
			LOG4CPLUS_WARN(
					log_,
					"cell " << cell->index() << " in state " << state
							<< " scheduled for discard has no associated FU resource ");
		} else {
			resources_[fuResourceId]->release(true);
			lock();
			freeResourceIds_.push(fuResourceId);
			assert(freeResourceIds_.size() <= resources_.size());
			unlock();

			sendDiscard(buResourceId);
			sendAllocate();
		}
	}

	if (!reschedule) {
		discardNoReschedule();
	}

	return reschedule;
}

//______________________________________________________________________________
bool FUResourceTable::discardWhileHalting(bool sendDiscards) {
	FUShmRawCell* cell = 0;
	// initialize to a value to avoid warnings
	evt::State_t state = evt::EMPTY;
	try {
		cell = shmBuffer_->rawCellToDiscard();
		state = shmBuffer_->evtState(cell->index());
	} catch (evf::Exception& e) {
		rethrowShmBufferException(e,
				"FUResourceTable:discardWhileHalting:rawCellToRead/evtState");
	}

	bool reschedule = true;
	bool shutDown = (state == evt::STOP);
	bool isLumi = (state == evt::USEDLS);
	UInt_t fuResourceId = cell->fuResourceId();
	UInt_t buResourceId = cell->buResourceId();

	if (state == evt::EMPTY) {
		LOG4CPLUS_ERROR(log_, "WARNING! ATTEMPTING TO DISCARD EMPTY CELL!!!");
		return true;
	}

	if (shutDown) {
		LOG4CPLUS_INFO(log_, "nbClientsToShutDown = " << nbClientsToShutDown_);
		if (nbClientsToShutDown_ > 0)
			--nbClientsToShutDown_;
		if (nbClientsToShutDown_ == 0) {
			LOG4CPLUS_INFO(log_, "Don't reschedule discard-workloop.");
			isActive_ = false;
			reschedule = false;
		}
	}

	try {
		shmBuffer_->discardRawCell(cell);
	} catch (evf::Exception& e) {
		rethrowShmBufferException(e,
				"FUResourceTable:discardWhileHalting:discardRawCell");
	}
	// UPDATED
	if (isLumi)
		nbEolDiscarded_++;

	if (!shutDown && !isLumi) {
		if (fuResourceId >= nbResources()) {
			LOG4CPLUS_WARN(
					log_,
					"cell " << cell->index() << " in state " << state
							<< " scheduled for discard has no associated FU resource ");
		} else {
			resources_[fuResourceId]->release(true);
			lock();
			freeResourceIds_.push(fuResourceId);
			assert(freeResourceIds_.size() <= resources_.size());
			unlock();

			/*
			 sendDiscard(buResourceId);
			 sendAllocate();
			 */
			if (sendDiscards)
				sendDiscard(buResourceId);
		}
	}

	if (!reschedule) {
		discardNoReschedule();
	}

	return reschedule;
}

//______________________________________________________________________________
bool FUResourceTable::buildResource(MemRef_t* bufRef) {
	bool eventComplete = false;
	// UPDATED
	bool lastMsg = isLastMessageOfEvent(bufRef);
	I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME *block =
			(I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME*) bufRef->getDataLocation();

	// Check input
	int fuResourceIdCheck = (int) block->fuTransactionId;
	int buResourceIdCheck = (int) block->buResourceId;
	if (fuResourceIdCheck < 0 || buResourceIdCheck < 0) {
		stringstream failureStr;
		failureStr << "Received TAKE message with invalid bu/fu resource id:"
				<< " fuResourceId: " << fuResourceIdCheck << " buResourceId: "
				<< buResourceIdCheck;
		LOG4CPLUS_ERROR(log_, failureStr.str());
		XCEPT_RAISE(evf::Exception, failureStr.str());
	}

	UInt_t fuResourceId = (UInt_t) block->fuTransactionId;
	UInt_t buResourceId = (UInt_t) block->buResourceId;
	FUResource* resource = resources_[fuResourceId];

	// allocate resource
	if (!resource->fatalError() && !resource->isAllocated()) {
		FUShmRawCell* cell = 0;
		try {
			cell = shmBuffer_->rawCellToWrite();
		} catch (evf::Exception& e) {
			rethrowShmBufferException(e,
					"FUResourceTable:buildResource:rawCellToWrite");
		}
		if (cell == 0) {
			bufRef->release();
			return eventComplete;
		}
		resource->allocate(cell);
		timeval now;
		gettimeofday(&now, 0);

		frb_->setRBTimeStamp(
				((uint64_t)(now.tv_sec) << 32) + (uint64_t)(now.tv_usec));

		frb_->setRBEventCount(nbCompleted_);

		if (doCrcCheck_ > 0 && 0 == nbAllocated_ % doCrcCheck_)
			resource->doCrcCheck(true);
		else
			resource->doCrcCheck(false);
	}

#ifdef DEBUG_RES_TAB
	std::cout << "Received frame for resource " << buResourceId << std::endl;
#endif
	// keep building this resource if it is healthy
	if (!resource->fatalError()) {
#ifdef DEBUG_RES_TAB
		std::cout << "No fatal error for  " << buResourceId << ", keep building..."<< std::endl;
#endif
		resource->process(bufRef);
		lock();
		nbErrors_ += resource->nbErrors();
		nbCrcErrors_ += resource->nbCrcErrors();
		unlock();
#ifdef DEBUG_RES_TAB
		std::cout << "Checking if resource is complete " << buResourceId << std::endl;
#endif
		// make resource available for pick-up
		if (resource->isComplete()) {
#ifdef DEBUG_RES_TAB
			std::cout << "@@@@RESOURCE is COMPLETE " << buResourceId << std::endl;
#endif
			lock();
			nbCompleted_++;
			nbPending_--;
			unlock();
			if (doDumpEvents_ > 0 && nbCompleted_ % doDumpEvents_ == 0)
				dumpEvent(resource->shmCell());
			try {
				shmBuffer_->finishWritingRawCell(resource->shmCell());
			} catch (evf::Exception& e) {
				rethrowShmBufferException(e,
						"FUResourceTable:buildResource:finishWritingRawCell");
			}
			eventComplete = true;
		}

	}
	// bad event, release msg, and the whole resource if this was the last one
	if (resource->fatalError()) {
		if (lastMsg) {
			try {
				shmBuffer_->releaseRawCell(resource->shmCell());
			} catch (evf::Exception& e) {
				rethrowShmBufferException(e,
						"FUResourceTable:buildResource:releaseRawCell");
			}
			resource->release(true);
			lock();
			freeResourceIds_.push(fuResourceId);
			nbDiscarded_++;
			nbLost_++;
			nbPending_--;
			unlock();
			bu_->sendDiscard(buResourceId);
			sendAllocate();
		}
		//bufRef->release(); // this should now be safe re: appendToSuperFrag as corrupted blocks will be removed...
	}

	return eventComplete;
}

//______________________________________________________________________________
bool FUResourceTable::discardDataEvent(MemRef_t* bufRef) {
	I2O_FU_DATA_DISCARD_MESSAGE_FRAME *msg;
	msg = (I2O_FU_DATA_DISCARD_MESSAGE_FRAME*) bufRef->getDataLocation();
	UInt_t recoIndex = msg->rbBufferID;

	// Check input
	int recoIndexCheck = (int) msg->rbBufferID;
	if (recoIndexCheck < 0)
		LOG4CPLUS_ERROR(
				log_,
				"Received DISCARD DATA message with invalid recoIndex:"
						<< recoIndexCheck);

	if (acceptSMDataDiscard_[recoIndex]) {
		lock();
		nbPendingSMDiscards_--;
		unlock();
		acceptSMDataDiscard_[recoIndex] = false;

		try {
			shmBuffer_->discardRecoCell(recoIndex);
		} catch (evf::Exception& e) {
			rethrowShmBufferException(e,
					"FUResourceTable:discardDataEvent:discardRecoCell");
		}
		bufRef->release();

	} else {
		LOG4CPLUS_ERROR(log_, "Spurious DATA discard by StorageManager, skip!");
	}

	return true;
}

//______________________________________________________________________________
bool FUResourceTable::discardDataEventWhileHalting(MemRef_t* bufRef) {
	I2O_FU_DATA_DISCARD_MESSAGE_FRAME *msg;
	msg = (I2O_FU_DATA_DISCARD_MESSAGE_FRAME*) bufRef->getDataLocation();
	UInt_t recoIndex = msg->rbBufferID;

	// Check input
	int recoIndexCheck = (int) msg->rbBufferID;
	if (recoIndexCheck < 0)
		LOG4CPLUS_ERROR(
				log_,
				"Received DISCARD DATA message with invalid recoIndex:"
						<< recoIndexCheck);

	if (acceptSMDataDiscard_[recoIndex]) {
		lock();
		nbPendingSMDiscards_--;
		unlock();
		acceptSMDataDiscard_[recoIndex] = false;

	} else {
		LOG4CPLUS_ERROR(log_, "Spurious DATA discard by StorageManager, skip!");
	}

	bufRef->release();
	return false;
}

//______________________________________________________________________________
bool FUResourceTable::discardDqmEvent(MemRef_t* bufRef) {
	I2O_FU_DQM_DISCARD_MESSAGE_FRAME *msg;
	msg = (I2O_FU_DQM_DISCARD_MESSAGE_FRAME*) bufRef->getDataLocation();
	UInt_t dqmIndex = msg->rbBufferID;

	// Check input
	int dqmIndexCheck = (int) msg->rbBufferID;
	if (dqmIndexCheck < 0)
		LOG4CPLUS_ERROR(
				log_,
				"Received DISCARD DQM message with invalid dqmIndex:"
						<< dqmIndexCheck);

	unsigned int ntries = 0;
	try {
		while (shmBuffer_->dqmState(dqmIndex) != dqm::SENT) {
			LOG4CPLUS_WARN(
					log_,
					"DQM discard for cell " << dqmIndex
							<< " which is not yet in SENT state - waiting");
			::usleep(10000);
			if (ntries++ > 10) {
				LOG4CPLUS_ERROR(
						log_,
						"DQM cell " << dqmIndex
								<< " discard timed out while cell still in state "
								<< shmBuffer_->dqmState(dqmIndex));
				bufRef->release();
				return true;
			}
		}
	} catch (evf::Exception& e) {
		rethrowShmBufferException(e, "FUResourceTable:discardDqmEvent:dqmState");
	}
	if (acceptSMDqmDiscard_[dqmIndex] > 0) {
		acceptSMDqmDiscard_[dqmIndex]--;
		if (nbPendingSMDqmDiscards_ > 0) {
			nbPendingSMDqmDiscards_--;
		} else {
			LOG4CPLUS_WARN(
					log_,
					"Spurious??? DQM discard by StorageManager, index "
							<< dqmIndex << " cell state "
							<< shmBuffer_->dqmState(dqmIndex)
							<< " accept flag " << acceptSMDqmDiscard_[dqmIndex]);
		}
		try {
			shmBuffer_->discardDqmCell(dqmIndex);
		} catch (evf::Exception& e) {
			rethrowShmBufferException(e,
					"FUResourceTable:discardDqmEvent:discardDqmCell");
		}
		bufRef->release();

	} else {
		LOG4CPLUS_ERROR(
				log_,
				"Spurious DQM discard for cell " << dqmIndex
						<< " from StorageManager while cell is not accepting discards");
	}

	return true;
}

// concept: discardDqmEventWhileHalting required??
//______________________________________________________________________________
bool FUResourceTable::discardDqmEventWhileHalting(MemRef_t* bufRef) {
	I2O_FU_DQM_DISCARD_MESSAGE_FRAME *msg;
	msg = (I2O_FU_DQM_DISCARD_MESSAGE_FRAME*) bufRef->getDataLocation();
	UInt_t dqmIndex = msg->rbBufferID;

	// Check input
	int dqmIndexCheck = (int) msg->rbBufferID;
	if (dqmIndexCheck < 0)
		LOG4CPLUS_ERROR(
				log_,
				"Received DISCARD DQM message with invalid dqmIndex:"
						<< dqmIndexCheck);

	unsigned int ntries = 0;
	try {
		while (shmBuffer_->dqmState(dqmIndex) != dqm::SENT) {
			LOG4CPLUS_WARN(
					log_,
					"DQM discard for cell " << dqmIndex
							<< " which is not yet in SENT state - waiting");
			::usleep(10000);
			if (ntries++ > 10) {
				LOG4CPLUS_ERROR(
						log_,
						"DQM cell " << dqmIndex
								<< " discard timed out while cell still in state "
								<< shmBuffer_->dqmState(dqmIndex));
				bufRef->release();
				return true;
			}
		}
	} catch (evf::Exception& e) {
		rethrowShmBufferException(e,
				"FUResourceTable:discardDqmEventWhileHalting:dqmState(2)");
	}
	if (acceptSMDqmDiscard_[dqmIndex] > 0) {
		acceptSMDqmDiscard_[dqmIndex]--;
		if (nbPendingSMDqmDiscards_ > 0) {
			nbPendingSMDqmDiscards_--;
		} else {
			try {
				LOG4CPLUS_WARN(
						log_,
						"Spurious??? DQM discard by StorageManager, index "
								<< dqmIndex << " cell state "
								<< shmBuffer_->dqmState(dqmIndex)
								<< " accept flag "
								<< acceptSMDqmDiscard_[dqmIndex]);
			} catch (evf::Exception& e) {
				rethrowShmBufferException(e,
						"FUResourceTable:discardDqmEventWhileHalting:dqmState");
			}
		}

	} else {
		LOG4CPLUS_ERROR(
				log_,
				"Spurious DQM discard for cell " << dqmIndex
						<< " from StorageManager while cell is not accepting discards");
	}

	bufRef->release();
	return false;
}

//______________________________________________________________________________
void FUResourceTable::postEndOfLumiSection(MemRef_t* bufRef) {
	I2O_EVM_END_OF_LUMISECTION_MESSAGE_FRAME
			*msg =
					(I2O_EVM_END_OF_LUMISECTION_MESSAGE_FRAME *) bufRef->getDataLocation();
	//make sure to fill up the shmem so no process will miss it
	// but processes will have to handle duplicates

	// Check input
	int lumiCheck = (int) msg->lumiSection;
	if (lumiCheck < 0)
		LOG4CPLUS_ERROR(log_,
				"Received EOL message with invalid index:" << lumiCheck);

	for (unsigned int i = 0; i < nbRawCells_; i++) {
		// UPDATED
		nbEolPosted_++;
		try {
			shmBuffer_->writeRawLumiSectionEvent(msg->lumiSection);
		} catch (evf::Exception& e) {
			rethrowShmBufferException(e,
					"FUResourceTable:postEndOfLumiSection:writeRawLumiSectionEvent");
		}
	}
}

//______________________________________________________________________________
void FUResourceTable::dropEvent() {
	FUShmRawCell* cell = 0;
	try {
		cell = shmBuffer_->rawCellToRead();
	} catch (evf::Exception& e) {
		rethrowShmBufferException(e, "FUResourceTable:dropEvent:rawCellToRead");
	}
	UInt_t fuResourceId = cell->fuResourceId();
	try {
		shmBuffer_->finishReadingRawCell(cell);
		shmBuffer_->scheduleRawCellForDiscard(fuResourceId);
	} catch (evf::Exception& e) {
		rethrowShmBufferException(e,
				"FUResourceTable:dropEvent:finishReadingRawCell/scheduleRawCellForDiscard");
	}
}

//______________________________________________________________________________
bool FUResourceTable::handleCrashedEP(UInt_t runNumber, pid_t pid) {
	bool retval = false;
	vector < pid_t > pids = cellPrcIds();
	UInt_t iRawCell = pids.size();
	for (UInt_t i = 0; i < pids.size(); i++) {
		if (pid == pids[i]) {
			iRawCell = i;
			break;
		}
	}

	if (iRawCell < pids.size()) {
		try {
			shmBuffer_->writeErrorEventData(runNumber, pid, iRawCell, true);
		} catch (evf::Exception& e) {
			rethrowShmBufferException(e,
					"FUResourceTable:handleCrashedEP:writeErrorEventData");
		}
		retval = true;
	} else
		LOG4CPLUS_WARN(log_,
				"No raw data to send to error stream for process " << pid);
	try {
		shmBuffer_->removeClientPrcId(pid);
	} catch (evf::Exception& e) {
		rethrowShmBufferException(e,
				"FUResourceTable:handleCrashedEP:removeClientPrcId");
	}
	return retval;
}

//______________________________________________________________________________
void FUResourceTable::shutDownClients() {
	nbClientsToShutDown_ = nbClients();
	isReadyToShutDown_ = false;

	if (nbClientsToShutDown_ == 0) {
		LOG4CPLUS_INFO(
				log_,
				"No clients to shut down. Checking if there are raw cells not assigned to any process yet");
		UInt_t n = nbResources();
		try {
			for (UInt_t i = 0; i < n; i++) {
				evt::State_t state = shmBuffer_->evtState(i);
				if (state != evt::EMPTY) {
					LOG4CPLUS_WARN(
							log_,
							"Schedule discard at STOP for orphaned event in state "
									<< state);
					shmBuffer_->scheduleRawCellForDiscardServerSide(i);
				}
			}
			shmBuffer_->scheduleRawEmptyCellForDiscard();
		} catch (evf::Exception& e) {
			rethrowShmBufferException(e,
					"FUResourceTable:shutDownClients:evtState/scheduleRawEmptyCellForDiscard");
		}
	} else {
		// UPDATED
		int checks = 0;
		try {
			while (shmBuffer_->nbRawCellsToWrite() < nbClients() && nbClients()
					!= 0) {
				checks++;
				vector < pid_t > prcids = clientPrcIds();
				for (UInt_t i = 0; i < prcids.size(); i++) {
					pid_t pid = prcids[i];
					int status = kill(pid, 0);
					if (status != 0) {
						LOG4CPLUS_ERROR(log_,
								"EP prc " << pid << " completed with error.");
						handleCrashedEP(runNumber_, pid);
					}
				}

				LOG4CPLUS_WARN(
						log_,
						"no cell to write stop "
								<< shmBuffer_->nbRawCellsToWrite()
								<< " nClients " << nbClients());
				if (checks > 10) {
					string msg = "No Raw Cell to Write STOP messages";
					XCEPT_RAISE(evf::Exception, msg);
				}
				::usleep(500000);
			}
		} catch (evf::Exception& e) {
			rethrowShmBufferException(e,
					"FUResourceTable:shutDownClients:nbRawCellsToWrite");
		}
		nbClientsToShutDown_ = nbClients();
		if (nbClientsToShutDown_ == 0) {
			UInt_t n = nbResources();
			for (UInt_t i = 0; i < n; i++) {
				// initialize to a value to avoid warnings
				evt::State_t state = evt::EMPTY;
				try {
					state = shmBuffer_->evtState(i);
				} catch (evf::Exception& e) {
					rethrowShmBufferException(e,
							"FUResourceTable:shutDownClients:evtState");
				}
				if (state != evt::EMPTY) {
					LOG4CPLUS_WARN(
							log_,
							"Schedule discard at STOP for orphaned event in state "
									<< state);
					try {
						shmBuffer_->setEvtDiscard(i, 1, true);
						shmBuffer_->scheduleRawCellForDiscardServerSide(i);
					} catch (evf::Exception& e) {
						rethrowShmBufferException(e,
								"FUResourceTable:shutDownClients:scheduleRawCellForDiscardServerSide");
					}
				}
			}
			try {
				shmBuffer_->scheduleRawEmptyCellForDiscard();
			} catch (evf::Exception& e) {
				rethrowShmBufferException(e,
						"FUResourceTable:shutDownClients:scheduleRawEmptyCellForDiscard");
			}
		}
		UInt_t n = nbClientsToShutDown_;
		try {
			for (UInt_t i = 0; i < n; ++i)
				shmBuffer_->writeRawEmptyEvent();
		} catch (evf::Exception& e) {
			rethrowShmBufferException(e,
					"FUResourceTable:shutDownClients:writeRawEmptyEvent");
		}
	}
}

//______________________________________________________________________________
void FUResourceTable::clear() {
	for (UInt_t i = 0; i < resources_.size(); i++) {
		resources_[i]->release(true);
		delete resources_[i];
	}
	resources_.clear();
	while (!freeResourceIds_.empty())
		freeResourceIds_.pop();
}

////////////////////////////////////////////////////////////////////////////////
// implementation of private member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void FUResourceTable::resetCounters() {
	if (0 != shmBuffer_) {
		try {
			for (UInt_t i = 0; i < shmBuffer_->nRecoCells(); i++)
				acceptSMDataDiscard_[i] = false;
			for (UInt_t i = 0; i < shmBuffer_->nDqmCells(); i++)
				acceptSMDqmDiscard_[i] = 0;
		} catch (evf::Exception& e) {
			rethrowShmBufferException(e,
					"FUResourceTable:resetCounters:nRecoCells/nDqmCells");
		}
	}

	// UPDATE: reset pending allocate's
	nbAllocated_ = 0;
	nbPending_ = 0;
	nbCompleted_ = 0;
	nbSent_ = 0;
	nbSentError_ = 0;
	nbSentDqm_ = 0;
	nbPendingSMDiscards_ = 0;
	nbPendingSMDqmDiscards_ = 0;
	nbDiscarded_ = 0;
	nbLost_ = 0;
	// UPDATED
	nbEolPosted_ = 0;
	nbEolDiscarded_ = 0;

	nbErrors_ = 0;
	nbCrcErrors_ = 0;
	nbAllocSent_ = 0;

	sumOfSquares_ = 0;
	sumOfSizes_ = 0;
}

//______________________________________________________________________________
UInt_t FUResourceTable::nbClients() const {
	UInt_t result(0);
	try {
		if (0 != shmBuffer_)
			result = shmBuffer_->nClients();
	} catch (evf::Exception& e) {
		rethrowShmBufferException(e, "FUResourceTable:nbClients:nClients");
	}
	return result;
}

//______________________________________________________________________________
vector<pid_t> FUResourceTable::clientPrcIds() const {
	vector < pid_t > result;
	try {
		if (0 != shmBuffer_) {
			UInt_t n = nbClients();
			for (UInt_t i = 0; i < n; i++)
				result.push_back(shmBuffer_->clientPrcId(i));
		}
	} catch (evf::Exception& e) {
		rethrowShmBufferException(e,
				"FUResourceTable:clientPrcIds:clientPrcIds");
	}
	return result;
}

//______________________________________________________________________________
string FUResourceTable::clientPrcIdsAsString() const {
	stringstream ss;
	try {
		if (0 != shmBuffer_) {
			UInt_t n = nbClients();
			for (UInt_t i = 0; i < n; i++) {
				if (i > 0)
					ss << ",";
				ss << shmBuffer_->clientPrcId(i);
			}
		}
	} catch (evf::Exception& e) {
		rethrowShmBufferException(e,
				"FUResourceTable:clientPrcIdsAsString:clientPrcId");
	}
	return ss.str();
}

//______________________________________________________________________________
vector<string> FUResourceTable::cellStates() const {
	vector < string > result;
	if (0 != shmBuffer_) {
		UInt_t n = nbResources();
		shmBuffer_->lock();
		try {
			for (UInt_t i = 0; i < n; i++) {
				evt::State_t state = shmBuffer_->evtState(i);
				if (state == evt::EMPTY)
					result.push_back("EMPTY");
				else if (state == evt::STOP)
					result.push_back("STOP");
				else if (state == evt::LUMISECTION)
					result.push_back("LUMISECTION");
				// UPDATED
				else if (state == evt::USEDLS)
					result.push_back("USEDLS");
				else if (state == evt::RAWWRITING)
					result.push_back("RAWWRITING");
				else if (state == evt::RAWWRITTEN)
					result.push_back("RAWWRITTEN");
				else if (state == evt::RAWREADING)
					result.push_back("RAWREADING");
				else if (state == evt::RAWREAD)
					result.push_back("RAWREAD");
				else if (state == evt::PROCESSING)
					result.push_back("PROCESSING");
				else if (state == evt::PROCESSED)
					result.push_back("PROCESSED");
				else if (state == evt::RECOWRITING)
					result.push_back("RECOWRITING");
				else if (state == evt::RECOWRITTEN)
					result.push_back("RECOWRITTEN");
				else if (state == evt::SENDING)
					result.push_back("SENDING");
				else if (state == evt::SENT)
					result.push_back("SENT");
				else if (state == evt::DISCARDING)
					result.push_back("DISCARDING");
			}
		} catch (evf::Exception& e) {
			rethrowShmBufferException(e, "FUResourceTable:cellStates:evtState");
		}
		shmBuffer_->unlock();
	}
	return result;
}

vector<string> FUResourceTable::dqmCellStates() const {
	vector < string > result;
	if (0 != shmBuffer_) {
		UInt_t n = nbDqmCells_;
		shmBuffer_->lock();
		try {
			for (UInt_t i = 0; i < n; i++) {
				dqm::State_t state = shmBuffer_->dqmState(i);
				if (state == dqm::EMPTY)
					result.push_back("EMPTY");
				else if (state == dqm::WRITING)
					result.push_back("WRITING");
				else if (state == dqm::WRITTEN)
					result.push_back("WRITTEN");
				else if (state == dqm::SENDING)
					result.push_back("SENDING");
				else if (state == dqm::SENT)
					result.push_back("SENT");
				else if (state == dqm::DISCARDING)
					result.push_back("DISCARDING");
			}
		} catch (evf::Exception& e) {
			rethrowShmBufferException(e,
					"FUResourceTable:dqmCellStates:dqmState");
		}
		shmBuffer_->unlock();
	}
	return result;
}

//______________________________________________________________________________
vector<UInt_t> FUResourceTable::cellEvtNumbers() const {
	vector < UInt_t > result;
	if (0 != shmBuffer_) {
		UInt_t n = nbResources();
		shmBuffer_->lock();
		try {
			for (UInt_t i = 0; i < n; i++)
				result.push_back(shmBuffer_->evtNumber(i));
		} catch (evf::Exception& e) {
			rethrowShmBufferException(e,
					"FUResourceTable:cellEvtNumbers:evtNumber");
		}
		shmBuffer_->unlock();
	}
	return result;
}

//______________________________________________________________________________
vector<pid_t> FUResourceTable::cellPrcIds() const {
	vector < pid_t > result;
	if (0 != shmBuffer_) {
		UInt_t n = nbResources();
		shmBuffer_->lock();
		try {
			for (UInt_t i = 0; i < n; i++)
				result.push_back(shmBuffer_->evtPrcId(i));
		} catch (evf::Exception& e) {
			rethrowShmBufferException(e, "FUResourceTable:cellPrcIds:evtPrcId");
		}
		shmBuffer_->unlock();
	}
	return result;
}

//______________________________________________________________________________
vector<time_t> FUResourceTable::cellTimeStamps() const {
	vector < time_t > result;
	try {
		if (0 != shmBuffer_) {
			UInt_t n = nbResources();
			shmBuffer_->lock();
			for (UInt_t i = 0; i < n; i++)
				result.push_back(shmBuffer_->evtTimeStamp(i));
			shmBuffer_->unlock();
		}
	} catch (evf::Exception& e) {
		rethrowShmBufferException(e,
				"FUResourceTable:cellTimeStamps:evtTimeStamp");
	}
	return result;
}

////////////////////////////////////////////////////////////////////////////////
// implementation of private member functions
////////////////////////////////////////////////////////////////////////////////

void FUResourceTable::lastResort() {
	try {
		std::cout << "lastResort: " << shmBuffer_->nbRawCellsToRead()
				<< " more rawcells to read " << std::endl;
		while (shmBuffer_->nbRawCellsToRead() != 0) {
			FUShmRawCell* newCell = shmBuffer_->rawCellToRead();
			std::cout << "lastResort: " << shmBuffer_->nbRawCellsToRead()
					<< std::endl;
			// UPDATED
			shmBuffer_->scheduleRawCellForDiscardServerSide(newCell->index());

			std::cout << "lastResort: schedule raw cell for discard "
					<< newCell->index() << std::endl;
		}
		//trigger the shutdown (again?)
		shmBuffer_->scheduleRawEmptyCellForDiscard();
	} catch (evf::Exception& e) {
		rethrowShmBufferException(
				e,
				"FUResourceTable:lastResort:nbRawCellsToRead/scheduleRawCellForDiscardServerSide");
	}
}

void FUResourceTable::resetIPC() {
	if (shmBuffer_ != 0) {
		shmBuffer_->reset();
		LOG4CPLUS_INFO(log_, "ShmBuffer was reset!");
	}
}

void FUResourceTable::rethrowShmBufferException(evf::Exception& e, string where) const
		throw (evf::Exception) {
	stringstream details;
	vector < string > dataStates = cellStates();
	vector < string > dqmStates = dqmCellStates();
	details << "Exception raised: " << e.what() << " (in module: "
			<< e.module() << " in function: " << e.function() << " at line: "
			<< e.line() << ")";
	details << "   Dumping cell state...   ";
	details << "data cells --> ";
	for (unsigned int i = 0; i < dataStates.size(); i++)
		details << dataStates[i] << " ";
	details << "dqm cells --> ";
	for (unsigned int i = 0; i < dqmStates.size(); i++)
		details << dqmStates[i] << " ";
	details << " ... originated in: " << where;
	XCEPT_RETHROW(evf::Exception, details.str(), e);
}
