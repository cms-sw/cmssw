////////////////////////////////////////////////////////////////////////////////
//
//							>> UNDER DEVELOPMENT <<
//
// FUResourceQueue
// ---------------
//
// Main class for Message Queue interprocess communication.
//
//                   28/10/2011 Andrei Spataru <andrei.cristian.spataru@cern.ch>
////////////////////////////////////////////////////////////////////////////////

#include "EventFilter/ResourceBroker/interface/FUResourceQueue.h"
#include "EventFilter/ResourceBroker/interface/RecoMsgBuf.h"
#include "EventFilter/ResourceBroker/interface/DQMMsgBuf.h"
#include "EventFilter/ResourceBroker/interface/msq_constants.h"
#include "EventFilter/ShmBuffer/interface/FUShmDqmCell.h"

#include "EvffedFillerRB.h"
#include "interface/evb/i2oEVBMsgs.h"
#include "xcept/tools.h"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <unistd.h>

using std::cout;
using std::endl;
using std::string;
using std::stringstream;
using std::vector;
using namespace evf;

////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
FUResourceQueue::FUResourceQueue(bool segmentationMode, UInt_t nbRawCells,
		UInt_t nbRecoCells, UInt_t nbDqmCells, UInt_t rawCellSize,
		UInt_t recoCellSize, UInt_t dqmCellSize, int freeResReq, BUProxy *bu,
		SMProxy *sm, log4cplus::Logger logger, unsigned int timeout,
		EvffedFillerRB *frb, xdaq::Application*app) throw (evf::Exception) :
			IPCMethod(segmentationMode, nbRawCells, nbRecoCells, nbDqmCells,
					rawCellSize, recoCellSize, dqmCellSize, freeResReq, bu, sm,
					logger, timeout, frb, app), msq_(99) {
	//improve fix UInt_t and msq_(99)

	initialize(segmentationMode, nbRawCells, nbRecoCells, nbDqmCells,
			rawCellSize, recoCellSize, dqmCellSize);

}

//______________________________________________________________________________
FUResourceQueue::~FUResourceQueue() {
	clear();

	// disconnect from queue
	if (msq_.disconnect() == 0)
		LOG4CPLUS_INFO(log_, "MESSAGE QUEUE SUCCESSFULLY RELEASED.");

	// needed??
	/*
	 if (0 != acceptSMDataDiscard_)
	 delete[] acceptSMDataDiscard_;
	 if (0 != acceptSMDqmDiscard_)
	 delete[] acceptSMDqmDiscard_;
	 */
}

////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void FUResourceQueue::initialize(bool segmentationMode, UInt_t nbRawCells,
		UInt_t nbRecoCells, UInt_t nbDqmCells, UInt_t rawCellSize,
		UInt_t recoCellSize, UInt_t dqmCellSize) throw (evf::Exception) {

	rawCellSize_ = rawCellSize;
	recoCellSize_ = recoCellSize;
	dqmCellSize_ = dqmCellSize;

	clear();

	if (0 == &msq_ || 0 == msq_.id()) {
		string msg = "CREATION OF MESSAGE QUEUE FAILED!";
		LOG4CPLUS_FATAL(log_, msg);
		XCEPT_RAISE(evf::Exception, msg);
	}

	cache_ = RawCache::getInstance();
	cache_->initialise(nbRawCells, rawCellSize);

	// SEC need cap on max resources
	for (UInt_t i = 0; i < nbRawCells_; i++) {
		FUResource* newResource = new FUResource(i, log_, frb_, app_);
		newResource->release(false);
		resources_.push_back(newResource);
		freeResourceIds_.push(i);
	}

	//acceptSMDataDiscard_ = new bool[nbRecoCells];
	//acceptSMDqmDiscard_ = new int[nbDqmCells];

	resetCounters();
}

// work loop to send data events to storage manager
//______________________________________________________________________________
bool FUResourceQueue::sendData() {
	bool reschedule = true;

	//FUShmRecoCell* cell = shmBuffer_->recoCellToRead();
	RecoMsgBuf recoMsg(recoCellSize_, RECO_MESSAGE_TYPE);

	bool rcvSuccess = msq_.rcvQuiet(recoMsg);
	if (!rcvSuccess) {
		cout << "RCV failed!" << endl;
		::sleep(5);
		return reschedule;
	}

	FUShmRecoCell* cell = recoMsg.recoCell();

	// event size 0 -> stop

	if (0 == cell->eventSize()) {
		LOG4CPLUS_INFO(log_, "Don't reschedule sendData workloop.");
		//UInt_t cellIndex = cell->index();
		/*
		 shmBuffer_->finishReadingRecoCell(cell);
		 shmBuffer_->discardRecoCell(cellIndex);
		 */
		reschedule = false;

		// halting
	} else if (/*isHalting_*/false) {
		LOG4CPLUS_INFO(log_, "sendData: isHalting, discard recoCell.");
		//UInt_t cellIndex = cell->index();
		/*
		 shmBuffer_->finishReadingRecoCell(cell);
		 shmBuffer_->discardRecoCell(cellIndex);
		 */

	} else {
		try {
			//init message
			if (cell->type() == 0) {
				UInt_t cellIndex = cell->index();
				UInt_t cellOutModId = cell->outModId();
				UInt_t cellFUProcId = cell->fuProcessId();
				UInt_t cellFUGuid = cell->fuGuid();
				UChar_t* cellPayloadAddr = cell->payloadAddr();
				UInt_t cellEventSize = cell->eventSize();
				UInt_t cellExpectedEPs = cell->nExpectedEPs();
				//shmBuffer_->finishReadingRecoCell(cell);

				lock();
				nbPendingSMDiscards_++;
				unlock();

				sendInitMessage(cellIndex, cellOutModId, cellFUProcId,
						cellFUGuid, cellPayloadAddr, cellEventSize,
						cellExpectedEPs);

				//
				// DATA event message
				//
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
				//shmBuffer_->finishReadingRecoCell(cell);
				lock();
				nbPendingSMDiscards_++;
				resources_[cellRawIndex]->incNbSent();
				if (resources_[cellRawIndex]->nbSent() == 1)
					nbSent_++;
				unlock();

				sendDataEvent(cellIndex, cellRunNumber, cellEvtNumber,
						cellOutModId, cellFUProcId, cellFUGuid,
						cellPayloadAddr, cellEventSize);
				//
				// ERROR event message
				//
			} else if (cell->type() == 2) {
				UInt_t cellIndex = cell->index();
				UInt_t cellRawIndex = cell->rawCellIndex();
				//UInt_t   cellRunNumber   = cell->runNumber();
				UInt_t cellEvtNumber = cell->evtNumber();
				UInt_t cellFUProcId = cell->fuProcessId();
				UInt_t cellFUGuid = cell->fuGuid();
				UChar_t *cellPayloadAddr = cell->payloadAddr();
				UInt_t cellEventSize = cell->eventSize();
				//shmBuffer_->finishReadingRecoCell(cell);

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
bool FUResourceQueue::sendDataWhileHalting(/*toolbox::task::WorkLoop*  wl */) {
	bool reschedule = true;

	//FUShmRecoCell* cell = shmBuffer_->recoCellToRead();
	RecoMsgBuf recoMsg(recoCellSize_, RECO_MESSAGE_TYPE);

	bool rcvSuccess = msq_.rcvQuiet(recoMsg);
	if (!rcvSuccess) {
		cout << "RCV failed!" << endl;
		::sleep(5);
		return reschedule;
	}

	FUShmRecoCell* cell = recoMsg.recoCell();

	// event size 0 -> stop

	if (0 == cell->eventSize()) {
		LOG4CPLUS_INFO(log_, "Don't reschedule sendData workloop.");
		//UInt_t cellIndex = cell->index();
		/*
		 shmBuffer_->finishReadingRecoCell(cell);
		 shmBuffer_->discardRecoCell(cellIndex);
		 */
		reschedule = false;

		// halting
	} else if (/*isHalting_*/true) {
		LOG4CPLUS_INFO(log_, "sendData: isHalting, discard recoCell.");
		//UInt_t cellIndex = cell->index();
		/*
		 shmBuffer_->finishReadingRecoCell(cell);
		 shmBuffer_->discardRecoCell(cellIndex);
		 */

	} else {
		try {
			//init message
			if (cell->type() == 0) {
				UInt_t cellIndex = cell->index();
				UInt_t cellOutModId = cell->outModId();
				UInt_t cellFUProcId = cell->fuProcessId();
				UInt_t cellFUGuid = cell->fuGuid();
				UChar_t* cellPayloadAddr = cell->payloadAddr();
				UInt_t cellEventSize = cell->eventSize();
				UInt_t cellExpectedEPs = cell->nExpectedEPs();
				//shmBuffer_->finishReadingRecoCell(cell);

				lock();
				nbPendingSMDiscards_++;
				unlock();

				sendInitMessage(cellIndex, cellOutModId, cellFUProcId,
						cellFUGuid, cellPayloadAddr, cellEventSize,
						cellExpectedEPs);

				//
				// DATA event message
				//
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

				//shmBuffer_->finishReadingRecoCell(cell);
				lock();
				nbPendingSMDiscards_++;
				resources_[cellRawIndex]->incNbSent();
				if (resources_[cellRawIndex]->nbSent() == 1)
					nbSent_++;
				unlock();

				sendDataEvent(cellIndex, cellRunNumber, cellEvtNumber,
						cellOutModId, cellFUProcId, cellFUGuid,
						cellPayloadAddr, cellEventSize);
				//
				// ERROR event message
				//
			} else if (cell->type() == 2) {
				UInt_t cellIndex = cell->index();
				UInt_t cellRawIndex = cell->rawCellIndex();
				//UInt_t   cellRunNumber   = cell->runNumber();
				UInt_t cellEvtNumber = cell->evtNumber();
				UInt_t cellFUProcId = cell->fuProcessId();
				UInt_t cellFUGuid = cell->fuGuid();
				UChar_t *cellPayloadAddr = cell->payloadAddr();
				UInt_t cellEventSize = cell->eventSize();
				//shmBuffer_->finishReadingRecoCell(cell);

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

// work loop to send dqm events to storage manager
//______________________________________________________________________________
bool FUResourceQueue::sendDqm() {
	bool reschedule = true;

	//FUShmDqmCell* cell = shmBuffer_->dqmCellToRead();
	//dqm::State_t state = shmBuffer_->dqmState(cell->index());

	DQMMsgBuf dqmMsg(dqmCellSize_, DQM_MESSAGE_TYPE);

	bool rcvSuccess = msq_.rcvQuiet(dqmMsg);
	if (!rcvSuccess) {
		cout << "RCV failed!" << endl;
		::sleep(5);
		return reschedule;
	}
	FUShmDqmCell* cell = dqmMsg.dqmCell();

	// concept add stop messages (there is no more "state")
	//if (state == dqm::EMPTY) {
	if (false) {
		LOG4CPLUS_WARN(log_, "Don't reschedule sendDqm workloop.");
		cout << "shut down dqm workloop " << endl;
		//UInt_t cellIndex = cell->index();
		/*
		 shmBuffer_->finishReadingDqmCell(cell);
		 shmBuffer_->discardDqmCell(cellIndex);
		 */
		reschedule = false;
	} else if (/*isHalting_*/false) {
		//UInt_t cellIndex = cell->index();
		/*
		 shmBuffer_->finishReadingDqmCell(cell);
		 shmBuffer_->discardDqmCell(cellIndex);
		 */
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
			//shmBuffer_->finishReadingDqmCell(cell);
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
bool FUResourceQueue::sendDqmWhileHalting(/*toolbox::task::WorkLoop* wl*/) {
	bool reschedule = true;

	//FUShmDqmCell* cell = shmBuffer_->dqmCellToRead();
	//dqm::State_t state = shmBuffer_->dqmState(cell->index());

	DQMMsgBuf dqmMsg(dqmCellSize_, DQM_MESSAGE_TYPE);

	bool rcvSuccess = msq_.rcvQuiet(dqmMsg);
	if (!rcvSuccess) {
		cout << "RCV failed!" << endl;
		::sleep(5);
		return reschedule;
	}
	FUShmDqmCell* cell = dqmMsg.dqmCell();

	// concept add stop messages (there is no more "state")
	//if (state == dqm::EMPTY) {
	if (false) {
		LOG4CPLUS_WARN(log_, "Don't reschedule sendDqm workloop.");
		cout << "shut down dqm workloop " << endl;
		//UInt_t cellIndex = cell->index();
		/*
		 shmBuffer_->finishReadingDqmCell(cell);
		 shmBuffer_->discardDqmCell(cellIndex);
		 */
		reschedule = false;
	} else if (/*isHalting_*/true) {
		//UInt_t cellIndex = cell->index();
		/*
		 shmBuffer_->finishReadingDqmCell(cell);
		 shmBuffer_->discardDqmCell(cellIndex);
		 */
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
			//shmBuffer_->finishReadingDqmCell(cell);
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

// work loop to discard events to builder unit
//______________________________________________________________________________
bool FUResourceQueue::discard(/*toolbox::task::WorkLoop* wl*/) {

	bool reschedule = true;

	/*
	 * DISCARDING raw msg buffers
	 */
	MsgBuf discardRaw(2 * sizeof(unsigned int), DISCARD_RAW_MESSAGE_TYPE);
	bool rcvSuccess = msq_.rcvQuiet(discardRaw);

	if (!rcvSuccess) {
		cout << "RCV failed!" << endl;
		::sleep(5);
		return reschedule;
	}

	unsigned int* pBuID = (unsigned int*) discardRaw->mtext;
	unsigned int* pFuID = (unsigned int*) (discardRaw->mtext
			+ sizeof(unsigned int));

	unsigned int buResourceId = *pBuID;
	unsigned int fuResourceId = *pFuID;

	cout << "Discard received for buResourceID: " << buResourceId
			<< " fuResourceID " << fuResourceId << endl << endl;

	//FUShmRawCell* cell = shmBuffer_->rawCellToDiscard();
	//evt::State_t state = shmBuffer_->evtState(cell->index());

	/*
	 bool shutDown = (state == evt::STOP);
	 bool isLumi = (state == evt::LUMISECTION);
	 */
	//UInt_t fuResourceId = cell->fuResourceId();
	//UInt_t buResourceId = cell->buResourceId();

	//  cout << "discard loop, state, shutDown, isLumi " << state << " "
	//	    << shutDown << " " << isLumi << endl;
	//  cout << "resource ids " << fuResourceId << " " << buResourceId << endl;

	/*
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
	 */

	//shmBuffer_->discardRawCell(cell);

	//if (!shutDown && !isLumi) {
	if (true) {
		// (false = no shmdt)
		resources_[fuResourceId]->release(false);
		// also release space in RawCache
		RawCache::getInstance()->releaseMsg(fuResourceId);

		lock();
		freeResourceIds_.push(fuResourceId);
		assert(freeResourceIds_.size() <= resources_.size());
		unlock();

		if (/*!isHalting_*/true) {
			sendDiscard(buResourceId);
			if (/*!isStopping_*/true)
				sendAllocate();
		}
	}

	// concept shutdown cycle
	/*
	 if (!reschedule) {
	 cout << " entered shutdown cycle " << endl;
	 shmBuffer_->writeRecoEmptyEvent();
	 UInt_t count = 0;
	 while (count < 100) {
	 cout << " shutdown cycle " << shmBuffer_->nClients() << " "
	 << FUShmBuffer::shm_nattch(shmBuffer_->shmid())
	 << endl;
	 if (shmBuffer_->nClients() == 0 && FUShmBuffer::shm_nattch(
	 shmBuffer_->shmid()) == 1) {
	 //	isReadyToShutDown_ = true;
	 break;
	 } else {
	 count++;
	 cout << " shutdown cycle attempt " << count << endl;
	 LOG4CPLUS_DEBUG(
	 log_,
	 "FUResourceTable: Wait for all clients to detach,"
	 << " nClients=" << shmBuffer_->nClients()
	 << " nattch=" << FUShmBuffer::shm_nattch(
	 shmBuffer_->shmid()) << " (" << count << ")");
	 ::usleep(shutdownTimeout_);
	 if (count * shutdownTimeout_ > 10000000)
	 LOG4CPLUS_WARN(
	 log_,
	 "FUResourceTable:LONG Wait (>10s) for all clients to detach,"
	 << " nClients=" << shmBuffer_->nClients()
	 << " nattch=" << FUShmBuffer::shm_nattch(
	 shmBuffer_->shmid()) << " (" << count
	 << ")");

	 }
	 }
	 bool allEmpty = false;
	 cout << "Checking if all dqm cells are empty " << endl;
	 while (!allEmpty) {
	 UInt_t n = nbDqmCells_;
	 allEmpty = true;
	 shmBuffer_->lock();
	 for (UInt_t i = 0; i < n; i++) {
	 dqm::State_t state = shmBuffer_->dqmState(i);
	 if (state != dqm::EMPTY)
	 allEmpty = false;
	 }
	 shmBuffer_->unlock();
	 }
	 cout << "Making sure there are no dqm pending discards "
	 << endl;
	 if (nbPendingSMDqmDiscards_ != 0) {
	 LOG4CPLUS_WARN(
	 log_,
	 "FUResourceTable: pending DQM discards not zero: ="
	 << nbPendingSMDqmDiscards_
	 << " while cells are all empty. This may cause problems at next start ");

	 }
	 shmBuffer_->writeDqmEmptyEvent();
	 isReadyToShutDown_ = true; // moved here from within the first while loop to make sure the
	 // sendDqm loop has been shut down as well
	 }
	 */

	return reschedule;
}

//______________________________________________________________________________
bool FUResourceQueue::discardWhileHalting(bool sendDiscards) {

	bool reschedule = true;

	/*
	 * DISCARDING raw msg buffers
	 */
	MsgBuf discardRaw(2 * sizeof(unsigned int), DISCARD_RAW_MESSAGE_TYPE);
	bool rcvSuccess = msq_.rcvQuiet(discardRaw);

	if (!rcvSuccess) {
		cout << "RCV failed!" << endl;
		::sleep(5);
		return reschedule;
	}

	unsigned int* pBuID = (unsigned int*) discardRaw->mtext;
	unsigned int* pFuID = (unsigned int*) (discardRaw->mtext
			+ sizeof(unsigned int));

	unsigned int buResourceId = *pBuID;
	unsigned int fuResourceId = *pFuID;

	cout << "Discard received for buResourceID: " << buResourceId
			<< " fuResourceID " << fuResourceId << endl << endl;

	//FUShmRawCell* cell = shmBuffer_->rawCellToDiscard();
	//evt::State_t state = shmBuffer_->evtState(cell->index());

	/*
	 bool shutDown = (state == evt::STOP);
	 bool isLumi = (state == evt::LUMISECTION);
	 */
	//UInt_t fuResourceId = cell->fuResourceId();
	//UInt_t buResourceId = cell->buResourceId();

	//  cout << "discard loop, state, shutDown, isLumi " << state << " "
	//	    << shutDown << " " << isLumi << endl;
	//  cout << "resource ids " << fuResourceId << " " << buResourceId << endl;

	/*
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
	 */

	//shmBuffer_->discardRawCell(cell);

	//if (!shutDown && !isLumi) {
	if (true) {
		// (false = no shmdt)
		resources_[fuResourceId]->release(false);
		// also release space in RawCache
		RawCache::getInstance()->releaseMsg(fuResourceId);

		lock();
		freeResourceIds_.push(fuResourceId);
		assert(freeResourceIds_.size() <= resources_.size());
		unlock();

		if (/*!isHalting_*/false) {
			sendDiscard(buResourceId);
			if (/*!isStopping_*/false)
				sendAllocate();
		}
	}

	// concept shutdown cycle
	/*
	 if (!reschedule) {
	 cout << " entered shutdown cycle " << endl;
	 shmBuffer_->writeRecoEmptyEvent();
	 UInt_t count = 0;
	 while (count < 100) {
	 cout << " shutdown cycle " << shmBuffer_->nClients() << " "
	 << FUShmBuffer::shm_nattch(shmBuffer_->shmid())
	 << endl;
	 if (shmBuffer_->nClients() == 0 && FUShmBuffer::shm_nattch(
	 shmBuffer_->shmid()) == 1) {
	 //	isReadyToShutDown_ = true;
	 break;
	 } else {
	 count++;
	 cout << " shutdown cycle attempt " << count << endl;
	 LOG4CPLUS_DEBUG(
	 log_,
	 "FUResourceTable: Wait for all clients to detach,"
	 << " nClients=" << shmBuffer_->nClients()
	 << " nattch=" << FUShmBuffer::shm_nattch(
	 shmBuffer_->shmid()) << " (" << count << ")");
	 ::usleep(shutdownTimeout_);
	 if (count * shutdownTimeout_ > 10000000)
	 LOG4CPLUS_WARN(
	 log_,
	 "FUResourceTable:LONG Wait (>10s) for all clients to detach,"
	 << " nClients=" << shmBuffer_->nClients()
	 << " nattch=" << FUShmBuffer::shm_nattch(
	 shmBuffer_->shmid()) << " (" << count
	 << ")");

	 }
	 }
	 bool allEmpty = false;
	 cout << "Checking if all dqm cells are empty " << endl;
	 while (!allEmpty) {
	 UInt_t n = nbDqmCells_;
	 allEmpty = true;
	 shmBuffer_->lock();
	 for (UInt_t i = 0; i < n; i++) {
	 dqm::State_t state = shmBuffer_->dqmState(i);
	 if (state != dqm::EMPTY)
	 allEmpty = false;
	 }
	 shmBuffer_->unlock();
	 }
	 cout << "Making sure there are no dqm pending discards "
	 << endl;
	 if (nbPendingSMDqmDiscards_ != 0) {
	 LOG4CPLUS_WARN(
	 log_,
	 "FUResourceTable: pending DQM discards not zero: ="
	 << nbPendingSMDqmDiscards_
	 << " while cells are all empty. This may cause problems at next start ");

	 }
	 shmBuffer_->writeDqmEmptyEvent();
	 isReadyToShutDown_ = true; // moved here from within the first while loop to make sure the
	 // sendDqm loop has been shut down as well
	 }
	 */

	return reschedule;
}

// process buffer received via I2O_FU_TAKE message
//______________________________________________________________________________

// BUILD RESOURCE (RAW CELL WRITING)

bool FUResourceQueue::buildResource(MemRef_t* bufRef) {

	bool eventComplete = false;

	I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME *block =
			(I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME*) bufRef->getDataLocation();

	UInt_t fuResourceId = (UInt_t) block->fuTransactionId;
	UInt_t buResourceId = (UInt_t) block->buResourceId;

	FUResource* resource = resources_[fuResourceId];

	RawMsgBuf* currentMessageToWrite = cache_->getMsgToWrite();

	if (!resource->fatalError() && !resource->isAllocated()) {
		FUShmRawCell* rawCell = currentMessageToWrite->rawCell();
		rawCell->initialize(fuResourceId);
		resource->allocate(rawCell);

		timeval now;
		gettimeofday(&now, 0);

		frb_->setRBTimeStamp(
				((uint64_t) (now.tv_sec) << 32) + (uint64_t) (now.tv_usec));

		frb_->setRBEventCount(nbCompleted_);

		if (doCrcCheck_ > 0 && 0 == nbAllocated_ % doCrcCheck_)
			resource->doCrcCheck(true);
		else
			resource->doCrcCheck(false);
	}

	// keep building this resource if it is healthy
	if (!resource->fatalError()) {
		resource->process(bufRef);

		lock();
		nbErrors_ += resource->nbErrors();
		nbCrcErrors_ += resource->nbCrcErrors();
		unlock();

		// make resource available for pick-up
		if (resource->isComplete()) {
			lock();
			nbCompleted_++;
			nbPending_--;
			unlock();
			/*
			 cout << "POSTING to q: msqid = " << msq_.id()
			 << ", message buf is allocated for:  "
			 << currentMessageToWrite->msize()
			 << "... Actually copied: "
			 << currentMessageToWrite->usedSize() << " fuResourceID = "
			 << currentMessageToWrite->rawCell()->fuResourceId()
			 << " buResourceID = "
			 << currentMessageToWrite->rawCell()->buResourceId() << endl;
			 */

			//msq_.post(*currentMessageToWrite);

			try {
				msq_.postLength(*currentMessageToWrite,
						currentMessageToWrite->usedSize());
			} catch (...) {
				string errmsg = "Failed to post message to Queue!";
				LOG4CPLUS_FATAL(log_, errmsg);
				XCEPT_RAISE(evf::Exception, errmsg);

			}

			// CURRENT RECEIVERS
			/*
			 vector<int> receivers = msq_.getReceivers();
			 cout << "--Receiving processes: ";
			 for (unsigned int i = 0; i < receivers.size(); ++i)
			 cout << i << " " << receivers[i];
			 cout << endl;
			 */

			eventComplete = true;
		}

	}
	// bad event, release msg, and the whole resource if this was the last one
	if (resource->fatalError()) {
		bool lastMsg = isLastMessageOfEvent(bufRef);
		if (lastMsg) {
			resource->release(false);
			lock();
			freeResourceIds_.push(fuResourceId);
			nbDiscarded_++;
			nbLost_++;
			nbPending_--;
			unlock();
			bu_->sendDiscard(buResourceId);
			sendAllocate();
		}
		bufRef->release(); // this should now be safe re: appendToSuperFrag as corrupted blocks will be removed...
	}

	return eventComplete;
}

//concept discardDataEvent still required
// process buffer received via I2O_SM_DATA_DISCARD message
//______________________________________________________________________________
bool FUResourceQueue::discardDataEvent(MemRef_t* bufRef) {

	/*
	 I2O_FU_DATA_DISCARD_MESSAGE_FRAME *msg;
	 msg = (I2O_FU_DATA_DISCARD_MESSAGE_FRAME*) bufRef->getDataLocation();
	 UInt_t recoIndex = msg->rbBufferID;

	 if (acceptSMDataDiscard_[recoIndex]) {
	 lock();
	 nbPendingSMDiscards_--;
	 unlock();
	 acceptSMDataDiscard_[recoIndex] = false;

	 if (!isHalting_) {
	 shmBuffer_->discardRecoCell(recoIndex);
	 bufRef->release();
	 }
	 } else {
	 LOG4CPLUS_ERROR(log_, "Spurious DATA discard by StorageManager, skip!");
	 }

	 if (isHalting_) {
	 bufRef->release();
	 return false;
	 }
	 */
	return true;
}

// process buffer received via I2O_SM_DATA_DISCARD message
//______________________________________________________________________________
bool FUResourceQueue::discardDataEventWhileHalting(MemRef_t* bufRef) {

	/*
	 I2O_FU_DATA_DISCARD_MESSAGE_FRAME *msg;
	 msg = (I2O_FU_DATA_DISCARD_MESSAGE_FRAME*) bufRef->getDataLocation();
	 UInt_t recoIndex = msg->rbBufferID;

	 if (acceptSMDataDiscard_[recoIndex]) {
	 lock();
	 nbPendingSMDiscards_--;
	 unlock();
	 acceptSMDataDiscard_[recoIndex] = false;

	 if (!isHalting_) {
	 shmBuffer_->discardRecoCell(recoIndex);
	 bufRef->release();
	 }
	 } else {
	 LOG4CPLUS_ERROR(log_, "Spurious DATA discard by StorageManager, skip!");
	 }

	 if (isHalting_) {
	 bufRef->release();
	 return false;
	 }
	 */
	return true;
}

//concept discardDqmEvent still required?
// process buffer received via I2O_SM_DQM_DISCARD message
//______________________________________________________________________________
bool FUResourceQueue::discardDqmEvent(MemRef_t* bufRef) {
	/*
	 I2O_FU_DQM_DISCARD_MESSAGE_FRAME *msg;
	 msg = (I2O_FU_DQM_DISCARD_MESSAGE_FRAME*) bufRef->getDataLocation();
	 UInt_t dqmIndex = msg->rbBufferID;
	 unsigned int ntries = 0;
	 while (shmBuffer_->dqmState(dqmIndex) != dqm::SENT) {
	 LOG4CPLUS_WARN(
	 log_,
	 "DQM discard for cell " << dqmIndex
	 << " which is not yer in SENT state - waiting");
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
	 if (acceptSMDqmDiscard_[dqmIndex] > 0) {
	 acceptSMDqmDiscard_[dqmIndex]--;
	 if (nbPendingSMDqmDiscards_ > 0) {
	 nbPendingSMDqmDiscards_--;
	 } else {
	 LOG4CPLUS_WARN		(log_,"Spurious??? DQM discard by StorageManager, index " << dqmIndex
	 << " cell state " << shmBuffer_->dqmState(dqmIndex) << " accept flag " << acceptSMDqmDiscard_[dqmIndex];);
	 }

	 if (!isHalting_) {
	 shmBuffer_->discardDqmCell(dqmIndex);
	 bufRef->release();
	 }

	 }
	 else {
	 LOG4CPLUS_ERROR(log_,"Spurious DQM discard for cell " << dqmIndex
	 << " from StorageManager while cell is not accepting discards");
	 }

	 if (isHalting_) {
	 bufRef->release();
	 return false;
	 }
	 */
	return true;
}

// process buffer received via I2O_SM_DQM_DISCARD message
//______________________________________________________________________________
bool FUResourceQueue::discardDqmEventWhileHalting(MemRef_t* bufRef) {
	/*
	 I2O_FU_DQM_DISCARD_MESSAGE_FRAME *msg;
	 msg = (I2O_FU_DQM_DISCARD_MESSAGE_FRAME*) bufRef->getDataLocation();
	 UInt_t dqmIndex = msg->rbBufferID;
	 unsigned int ntries = 0;
	 while (shmBuffer_->dqmState(dqmIndex) != dqm::SENT) {
	 LOG4CPLUS_WARN(
	 log_,
	 "DQM discard for cell " << dqmIndex
	 << " which is not yer in SENT state - waiting");
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
	 if (acceptSMDqmDiscard_[dqmIndex] > 0) {
	 acceptSMDqmDiscard_[dqmIndex]--;
	 if (nbPendingSMDqmDiscards_ > 0) {
	 nbPendingSMDqmDiscards_--;
	 } else {
	 LOG4CPLUS_WARN		(log_,"Spurious??? DQM discard by StorageManager, index " << dqmIndex
	 << " cell state " << shmBuffer_->dqmState(dqmIndex) << " accept flag " << acceptSMDqmDiscard_[dqmIndex];);
	 }

	 if (!isHalting_) {
	 shmBuffer_->discardDqmCell(dqmIndex);
	 bufRef->release();
	 }

	 }
	 else {
	 LOG4CPLUS_ERROR(log_,"Spurious DQM discard for cell " << dqmIndex
	 << " from StorageManager while cell is not accepting discards");
	 }

	 if (isHalting_) {
	 bufRef->release();
	 return false;
	 }
	 */
	return true;
}

//concept add message type, post end-of-ls event to msq
//______________________________________________________________________________
void FUResourceQueue::postEndOfLumiSection(MemRef_t* bufRef) {
	/*
	 I2O_EVM_END_OF_LUMISECTION_MESSAGE_FRAME
	 *msg =
	 (I2O_EVM_END_OF_LUMISECTION_MESSAGE_FRAME *) bufRef->getDataLocation();
	 //make sure to fill up the shmem so no process will miss it
	 // but processes will have to handle duplicates

	 for (unsigned int i = 0; i < nbRawCells_; i++)
	 shmBuffer_->writeRawLumiSectionEvent(msg->lumiSection);
	 */
}

//concept dropEvent required?
//______________________________________________________________________________
void FUResourceQueue::dropEvent() {
	/*
	 FUShmRawCell* cell = shmBuffer_->rawCellToRead();
	 UInt_t fuResourceId = cell->fuResourceId();
	 shmBuffer_->finishReadingRawCell(cell);
	 shmBuffer_->scheduleRawCellForDiscard(fuResourceId);
	 */
}

//concept switch to error message send
//______________________________________________________________________________
bool FUResourceQueue::handleCrashedEP(UInt_t runNumber, pid_t pid) {
	bool retval = false;
	/*
	 vector<pid_t> pids = cellPrcIds();
	 UInt_t iRawCell = pids.size();
	 for (UInt_t i = 0; i < pids.size(); i++) {
	 if (pid == pids[i]) {
	 iRawCell = i;
	 break;
	 }
	 }

	 if (iRawCell < pids.size()) {
	 shmBuffer_->writeErrorEventData(runNumber, pid, iRawCell);
	 retval = true;
	 } else
	 LOG4CPLUS_WARN(log_,
	 "No raw data to send to error stream for process " << pid);
	 shmBuffer_->removeClientPrcId(pid);
	 */
	return retval;
}

//concept RAW with nothing in it
//______________________________________________________________________________
void FUResourceQueue::shutDownClients() {
	isReadyToShutDown_ = true;
	/*
	 nbClientsToShutDown_ = nbClients();
	 isReadyToShutDown_ = false;

	 if (nbClientsToShutDown_ == 0) {
	 LOG4CPLUS_INFO(
	 log_,
	 "No clients to shut down. Checking if there are raw cells not assigned to any process yet");
	 UInt_t n = nbResources();
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
	 } else {
	 UInt_t n = nbClientsToShutDown_;
	 for (UInt_t i = 0; i < n; ++i)
	 shmBuffer_->writeRawEmptyEvent();
	 }
	 */
}

//______________________________________________________________________________
void FUResourceQueue::clear() {
	for (UInt_t i = 0; i < resources_.size(); i++) {
		resources_[i]->release(false);
		delete resources_[i];
	}
	resources_.clear();

	while (!freeResourceIds_.empty())
		freeResourceIds_.pop();
}

////////////////////////////////////////////////////////////////////////////////
// implementation of private member functions
////////////////////////////////////////////////////////////////////////////////

//concept adapt resetCounters
//______________________________________________________________________________
void FUResourceQueue::resetCounters() {
	/*
	 if (0 != shmBuffer_) {
	 for (UInt_t i = 0; i < shmBuffer_->nRecoCells(); i++)
	 acceptSMDataDiscard_[i] = false;
	 for (UInt_t i = 0; i < shmBuffer_->nDqmCells(); i++)
	 acceptSMDqmDiscard_[i] = 0;
	 }
	 */
	nbAllocated_ = nbPending_;
	nbCompleted_ = 0;
	nbSent_ = 0;
	nbSentError_ = 0;
	nbSentDqm_ = 0;
	nbPendingSMDiscards_ = 0;
	nbPendingSMDqmDiscards_ = 0;
	nbDiscarded_ = 0;
	nbLost_ = 0;

	nbErrors_ = 0;
	nbCrcErrors_ = 0;
	nbAllocSent_ = 0;

	sumOfSquares_ = 0;
	sumOfSizes_ = 0;
	//isStopping_ = false;

}

//concept adapt nbClients
//______________________________________________________________________________
UInt_t FUResourceQueue::nbClients() const {
	UInt_t result(0);

	/*
	 if (0 != shmBuffer_)
	 result = shmBuffer_->nClients();
	 */
	return result;
}

//concept adapt clientPrcIds
//______________________________________________________________________________
vector<pid_t> FUResourceQueue::clientPrcIds() const {
	vector<pid_t> result;

	/*
	 if (0 != shmBuffer_) {
	 UInt_t n = nbClients();
	 for (UInt_t i = 0; i < n; i++)
	 result.push_back(shmBuffer_->clientPrcId(i));
	 }
	 */
	return result;
}

//concept adapt clientPrcIdsAsString
//______________________________________________________________________________
string FUResourceQueue::clientPrcIdsAsString() const {
	stringstream ss;

	/*
	 if (0 != shmBuffer_) {
	 UInt_t n = nbClients();
	 for (UInt_t i = 0; i < n; i++) {
	 if (i > 0)
	 ss << ",";
	 ss << shmBuffer_->clientPrcId(i);
	 }
	 }
	 */
	return ss.str();
}

//concept adapt cellStates
//______________________________________________________________________________
vector<string> FUResourceQueue::cellStates() const {
	vector<string> result;
	/*
	 if (0 != shmBuffer_) {
	 UInt_t n = nbResources();
	 shmBuffer_->lock();
	 for (UInt_t i = 0; i < n; i++) {
	 evt::State_t state = shmBuffer_->evtState(i);
	 if (state == evt::EMPTY)
	 result.push_back("EMPTY");
	 else if (state == evt::STOP)
	 result.push_back("STOP");
	 else if (state == evt::LUMISECTION)
	 result.push_back("LUMISECTION");
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
	 shmBuffer_->unlock();
	 }
	 */
	return result;
}

//concept adapt dqmCellStates
vector<string> FUResourceQueue::dqmCellStates() const {
	vector<string> result;

	/*
	 if (0 != shmBuffer_) {
	 UInt_t n = nbDqmCells_;
	 shmBuffer_->lock();
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
	 shmBuffer_->unlock();
	 }
	 */
	return result;
}

//concept adapt cellEvtNumbers
//______________________________________________________________________________
vector<UInt_t> FUResourceQueue::cellEvtNumbers() const {
	vector<UInt_t> result;

	/*
	 if (0 != shmBuffer_) {
	 UInt_t n = nbResources();
	 shmBuffer_->lock();
	 for (UInt_t i = 0; i < n; i++)
	 result.push_back(shmBuffer_->evtNumber(i));
	 shmBuffer_->unlock();
	 }
	 */
	return result;
}

//concept adapt cellPrcIds
//______________________________________________________________________________
vector<pid_t> FUResourceQueue::cellPrcIds() const {
	vector<pid_t> result;

	/*
	 if (0 != shmBuffer_) {
	 UInt_t n = nbResources();
	 shmBuffer_->lock();
	 for (UInt_t i = 0; i < n; i++)
	 result.push_back(shmBuffer_->evtPrcId(i));
	 shmBuffer_->unlock();
	 }
	 */
	return result;
}

//concept adapt cellTimeStamps
//______________________________________________________________________________
vector<time_t> FUResourceQueue::cellTimeStamps() const {
	vector<time_t> result;

	/*
	 if (0 != shmBuffer_) {
	 UInt_t n = nbResources();
	 shmBuffer_->lock();
	 for (UInt_t i = 0; i < n; i++)
	 result.push_back(shmBuffer_->evtTimeStamp(i));
	 shmBuffer_->unlock();
	 }
	 */
	return result;
}

////////////////////////////////////////////////////////////////////////////////
// implementation of private member functions
////////////////////////////////////////////////////////////////////////////////

//concept adapt lastResort
void FUResourceQueue::lastResort() {

	/*
	 cout << "lastResort: " << shmBuffer_->nbRawCellsToRead()
	 << " more rawcells to read " << endl;
	 while (shmBuffer_->nbRawCellsToRead() != 0) {
	 FUShmRawCell* newCell = shmBuffer_->rawCellToRead();
	 cout << "lastResort: " << shmBuffer_->nbRawCellsToRead()
	 << endl;
	 shmBuffer_->scheduleRawEmptyCellForDiscardServerSide(newCell);
	 cout << "lastResort: schedule raw cell for discard" << endl;
	 }
	 */
}

