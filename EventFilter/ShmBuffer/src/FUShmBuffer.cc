////////////////////////////////////////////////////////////////////////////////
//
// FUShmBuffer
// -----------
//
//            15/11/2006 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////


#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "EventFilter/FEDInterface/interface/GlobalEventNumber.h"
#include "EventFilter/ShmBuffer/interface/FUShmBuffer.h"
#include "EventFilter/Utilities/interface/Exception.h"

#include <unistd.h>
#include <iostream>
#include <string>
#include <cassert>

#include <sstream>
#include <fstream>

#include <cstdlib>
#include <cstring>
#include <stdint.h>

// the shmem keys are henceforth going to be FIXED for a give userid
// prior to creation, the application will attempt to get ownership
// of existing segments by the same key and destroy them

#define SHM_DESCRIPTOR_KEYID           1 /* Id used on ftok for 1. shmget key */
#define SHM_KEYID                      2 /* Id used on ftok for 2. shmget key */
#define SEM_KEYID                      1 /* Id used on ftok for semget key    */

#define NSKIP_MAX                    100

using namespace std;
using namespace evf;

//obsolete!!!
const char* FUShmBuffer::shmKeyPath_ =
		(getenv("FUSHM_KEYFILE") == NULL ? "/dev/null"
				: getenv("FUSHM_KEYFILE"));
const char* FUShmBuffer::semKeyPath_ =
		(getenv("FUSEM_KEYFILE") == NULL ? "/dev/null"
				: getenv("FUSEM_KEYFILE"));

////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
FUShmBuffer::FUShmBuffer(bool segmentationMode, unsigned int nRawCells,
		unsigned int nRecoCells, unsigned int nDqmCells,
		unsigned int rawCellSize, unsigned int recoCellSize,
		unsigned int dqmCellSize) :
	segmentationMode_(segmentationMode), nClientsMax_(128),
			nRawCells_(nRawCells), rawCellPayloadSize_(rawCellSize),
			nRecoCells_(nRecoCells), recoCellPayloadSize_(recoCellSize),
			nDqmCells_(nDqmCells), dqmCellPayloadSize_(dqmCellSize) {
	rawCellTotalSize_ = FUShmRawCell::size(rawCellPayloadSize_);
	recoCellTotalSize_ = FUShmRecoCell::size(recoCellPayloadSize_);
	dqmCellTotalSize_ = FUShmDqmCell::size(dqmCellPayloadSize_);

	void* addr;

	rawWriteOffset_ = sizeof(FUShmBuffer);
	addr = (void*) ((unsigned long) this + rawWriteOffset_);
	new (addr) unsigned int[nRawCells_];

	rawReadOffset_ = rawWriteOffset_ + nRawCells_ * sizeof(unsigned int);
	addr = (void*) ((unsigned long) this + rawReadOffset_);
	new (addr) unsigned int[nRawCells_];

	recoWriteOffset_ = rawReadOffset_ + nRawCells_ * sizeof(unsigned int);
	addr = (void*) ((unsigned long) this + recoWriteOffset_);
	new (addr) unsigned int[nRecoCells_];

	recoReadOffset_ = recoWriteOffset_ + nRecoCells_ * sizeof(unsigned int);
	addr = (void*) ((unsigned long) this + recoReadOffset_);
	new (addr) unsigned int[nRecoCells_];

	dqmWriteOffset_ = recoReadOffset_ + nRecoCells_ * sizeof(unsigned int);
	addr = (void*) ((unsigned long) this + dqmWriteOffset_);
	new (addr) unsigned int[nDqmCells_];

	dqmReadOffset_ = dqmWriteOffset_ + nDqmCells_ * sizeof(unsigned int);
	addr = (void*) ((unsigned long) this + dqmReadOffset_);
	new (addr) unsigned int[nDqmCells_];

	evtStateOffset_ = dqmReadOffset_ + nDqmCells_ * sizeof(unsigned int);
	addr = (void*) ((unsigned long) this + evtStateOffset_);
	new (addr) evt::State_t[nRawCells_];

	evtDiscardOffset_ = evtStateOffset_ + nRawCells_ * sizeof(evt::State_t);
	addr = (void*) ((unsigned long) this + evtDiscardOffset_);
	new (addr) unsigned int[nRawCells_];

	evtNumberOffset_ = evtDiscardOffset_ + nRawCells_ * sizeof(unsigned int);
	addr = (void*) ((unsigned long) this + evtNumberOffset_);
	new (addr) unsigned int[nRawCells_];

	evtPrcIdOffset_ = evtNumberOffset_ + nRawCells_ * sizeof(unsigned int);
	addr = (void*) ((unsigned long) this + evtPrcIdOffset_);
	new (addr) pid_t[nRawCells_];

	evtTimeStampOffset_ = evtPrcIdOffset_ + nRawCells_ * sizeof(pid_t);
	addr = (void*) ((unsigned long) this + evtTimeStampOffset_);
	new (addr) time_t[nRawCells_];

	dqmStateOffset_ = evtTimeStampOffset_ + nRawCells_ * sizeof(time_t);
	addr = (void*) ((unsigned long) this + dqmStateOffset_);
	new (addr) dqm::State_t[nDqmCells_];

	clientPrcIdOffset_ = dqmStateOffset_ + nDqmCells_ * sizeof(dqm::State_t);
	addr = (void*) ((unsigned long) this + clientPrcIdOffset_);
	new (addr) pid_t[nClientsMax_];

	rawCellOffset_ = clientPrcIdOffset_ + nClientsMax_ * sizeof(pid_t);

	if (segmentationMode_) {
		recoCellOffset_ = rawCellOffset_ + nRawCells_ * sizeof(key_t);
		dqmCellOffset_ = recoCellOffset_ + nRecoCells_ * sizeof(key_t);
		addr = (void*) ((unsigned long) this + rawCellOffset_);
		new (addr) key_t[nRawCells_];
		addr = (void*) ((unsigned long) this + recoCellOffset_);
		new (addr) key_t[nRecoCells_];
		addr = (void*) ((unsigned long) this + dqmCellOffset_);
		new (addr) key_t[nDqmCells_];
	} else {
		recoCellOffset_ = rawCellOffset_ + nRawCells_ * rawCellTotalSize_;
		dqmCellOffset_ = recoCellOffset_ + nRecoCells_ * recoCellTotalSize_;
		for (unsigned int i = 0; i < nRawCells_; i++) {
			addr = (void*) ((unsigned long) this + rawCellOffset_ + i
					* rawCellTotalSize_);
			new (addr) FUShmRawCell(rawCellSize);
		}
		for (unsigned int i = 0; i < nRecoCells_; i++) {
			addr = (void*) ((unsigned long) this + recoCellOffset_ + i
					* recoCellTotalSize_);
			new (addr) FUShmRecoCell(recoCellSize);
		}
		for (unsigned int i = 0; i < nDqmCells_; i++) {
			addr = (void*) ((unsigned long) this + dqmCellOffset_ + i
					* dqmCellTotalSize_);
			new (addr) FUShmDqmCell(dqmCellSize);
		}
	}
}

//______________________________________________________________________________
FUShmBuffer::~FUShmBuffer() {

}

////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////


//______________________________________________________________________________
void FUShmBuffer::initialize(unsigned int shmid, unsigned int semid) {
	shmid_ = shmid;
	semid_ = semid;

	if (segmentationMode_) {
		int shmKeyId = 666;
		key_t* keyAddr = (key_t*) ((unsigned long) this + rawCellOffset_);
		for (unsigned int i = 0; i < nRawCells_; i++) {
			*keyAddr = ftok(shmKeyPath_, shmKeyId++);
			int shmid = shm_create(*keyAddr, rawCellTotalSize_);
			void* shmAddr = shm_attach(shmid);
			new (shmAddr) FUShmRawCell(rawCellPayloadSize_);
			shmdt(shmAddr);
			++keyAddr;
		}
		keyAddr = (key_t*) ((unsigned long) this + recoCellOffset_);
		for (unsigned int i = 0; i < nRecoCells_; i++) {
			*keyAddr = ftok(shmKeyPath_, shmKeyId++);
			int shmid = shm_create(*keyAddr, recoCellTotalSize_);
			void* shmAddr = shm_attach(shmid);
			new (shmAddr) FUShmRecoCell(recoCellPayloadSize_);
			shmdt(shmAddr);
			++keyAddr;
		}
		keyAddr = (key_t*) ((unsigned long) this + dqmCellOffset_);
		for (unsigned int i = 0; i < nDqmCells_; i++) {
			*keyAddr = ftok(shmKeyPath_, shmKeyId++);
			int shmid = shm_create(*keyAddr, dqmCellTotalSize_);
			void* shmAddr = shm_attach(shmid);
			new (shmAddr) FUShmDqmCell(dqmCellPayloadSize_);
			shmdt(shmAddr);
			++keyAddr;
		}
	}

	reset(true);
}

//______________________________________________________________________________
void FUShmBuffer::reset(bool shm_detach) {
	nClients_ = 0;


	for (unsigned int i = 0; i < nRawCells_; i++) {
		FUShmRawCell* cell = rawCell(i);
		cell->initialize(i);
		if (segmentationMode_ && shm_detach)
			shmdt(cell);
	}

	for (unsigned int i = 0; i < nRecoCells_; i++) {
		FUShmRecoCell* cell = recoCell(i);
		cell->initialize(i);
		if (segmentationMode_ && shm_detach)
			shmdt(cell);
	}

	for (unsigned int i = 0; i < nDqmCells_; i++) {
		FUShmDqmCell* cell = dqmCell(i);
		cell->initialize(i);
		if (segmentationMode_ && shm_detach)
			shmdt(cell);
	}


	// setup ipc semaphores
	sem_init(0, 1); // lock (binary)
	sem_init(1, nRawCells_); // raw  write semaphore
	sem_init(2, 0); // raw  read  semaphore
	sem_init(3, 1); // binary semaphore to schedule raw event for discard
	sem_init(4, 0); // binary semaphore to discard raw event
	sem_init(5, nRecoCells_);// reco write semaphore
	sem_init(6, 0); // reco send (read) semaphore
	sem_init(7, nDqmCells_); // dqm  write semaphore
	sem_init(8, 0); // dqm  send (read) semaphore

	sem_print();

	unsigned int *iWrite, *iRead;

	rawWriteNext_ = 0;
	rawWriteLast_ = 0;
	rawReadNext_ = 0;
	rawReadLast_ = 0;
	iWrite = (unsigned int*) ((unsigned long) this + rawWriteOffset_);
	iRead = (unsigned int*) ((unsigned long) this + rawReadOffset_);
	for (unsigned int i = 0; i < nRawCells_; i++) {
		*iWrite++ = i;
		*iRead++ = 0xffffffff;
	}

	recoWriteNext_ = 0;
	recoWriteLast_ = 0;
	recoReadNext_ = 0;
	recoReadLast_ = 0;
	iWrite = (unsigned int*) ((unsigned long) this + recoWriteOffset_);
	iRead = (unsigned int*) ((unsigned long) this + recoReadOffset_);
	for (unsigned int i = 0; i < nRecoCells_; i++) {
		*iWrite++ = i;
		*iRead++ = 0xffffffff;
	}

	dqmWriteNext_ = 0;
	dqmWriteLast_ = 0;
	dqmReadNext_ = 0;
	dqmReadLast_ = 0;
	iWrite = (unsigned int*) ((unsigned long) this + dqmWriteOffset_);
	iRead = (unsigned int*) ((unsigned long) this + dqmReadOffset_);
	for (unsigned int i = 0; i < nDqmCells_; i++) {
		*iWrite++ = i;
		*iRead++ = 0xffffffff;
	}

	for (unsigned int i = 0; i < nRawCells_; i++) {
		setEvtState(i, evt::EMPTY);
		setEvtDiscard(i, 0);
		setEvtNumber(i, 0xffffffff);
		setEvtPrcId(i, 0);
		setEvtTimeStamp(i, 0);
	}

	for (unsigned int i = 0; i < nDqmCells_; i++)
		setDqmState(i, dqm::EMPTY);
}

//______________________________________________________________________________
unsigned int FUShmBuffer::nbRawCellsToWrite() const {
	return semctl(semid(), 1, GETVAL);
}

//______________________________________________________________________________
int FUShmBuffer::nbRawCellsToRead() const {
	return semctl(semid(), 2, GETVAL);
}

//______________________________________________________________________________
FUShmRawCell* FUShmBuffer::rawCellToWrite() {
	if (waitRawWrite() != 0)
		return 0;
	unsigned int iCell = nextRawWriteIndex();
	FUShmRawCell* cell = rawCell(iCell);
	evt::State_t state = evtState(iCell);
	stringstream details;
	details << "state==evt::EMPTY assertion failed! Actual state is " << state
			<< ", iCell = " << iCell;
	XCEPT_ASSERT(state == evt::EMPTY, evf::Exception, details.str());
	setEvtState(iCell, evt::RAWWRITING);
	setEvtDiscard(iCell, 1);
	return cell;
}

//______________________________________________________________________________
FUShmRawCell* FUShmBuffer::rawCellToRead() {
	waitRawRead();
	unsigned int iCell = nextRawReadIndex();
	FUShmRawCell* cell = rawCell(iCell);
	evt::State_t state = evtState(iCell);
	stringstream details;
	details
			<< "state==evt::RAWWRITTEN ||state==evt::EMPTY ||state==evt::STOP ||state==evt::LUMISECTION assertion failed! Actual state is "
			<< state << ", iCell = " << iCell;
	XCEPT_ASSERT(
			state == evt::RAWWRITTEN || state == evt::EMPTY || state
					== evt::STOP || state == evt::LUMISECTION, evf::Exception,
			details.str());
	if (state == evt::RAWWRITTEN) {
		setEvtPrcId(iCell, getpid());
		setEvtState(iCell, evt::RAWREADING);
	}
	return cell;
}

//______________________________________________________________________________
FUShmRecoCell* FUShmBuffer::recoCellToRead() {
	waitRecoRead();
	unsigned int iCell = nextRecoReadIndex();
	FUShmRecoCell* cell = recoCell(iCell);
	unsigned int iRawCell = cell->rawCellIndex();
	if (iRawCell < nRawCells_) {
		//evt::State_t   state=evtState(iRawCell);
		//XCEPT_ASSERT(state==evt::RECOWRITTEN, evf::Exception, "state==evt::RECOWRITTEN assertion failed!");
		setEvtState(iRawCell, evt::SENDING);
	}
	return cell;
}

//______________________________________________________________________________
FUShmDqmCell* FUShmBuffer::dqmCellToRead() {
	waitDqmRead();
	unsigned int iCell = nextDqmReadIndex();
	FUShmDqmCell* cell = dqmCell(iCell);
	dqm::State_t state = dqmState(iCell);
	stringstream details;
	details
			<< "state==dqm::WRITTEN || state==dqm::EMPTY assertion failed! Actual state is "
			<< state << ", iCell = " << iCell;
	XCEPT_ASSERT(state == dqm::WRITTEN || state == dqm::EMPTY, evf::Exception,
			details.str());
	if (state == dqm::WRITTEN)
		setDqmState(iCell, dqm::SENDING);
	return cell;
}

//______________________________________________________________________________
FUShmRawCell* FUShmBuffer::rawCellToDiscard() {
	waitRawDiscarded();
	FUShmRawCell* cell = rawCell(rawDiscardIndex_);
	evt::State_t state = evtState(cell->index());
	stringstream details;
	details
			<< "state==evt::PROCESSED || state==evt::SENT || state==evt::EMPTY || state==evt::STOP || state==evt::USEDLS assertion failed! Actual state is "
			<< state << ", index = " << cell->index();
	XCEPT_ASSERT(
			state == evt::PROCESSED || state == evt::SENT || state
					== evt::EMPTY || state == evt::STOP || state == evt::USEDLS,
			evf::Exception, details.str());
	if (state != evt::EMPTY && state != evt::USEDLS && state != evt::STOP)
		setEvtState(cell->index(), evt::DISCARDING);
	return cell;
}

//______________________________________________________________________________
void FUShmBuffer::finishWritingRawCell(FUShmRawCell* cell) {
	evt::State_t state = evtState(cell->index());
	stringstream details;
	details << "state==evt::RAWWRITING assertion failed! Actual state is "
			<< state << ", index = " << cell->index();
	XCEPT_ASSERT(state == evt::RAWWRITING, evf::Exception, details.str());
	setEvtState(cell->index(), evt::RAWWRITTEN);
	setEvtNumber(cell->index(), cell->evtNumber());
	postRawIndexToRead(cell->index());
	if (segmentationMode_)
		shmdt(cell);
	postRawRead();
}

//______________________________________________________________________________
void FUShmBuffer::finishReadingRawCell(FUShmRawCell* cell) {
	evt::State_t state = evtState(cell->index());
	stringstream details;
	details << "state==evt::RAWREADING assertion failed! Actual state is "
			<< state << ", index = " << cell->index();
	XCEPT_ASSERT(state == evt::RAWREADING, evf::Exception, details.str());
	setEvtState(cell->index(), evt::RAWREAD);
	setEvtState(cell->index(), evt::PROCESSING);
	setEvtTimeStamp(cell->index(), time(0));
	if (segmentationMode_)
		shmdt(cell);
}

//______________________________________________________________________________
void FUShmBuffer::finishReadingRecoCell(FUShmRecoCell* cell) {
	unsigned int iRawCell = cell->rawCellIndex();
	if (iRawCell < nRawCells_) {
		//evt::State_t state=evtState(cell->rawCellIndex());
		//XCEPT_ASSERT(state==evt::SENDING, evf::Exception, "state==evt::SENDING assertion failed!");
		setEvtState(cell->rawCellIndex(), evt::SENT);
	}
	if (segmentationMode_)
		shmdt(cell);
}

//______________________________________________________________________________
void FUShmBuffer::finishReadingDqmCell(FUShmDqmCell* cell) {
	dqm::State_t state = dqmState(cell->index());
	stringstream details;
	details
			<< "state==dqm::SENDING||state==dqm::EMPTY assertion failed! Actual state is "
			<< state << ", index = " << cell->index();
	XCEPT_ASSERT(state == dqm::SENDING || state == dqm::EMPTY, evf::Exception,
			details.str());
	if (state == dqm::SENDING)
		setDqmState(cell->index(), dqm::SENT);
	if (segmentationMode_)
		shmdt(cell);
}

//______________________________________________________________________________
void FUShmBuffer::scheduleRawCellForDiscard(unsigned int iCell) {
	waitRawDiscard();
	if (rawCellReadyForDiscard(iCell)) {
		rawDiscardIndex_ = iCell;
		evt::State_t state = evtState(iCell);
		stringstream details;
		details
			<< "state==evt::PROCESSING||state==evt::SENT||state==evt::EMPTY||"
			<<"state==evt::STOP||state==evt::LUMISECTION||state==evt::RECOWRITTEN assertion failed! Actual state is "
			<< state << ", iCell = " << iCell;
		XCEPT_ASSERT(  state == evt::PROCESSING || state == evt::SENT 
				|| state == evt::EMPTY || state == evt::STOP
				|| state == evt::LUMISECTION || state == evt::RECOWRITTEN, evf::Exception, details.str());
		if (state == evt::PROCESSING)
			setEvtState(iCell, evt::PROCESSED);
		if (state == evt::LUMISECTION)
			setEvtState(iCell, evt::USEDLS);
		postRawDiscarded();
	} else
		postRawDiscard();
}

//______________________________________________________________________________
void FUShmBuffer::scheduleRawCellForDiscardServerSide(unsigned int iCell) {
	waitRawDiscard();
	if (rawCellReadyForDiscard(iCell)) {
		rawDiscardIndex_ = iCell;
		evt::State_t state = evtState(iCell);
		// UPDATE: aspataru
		if (state != evt::LUMISECTION && state != evt::EMPTY 
		    && state != evt::USEDLS && state != evt::STOP)
			setEvtState(iCell, evt::PROCESSED);
		if (state == evt::LUMISECTION)
			setEvtState(iCell, evt::USEDLS);
		postRawDiscarded();
	} else
		postRawDiscard();
}

//______________________________________________________________________________
void FUShmBuffer::discardRawCell(FUShmRawCell* cell) {
	releaseRawCell(cell);
	postRawDiscard();
}

//______________________________________________________________________________
void FUShmBuffer::discardRecoCell(unsigned int iCell) {
	FUShmRecoCell* cell = recoCell(iCell);
	unsigned int iRawCell = cell->rawCellIndex();
	if (iRawCell < nRawCells_) {
		//evt::State_t state=evtState(iRawCell);
		//XCEPT_ASSERT(state==evt::SENT, evf::Exception, "state==evt::SENT assertion failed!");
		scheduleRawCellForDiscard(iRawCell);
	}
	cell->clear();
	if (segmentationMode_)
		shmdt(cell);
	postRecoIndexToWrite(iCell);
	postRecoWrite();
}

//______________________________________________________________________________
void FUShmBuffer::discardOrphanedRecoCell(unsigned int iCell) {
	FUShmRecoCell* cell = recoCell(iCell);
	cell->clear();
	if (segmentationMode_)
		shmdt(cell);
	postRecoIndexToWrite(iCell);
	postRecoWrite();
}

//______________________________________________________________________________
void FUShmBuffer::discardDqmCell(unsigned int iCell) {
	dqm::State_t state = dqmState(iCell);
	stringstream details;
	details
			<< "state==dqm::EMPTY||state==dqm::SENT assertion failed! Actual state is "
			<< state << ", iCell = " << iCell;
	XCEPT_ASSERT(state == dqm::EMPTY || state == dqm::SENT, evf::Exception,
			details.str());
	setDqmState(iCell, dqm::DISCARDING);
	FUShmDqmCell* cell = dqmCell(iCell);
	cell->clear();
	if (segmentationMode_)
		shmdt(cell);
	setDqmState(iCell, dqm::EMPTY);
	postDqmIndexToWrite(iCell);
	postDqmWrite();
}

//______________________________________________________________________________
void FUShmBuffer::releaseRawCell(FUShmRawCell* cell) {
	evt::State_t state = evtState(cell->index());
	if (!(state == evt::DISCARDING || state == evt::RAWWRITING || state
			== evt::EMPTY || state == evt::STOP
	//     ||state==evt::LUMISECTION
			|| state == evt::USEDLS))
		std::cout << "=================releaseRawCell state " << state
				<< std::endl;
	stringstream details;
	details
			<< "state==evt::DISCARDING||state==evt::RAWWRITING||state==evt::EMPTY||state==evt::STOP||state==evt::USEDLS assertion failed! Actual state is "
			<< state << ", index = " << cell->index();
	XCEPT_ASSERT(
			state == evt::DISCARDING || state == evt::RAWWRITING || state
					== evt::EMPTY || state == evt::STOP
			/*||state==evt::LUMISECTION*/
			|| state == evt::USEDLS, evf::Exception, details.str());
	setEvtState(cell->index(), evt::EMPTY);
	setEvtDiscard(cell->index(), 0);
	setEvtNumber(cell->index(), 0xffffffff);
	setEvtPrcId(cell->index(), 0);
	setEvtTimeStamp(cell->index(), 0);
	cell->clear();
	postRawIndexToWrite(cell->index());
	if (segmentationMode_)
		shmdt(cell);
	postRawWrite();
}

//______________________________________________________________________________
void FUShmBuffer::writeRawEmptyEvent() {
	FUShmRawCell* cell = rawCellToWrite();
	if (cell == 0)
		return;
	evt::State_t state = evtState(cell->index());
	stringstream details;
	details << "state==evt::RAWWRITING assertion failed! Actual state is "
			<< state << ", index = " << cell->index();
	XCEPT_ASSERT(state == evt::RAWWRITING, evf::Exception, details.str());
	setEvtState(cell->index(), evt::STOP);
	cell->setEventTypeStopper();
	postRawIndexToRead(cell->index());
	if (segmentationMode_)
		shmdt(cell);
	postRawRead();
}

//______________________________________________________________________________
void FUShmBuffer::writeRawLumiSectionEvent(unsigned int ls) {
	FUShmRawCell* cell = rawCellToWrite();
	if (cell == 0)
		return;
	cell->setLumiSection(ls);
	evt::State_t state = evtState(cell->index());
	stringstream details;
	details << "state==evt::RAWWRITING assertion failed! Actual state is "
			<< state << ", index = " << cell->index();
	XCEPT_ASSERT(state == evt::RAWWRITING, evf::Exception, details.str());
        setEvtNumber(cell->index(),0xfffffffe);
	setEvtState(cell->index(), evt::LUMISECTION);
	cell->setEventTypeEol();
	postRawIndexToRead(cell->index());
	if (segmentationMode_)
		shmdt(cell);
	postRawRead();
}

//______________________________________________________________________________
void FUShmBuffer::writeRecoEmptyEvent() {
	waitRecoWrite();
	unsigned int iCell = nextRecoWriteIndex();
	FUShmRecoCell* cell = recoCell(iCell);
	cell->clear();
	postRecoIndexToRead(iCell);
	if (segmentationMode_)
		shmdt(cell);
	postRecoRead();
}

//______________________________________________________________________________
void FUShmBuffer::writeDqmEmptyEvent() {
	waitDqmWrite();
	unsigned int iCell = nextDqmWriteIndex();
	FUShmDqmCell* cell = dqmCell(iCell);
	cell->clear();
	postDqmIndexToRead(iCell);
	if (segmentationMode_)
		shmdt(cell);
	postDqmRead();
}

//______________________________________________________________________________
void FUShmBuffer::scheduleRawEmptyCellForDiscard() {
	FUShmRawCell* cell = rawCellToWrite();
	if (cell == 0)
		return;
	rawDiscardIndex_ = cell->index();
	setEvtState(cell->index(), evt::STOP);
	cell->setEventTypeStopper();
	setEvtNumber(cell->index(), 0xffffffff);
	setEvtPrcId(cell->index(), 0);
	setEvtTimeStamp(cell->index(), 0);
	postRawDiscarded();
}

//______________________________________________________________________________
bool FUShmBuffer::scheduleRawEmptyCellForDiscard(FUShmRawCell* cell, bool & pidstatus) {
	waitRawDiscard();
	if (rawCellReadyForDiscard(cell->index())) {
		rawDiscardIndex_ = cell->index();
		// as this function is called by the reader, the state and type should
		// already be correct
		//    setEvtState(cell->index(),evt::STOP);
		//    cell->setEventType(evt::STOPPER);
		//    setEvtNumber(cell->index(),0xffffffff);
		//    setEvtPrcId(cell->index(),0);
		//    setEvtTimeStamp(cell->index(),0);
		pidstatus = removeClientPrcId(getpid());
		if (segmentationMode_)
			shmdt(cell);
		postRawDiscarded();
		return true;
	} else {
		postRawDiscard();
		return false;
	}
}

//______________________________________________________________________________
void FUShmBuffer::scheduleRawEmptyCellForDiscardServerSide(FUShmRawCell* cell) {
	//  waitRawDiscard();
	if (rawCellReadyForDiscard(cell->index())) {
		rawDiscardIndex_ = cell->index();
		//    setEvtState(cell->index(),evt::STOP);
		//    cell->setEventType(evt::STOPPER);
		//    setEvtNumber(cell->index(),0xffffffff);
		//    setEvtPrcId(cell->index(),0);
		//    setEvtTimeStamp(cell->index(),0);
		//    removeClientPrcId(getpid());
		if (segmentationMode_)
			shmdt(cell);
		postRawDiscarded();
	} else
		postRawDiscard();
}

//______________________________________________________________________________
bool FUShmBuffer::writeRecoInitMsg(unsigned int outModId,
		unsigned int fuProcessId, unsigned int fuGuid, unsigned char *data,
		unsigned int dataSize, unsigned int nExpectedEPs) {
	if (dataSize > recoCellPayloadSize_) {
		cout << "FUShmBuffer::writeRecoInitMsg() ERROR: buffer overflow."
				<< endl;
		return false;
	}

	waitRecoWrite();
	unsigned int iCell = nextRecoWriteIndex();
	FUShmRecoCell* cell = recoCell(iCell);
	cell->writeInitMsg(outModId, fuProcessId, fuGuid, data, dataSize,nExpectedEPs);
	postRecoIndexToRead(iCell);
	if (segmentationMode_)
		shmdt(cell);
	postRecoRead();
	return true;
}

//______________________________________________________________________________
bool FUShmBuffer::writeRecoEventData(unsigned int runNumber,
		unsigned int evtNumber, unsigned int outModId,
		unsigned int fuProcessId, unsigned int fuGuid, unsigned char *data,
		unsigned int dataSize) {
	if (dataSize > recoCellPayloadSize_) {
		cout << "FUShmBuffer::writeRecoEventData() ERROR: buffer overflow."
				<< endl;
		return false;
	}

	waitRecoWrite();
	unsigned int rawCellIndex = indexForEvtNumber(evtNumber);
	unsigned int iCell = nextRecoWriteIndex();
	FUShmRecoCell* cell = recoCell(iCell);
	//evt::State_t state=evtState(rawCellIndex);
	//XCEPT_ASSERT(state==evt::PROCESSING||state==evt::RECOWRITING||state==evt::SENT, evf::Exception, "state==evt::PROCESSING||state==evt::RECOWRITING||state==evt::SENT assertion failed!");
	setEvtState(rawCellIndex, evt::RECOWRITING);
	incEvtDiscard(rawCellIndex);
	cell->writeEventData(rawCellIndex, runNumber, evtNumber, outModId,
			fuProcessId, fuGuid, data, dataSize);
	setEvtState(rawCellIndex, evt::RECOWRITTEN);
	postRecoIndexToRead(iCell);
	if (segmentationMode_)
		shmdt(cell);
	postRecoRead();
	return true;
}

//______________________________________________________________________________
bool FUShmBuffer::writeErrorEventData(unsigned int runNumber,
		unsigned int fuProcessId, unsigned int iRawCell, bool checkValue) {
	FUShmRawCell *raw = rawCell(iRawCell);

	unsigned int dataSize = sizeof(uint32_t) * (4 + 1024) + raw->eventSize();
	unsigned char *data = new unsigned char[dataSize];
	uint32_t *pos = (uint32_t*) data;
	// 06-Oct-2008, KAB - added a version number for the error event format.
	//
	// Version 1 had no version number, so the run number appeared in the
	// first value.  So, a reasonable test for version 1 is whether the
	// first value is larger than some relatively small cutoff (say 32).
	// Version 2 added the lumi block number.
	//
	*pos++ = (uint32_t) 2; // protocol version number
	*pos++ = (uint32_t) runNumber;
	*pos++ = (uint32_t) evf::evtn::getlbn(
			raw->fedAddr(FEDNumbering::MINTriggerGTPFEDID)) + 1;
	*pos++ = (uint32_t) raw->evtNumber();
	for (unsigned int i = 0; i < 1024; i++)
		*pos++ = (uint32_t) raw->fedSize(i);
	memcpy(pos, raw->payloadAddr(), raw->eventSize());

	// DEBUG
	/*
	 if (1) {
	 stringstream ss;
	 ss<<"/tmp/run"<<runNumber<<"_evt"<<raw->evtNumber()<<".err";
	 ofstream fout;
	 fout.open(ss.str().c_str(),ios::out|ios::binary);
	 if (!fout.write((char*)data,dataSize))
	 cout<<"Failed to write error event to "<<ss.str()<<endl;
	 fout.close();

	 stringstream ss2;
	 ss2<<"/tmp/run"<<runNumber<<"_evt"<<raw->evtNumber()<<".info";
	 ofstream fout2;
	 fout2.open(ss2.str().c_str());
	 fout2<<"dataSize = "<<dataSize<<endl;
	 fout2<<"runNumber = "<<runNumber<<endl;
	 fout2<<"evtNumber = "<<raw->evtNumber()<<endl;
	 fout2<<"eventSize = "<<raw->eventSize()<<endl;
	 unsigned int totalSize(0);
	 for (unsigned int i=0;i<1024;i++) {
	 unsigned int fedSize = raw->fedSize(i);
	 totalSize += fedSize;
	 if (fedSize>0) fout2<<i<<": "<<fedSize<<endl;
	 }
	 fout2<<"totalSize = "<<totalSize<<endl;
	 fout2.close();
	 }
	 */// END DEBUG

	waitRecoWrite();
	unsigned int iRecoCell = nextRecoWriteIndex();
	FUShmRecoCell* reco = recoCell(iRecoCell);
	setEvtState(iRawCell, evt::RECOWRITING);
	setEvtDiscard(iRawCell, 1, checkValue);
	reco->writeErrorEvent(iRawCell, runNumber, raw->evtNumber(), fuProcessId,
			data, dataSize);
	delete[] data;
	setEvtState(iRawCell, evt::RECOWRITTEN);
	postRecoIndexToRead(iRecoCell);
	if (segmentationMode_) {
		shmdt(raw);
		shmdt(reco);
	}
	postRecoRead();
	return true;
}

//______________________________________________________________________________
bool FUShmBuffer::writeDqmEventData(unsigned int runNumber,
		unsigned int evtAtUpdate, unsigned int folderId,
		unsigned int fuProcessId, unsigned int fuGuid, unsigned char *data,
		unsigned int dataSize) {
	if (dataSize > dqmCellPayloadSize_) {
		cout << "FUShmBuffer::writeDqmEventData() ERROR: buffer overflow."
				<< endl;
		return false;
	}

	waitDqmWrite();
	unsigned int iCell = nextDqmWriteIndex();
	FUShmDqmCell* cell = dqmCell(iCell);
	dqm::State_t state = dqmState(iCell);
	stringstream details;
	details << "state==dqm::EMPTY assertion failed! Actual state is " << state
			<< ", iCell = " << iCell;
	XCEPT_ASSERT(state == dqm::EMPTY, evf::Exception, details.str());
	setDqmState(iCell, dqm::WRITING);
	cell->writeData(runNumber, evtAtUpdate, folderId, fuProcessId, fuGuid,
			data, dataSize);
	setDqmState(iCell, dqm::WRITTEN);
	postDqmIndexToRead(iCell);
	if (segmentationMode_)
		shmdt(cell);
	postDqmRead();
	return true;
}

//______________________________________________________________________________
void FUShmBuffer::sem_print() {
	cout << "--> current sem values:" << endl << " lock=" << semctl(semid(), 0,
			GETVAL) << endl << " wraw=" << semctl(semid(), 1, GETVAL)
			<< " rraw=" << semctl(semid(), 2, GETVAL) << endl << " wdsc="
			<< semctl(semid(), 3, GETVAL) << " rdsc=" << semctl(semid(), 4,
			GETVAL) << endl << " wrec=" << semctl(semid(), 5, GETVAL)
			<< " rrec=" << semctl(semid(), 6, GETVAL) << endl << " wdqm="
			<< semctl(semid(), 7, GETVAL) << " rdqm=" << semctl(semid(), 8,
			GETVAL) << endl;
}

std::string FUShmBuffer::sem_print_s() {
	ostringstream ostr;
	ostr    << "--> current sem values:" << endl << " lock=" << semctl(semid(), 0,
			GETVAL) << endl << " wraw=" << semctl(semid(), 1, GETVAL)
		<< " rraw=" << semctl(semid(), 2, GETVAL) << endl << " wdsc="
		<< semctl(semid(), 3, GETVAL) << " rdsc=" << semctl(semid(), 4,
				GETVAL) << endl << " wrec=" << semctl(semid(), 5, GETVAL)
		<< " rrec=" << semctl(semid(), 6, GETVAL) << endl << " wdqm="
		<< semctl(semid(), 7, GETVAL) << " rdqm=" << semctl(semid(), 8,
				GETVAL) << endl;
	return ostr.str();

}

//______________________________________________________________________________
void FUShmBuffer::printEvtState(unsigned int index) {
	evt::State_t state = evtState(index);
	std::string stateName;
	if (state == evt::EMPTY)
		stateName = "EMPTY";
	else if (state == evt::STOP)
		stateName = "STOP";
	else if (state == evt::RAWWRITING)
		stateName = "RAWWRITING";
	else if (state == evt::RAWWRITTEN)
		stateName = "RAWRITTEN";
	else if (state == evt::RAWREADING)
		stateName = "RAWREADING";
	else if (state == evt::RAWREAD)
		stateName = "RAWREAD";
	else if (state == evt::PROCESSING)
		stateName = "PROCESSING";
	else if (state == evt::PROCESSED)
		stateName = "PROCESSED";
	else if (state == evt::RECOWRITING)
		stateName = "RECOWRITING";
	else if (state == evt::RECOWRITTEN)
		stateName = "RECOWRITTEN";
	else if (state == evt::SENDING)
		stateName = "SENDING";
	else if (state == evt::SENT)
		stateName = "SENT";
	else if (state == evt::DISCARDING)
		stateName = "DISCARDING";
	cout << "evt " << index << " in state '" << stateName << "'." << endl;
}

//______________________________________________________________________________
void FUShmBuffer::printDqmState(unsigned int index) {
	dqm::State_t state = dqmState(index);
	cout << "dqm evt " << index << " in state '" << state << "'." << endl;
}

//______________________________________________________________________________
FUShmBuffer* FUShmBuffer::createShmBuffer(bool segmentationMode,
		unsigned int nRawCells, unsigned int nRecoCells,
		unsigned int nDqmCells, unsigned int rawCellSize,
		unsigned int recoCellSize, unsigned int dqmCellSize) {
	// if necessary, release shared memory first!
	if (FUShmBuffer::releaseSharedMemory())
		cout << "FUShmBuffer::createShmBuffer: "
				<< "REMOVAL OF OLD SHARED MEM SEGMENTS SUCCESSFULL." << endl;

	// create bookkeeping shared memory segment
	int size = sizeof(unsigned int) * 7;
	int shmid = shm_create(FUShmBuffer::getShmDescriptorKey(), size);
	if (shmid < 0)
		return 0;
	void*shmAddr = shm_attach(shmid);
	if (0 == shmAddr)
		return 0;

	if (1 != shm_nattch(shmid)) {
		cout << "FUShmBuffer::createShmBuffer() FAILED: nattch=" << shm_nattch(
				shmid) << endl;
		shmdt(shmAddr);
		return 0;
	}

	unsigned int* p = (unsigned int*) shmAddr;
	*p++ = segmentationMode;
	*p++ = nRawCells;
	*p++ = nRecoCells;
	*p++ = nDqmCells;
	*p++ = rawCellSize;
	*p++ = recoCellSize;
	*p++ = dqmCellSize;
	shmdt(shmAddr);

	// create the 'real' shared memory buffer
	size = FUShmBuffer::size(segmentationMode, nRawCells, nRecoCells,
			nDqmCells, rawCellSize, recoCellSize, dqmCellSize);
	shmid = shm_create(FUShmBuffer::getShmKey(), size);
	if (shmid < 0)
		return 0;
	int semid = sem_create(FUShmBuffer::getSemKey(), 9);
	if (semid < 0)
		return 0;
	shmAddr = shm_attach(shmid);
	if (0 == shmAddr)
		return 0;

	if (1 != shm_nattch(shmid)) {
		cout << "FUShmBuffer::createShmBuffer FAILED: nattch=" << shm_nattch(
				shmid) << endl;
		shmdt(shmAddr);
		return 0;
	}
	FUShmBuffer* buffer = new (shmAddr) FUShmBuffer(segmentationMode,
			nRawCells, nRecoCells, nDqmCells, rawCellSize, recoCellSize,
			dqmCellSize);

	cout << "FUShmBuffer::createShmBuffer(): CREATED shared memory buffer."
			<< endl;
	cout << "                                segmentationMode="
			<< segmentationMode << endl;

	buffer->initialize(shmid, semid);

	return buffer;
}

//______________________________________________________________________________
FUShmBuffer* FUShmBuffer::getShmBuffer() {
	// get bookkeeping shared memory segment
	int size = sizeof(unsigned int) * 7;
	int shmid = shm_get(FUShmBuffer::getShmDescriptorKey(), size);
	if (shmid < 0)
		return 0;
	void* shmAddr = shm_attach(shmid);
	if (0 == shmAddr)
		return 0;

	unsigned int *p = (unsigned int*) shmAddr;
	bool segmentationMode = *p++;
	unsigned int nRawCells = *p++;
	unsigned int nRecoCells = *p++;
	unsigned int nDqmCells = *p++;
	unsigned int rawCellSize = *p++;
	unsigned int recoCellSize = *p++;
	unsigned int dqmCellSize = *p++;
	shmdt(shmAddr);

	cout << "FUShmBuffer::getShmBuffer():" << " segmentationMode="
			<< segmentationMode << " nRawCells=" << nRawCells << " nRecoCells="
			<< nRecoCells << " nDqmCells=" << nDqmCells << " rawCellSize="
			<< rawCellSize << " recoCellSize=" << recoCellSize
			<< " dqmCellSize=" << dqmCellSize << endl;

	// get the 'real' shared memory buffer
	size = FUShmBuffer::size(segmentationMode, nRawCells, nRecoCells,
			nDqmCells, rawCellSize, recoCellSize, dqmCellSize);
	shmid = shm_get(FUShmBuffer::getShmKey(), size);
	if (shmid < 0)
		return 0;
	int semid = sem_get(FUShmBuffer::getSemKey(), 9);
	if (semid < 0)
		return 0;
	shmAddr = shm_attach(shmid);
	if (0 == shmAddr)
		return 0;

	if (0 == shm_nattch(shmid)) {
		cout << "FUShmBuffer::getShmBuffer() FAILED: nattch=" << shm_nattch(
				shmid) << endl;
		return 0;
	}
	FUShmBuffer* buffer = new (shmAddr) FUShmBuffer(segmentationMode,
			nRawCells, nRecoCells, nDqmCells, rawCellSize, recoCellSize,
			dqmCellSize);

	cout << "FUShmBuffer::getShmBuffer(): shared memory buffer RETRIEVED."
			<< endl;
	cout << "                             segmentationMode="
			<< segmentationMode << endl;

	buffer->setClientPrcId(getpid());

	return buffer;
}

//______________________________________________________________________________
bool FUShmBuffer::releaseSharedMemory() {
	// get bookkeeping shared memory segment
	int size = sizeof(unsigned int) * 7;
	int shmidd = shm_get(FUShmBuffer::getShmDescriptorKey(), size);
	if (shmidd < 0)
		return false;
	void* shmAddr = shm_attach(shmidd);
	if (0 == shmAddr)
		return false;

	unsigned int*p = (unsigned int*) shmAddr;
	bool segmentationMode = *p++;
	unsigned int nRawCells = *p++;
	unsigned int nRecoCells = *p++;
	unsigned int nDqmCells = *p++;
	unsigned int rawCellSize = *p++;
	unsigned int recoCellSize = *p++;
	unsigned int dqmCellSize = *p++;
	shmdt(shmAddr);

	// get the 'real' shared memory segment
	size = FUShmBuffer::size(segmentationMode, nRawCells, nRecoCells,
			nDqmCells, rawCellSize, recoCellSize, dqmCellSize);
	int shmid = shm_get(FUShmBuffer::getShmKey(), size);
	if (shmid < 0)
		return false;
	int semid = sem_get(FUShmBuffer::getSemKey(), 9);
	if (semid < 0)
		return false;
	shmAddr = shm_attach(shmid);
	if (0 == shmAddr)
		return false;

	int att = 0;
	for (; att < 10; att++) {
		if (shm_nattch(shmid) > 1) {
			cout << att << " FUShmBuffer::releaseSharedMemory(): nattch="
					<< shm_nattch(shmid)
					<< ", failed attempt to release shared memory." << endl;
			::sleep(1);
		} else
			break;
	}

	if (att >= 10)
		return false;

	if (segmentationMode) {
		FUShmBuffer* buffer = new (shmAddr) FUShmBuffer(segmentationMode,
				nRawCells, nRecoCells, nDqmCells, rawCellSize, recoCellSize,
				dqmCellSize);
		int cellid;
		for (unsigned int i = 0; i < nRawCells; i++) {
			cellid = shm_get(buffer->rawCellShmKey(i),
					FUShmRawCell::size(rawCellSize));
			if ((shm_destroy(cellid) == -1))
				return false;
		}
		for (unsigned int i = 0; i < nRecoCells; i++) {
			cellid = shm_get(buffer->recoCellShmKey(i),
					FUShmRecoCell::size(recoCellSize));
			if ((shm_destroy(cellid) == -1))
				return false;
		}
		for (unsigned int i = 0; i < nDqmCells; i++) {
			cellid = shm_get(buffer->dqmCellShmKey(i),
					FUShmDqmCell::size(dqmCellSize));
			if ((shm_destroy(cellid) == -1))
				return false;
		}
	}
	shmdt(shmAddr);
	if (sem_destroy(semid) == -1)
		return false;
	if (shm_destroy(shmid) == -1)
		return false;
	if (shm_destroy(shmidd) == -1)
		return false;

	return true;
}

//______________________________________________________________________________
unsigned int FUShmBuffer::size(bool segmentationMode, unsigned int nRawCells,
		unsigned int nRecoCells, unsigned int nDqmCells,
		unsigned int rawCellSize, unsigned int recoCellSize,
		unsigned int dqmCellSize) {
	unsigned int offset = sizeof(FUShmBuffer) + sizeof(unsigned int) * 4
			* nRawCells + sizeof(evt::State_t) * nRawCells
			+ sizeof(dqm::State_t) * nDqmCells;

	unsigned int rawCellTotalSize = FUShmRawCell::size(rawCellSize);
	unsigned int recoCellTotalSize = FUShmRecoCell::size(recoCellSize);
	unsigned int dqmCellTotalSize = FUShmDqmCell::size(dqmCellSize);

	unsigned int realSize = (segmentationMode) ? offset + sizeof(key_t)
			* (nRawCells + nRecoCells + nDqmCells) : offset + rawCellTotalSize
			* nRawCells + recoCellTotalSize * nRecoCells + dqmCellTotalSize
			* nDqmCells;

	unsigned int result = realSize / 0x10 * 0x10 + (realSize % 0x10 > 0) * 0x10;

	return result;
}

//______________________________________________________________________________
key_t FUShmBuffer::getShmDescriptorKey() {
	key_t result = getuid() * 1000 + SHM_DESCRIPTOR_KEYID;
	if (result == (key_t) -1)
		cout << "FUShmBuffer::getShmDescriptorKey: failed " << "for file "
				<< shmKeyPath_ << "!" << endl;
	return result;
}

//______________________________________________________________________________
key_t FUShmBuffer::getShmKey() {
	key_t result = getuid() * 1000 + SHM_KEYID;
	if (result == (key_t) -1)
		cout << "FUShmBuffer::getShmKey: ftok() failed " << "for file "
				<< shmKeyPath_ << "!" << endl;
	return result;
}

//______________________________________________________________________________
key_t FUShmBuffer::getSemKey() {
	key_t result = getuid() * 1000 + SEM_KEYID;
	if (result == (key_t) -1)
		cout << "FUShmBuffer::getSemKey: ftok() failed " << "for file "
				<< semKeyPath_ << "!" << endl;
	return result;
}

//______________________________________________________________________________
int FUShmBuffer::shm_create(key_t key, int size) {
	// first check and possibly remove existing segment with same id
	int shmid = shmget(key, 1, 0644);//using minimal size any segment with key "key" will be connected
	if (shmid != -1) {
		// an existing segment was found, remove it
		shmid_ds shmstat;
		shmctl(shmid, IPC_STAT, &shmstat);
		cout << "FUShmBuffer found segment for key 0x " << hex << key << dec
				<< " created by process " << shmstat.shm_cpid << " owned by "
				<< shmstat.shm_perm.uid << " permissions " << hex
				<< shmstat.shm_perm.mode << dec << endl;
		shmctl(shmid, IPC_RMID, &shmstat);
	}
	shmid = shmget(key, size, IPC_CREAT | 0644);
	if (shmid == -1) {
		int err = errno;
		cout << "FUShmBuffer::shm_create(" << key << "," << size
				<< ") failed: " << strerror(err) << endl;
	}
	return shmid;
}

//______________________________________________________________________________
int FUShmBuffer::shm_get(key_t key, int size) {
	int shmid = shmget(key, size, 0644);
	if (shmid == -1) {
		int err = errno;
		cout << "FUShmBuffer::shm_get(" << key << "," << size << ") failed: "
				<< strerror(err) << endl;
	}
	return shmid;
}

//______________________________________________________________________________
void* FUShmBuffer::shm_attach(int shmid) {
	void* result = shmat(shmid, NULL, 0);
	if (0 == result) {
		int err = errno;
		cout << "FUShmBuffer::shm_attach(" << shmid << ") failed: "
				<< strerror(err) << endl;
	}
	return result;
}

//______________________________________________________________________________
int FUShmBuffer::shm_nattch(int shmid) {
	shmid_ds shmstat;
	shmctl(shmid, IPC_STAT, &shmstat);
	return shmstat.shm_nattch;
}

//______________________________________________________________________________
int FUShmBuffer::shm_destroy(int shmid) {
	shmid_ds shmstat;
	int result = shmctl(shmid, IPC_RMID, &shmstat);
	if (result == -1)
		cout << "FUShmBuffer::shm_destroy(shmid=" << shmid << ") failed."
				<< endl;
	return result;
}

//______________________________________________________________________________
int FUShmBuffer::sem_create(key_t key, int nsem) {
	int semid = semget(key, nsem, IPC_CREAT | 0666);
	if (semid < 0) {
		int err = errno;
		cout << "FUShmBuffer::sem_create(key=" << key << ",nsem=" << nsem
				<< ") failed: " << strerror(err) << endl;
	}
	return semid;
}

//______________________________________________________________________________
int FUShmBuffer::sem_get(key_t key, int nsem) {
	int semid = semget(key, nsem, 0666);
	if (semid < 0) {
		int err = errno;
		cout << "FUShmBuffer::sem_get(key=" << key << ",nsem=" << nsem
				<< ") failed: " << strerror(err) << endl;
	}
	return semid;
}

//______________________________________________________________________________
int FUShmBuffer::sem_destroy(int semid) {
	int result = semctl(semid, 0, IPC_RMID);
	if (result == -1)
		cout << "FUShmBuffer::sem_destroy(semid=" << semid << ") failed."
				<< endl;
	return result;
}

////////////////////////////////////////////////////////////////////////////////
// implementation of private member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
unsigned int FUShmBuffer::nextIndex(unsigned int offset, unsigned int nCells,
		unsigned int& iNext) {
	lock();
	unsigned int* pindex = (unsigned int*) ((unsigned long) this + offset);
	pindex += iNext;
	iNext = (iNext + 1) % nCells;
	unsigned int result = *pindex;
	unlock();
	return result;
}

//______________________________________________________________________________
void FUShmBuffer::postIndex(unsigned int index, unsigned int offset,
		unsigned int nCells, unsigned int& iLast) {
	lock();
	unsigned int* pindex = (unsigned int*) ((unsigned long) this + offset);
	pindex += iLast;
	*pindex = index;
	iLast = (iLast + 1) % nCells;
	unlock();
}

//______________________________________________________________________________
unsigned int FUShmBuffer::nextRawWriteIndex() {
	return nextIndex(rawWriteOffset_, nRawCells_, rawWriteNext_);
}

//______________________________________________________________________________
unsigned int FUShmBuffer::nextRawReadIndex() {
	return nextIndex(rawReadOffset_, nRawCells_, rawReadNext_);
}

//______________________________________________________________________________
void FUShmBuffer::postRawIndexToWrite(unsigned int index) {
	postIndex(index, rawWriteOffset_, nRawCells_, rawWriteLast_);
}

//______________________________________________________________________________
void FUShmBuffer::postRawIndexToRead(unsigned int index) {
	postIndex(index, rawReadOffset_, nRawCells_, rawReadLast_);
}

//______________________________________________________________________________
unsigned int FUShmBuffer::nextRecoWriteIndex() {
	return nextIndex(recoWriteOffset_, nRecoCells_, recoWriteNext_);
}

//______________________________________________________________________________
unsigned int FUShmBuffer::nextRecoReadIndex() {
	return nextIndex(recoReadOffset_, nRecoCells_, recoReadNext_);
}

//______________________________________________________________________________
void FUShmBuffer::postRecoIndexToWrite(unsigned int index) {
	postIndex(index, recoWriteOffset_, nRecoCells_, recoWriteLast_);
}

//______________________________________________________________________________
void FUShmBuffer::postRecoIndexToRead(unsigned int index) {
	postIndex(index, recoReadOffset_, nRecoCells_, recoReadLast_);
}

//______________________________________________________________________________
unsigned int FUShmBuffer::nextDqmWriteIndex() {
	return nextIndex(dqmWriteOffset_, nDqmCells_, dqmWriteNext_);
}

//______________________________________________________________________________
unsigned int FUShmBuffer::nextDqmReadIndex() {
	return nextIndex(dqmReadOffset_, nDqmCells_, dqmReadNext_);
}

//______________________________________________________________________________
void FUShmBuffer::postDqmIndexToWrite(unsigned int index) {
	postIndex(index, dqmWriteOffset_, nDqmCells_, dqmWriteLast_);
}

//______________________________________________________________________________
void FUShmBuffer::postDqmIndexToRead(unsigned int index) {
	postIndex(index, dqmReadOffset_, nDqmCells_, dqmReadLast_);
}

//______________________________________________________________________________
unsigned int FUShmBuffer::indexForEvtNumber(unsigned int evtNumber) {
	unsigned int *pevt = (unsigned int*) ((unsigned long) this
			+ evtNumberOffset_);
	for (unsigned int i = 0; i < nRawCells_; i++) {
		if ((*pevt++) == evtNumber)
			return i;
	}
	XCEPT_ASSERT(false, evf::Exception, "This point should not be reached!");
	return 0xffffffff;
}

//______________________________________________________________________________
unsigned int FUShmBuffer::indexForEvtPrcId(pid_t prcid) {
	pid_t *pevt = (pid_t*) ((unsigned long) this + evtPrcIdOffset_);
	for (unsigned int i = 0; i < nRawCells_; i++) {
		if ((*pevt++) == prcid)
			return i;
	}
	XCEPT_ASSERT(false, evf::Exception, "This point should not be reached!");
	return 0xffffffff;
}

//______________________________________________________________________________
evt::State_t FUShmBuffer::evtState(unsigned int index) {
	stringstream details;
	details << "index<nRawCells_ assertion failed! Actual index is " << index;
	XCEPT_ASSERT(index < nRawCells_, evf::Exception, details.str());
	evt::State_t *pstate = (evt::State_t*) ((unsigned long) this
			+ evtStateOffset_);
	pstate += index;
	return *pstate;
}

//______________________________________________________________________________
dqm::State_t FUShmBuffer::dqmState(unsigned int index) {
	stringstream details;
	details << "index<nDqmCells_ assertion failed! Actual index is " << index;
	XCEPT_ASSERT(index < nDqmCells_, evf::Exception, details.str());
	dqm::State_t *pstate = (dqm::State_t*) ((unsigned long) this
			+ dqmStateOffset_);
	pstate += index;
	return *pstate;
}

//______________________________________________________________________________
unsigned int FUShmBuffer::evtNumber(unsigned int index) {
	stringstream details;
	details << "index<nRawCells_ assertion failed! Actual index is " << index;
	XCEPT_ASSERT(index < nRawCells_, evf::Exception, details.str());
	unsigned int *pevt = (unsigned int*) ((unsigned long) this
			+ evtNumberOffset_);
	pevt += index;
	return *pevt;
}

//______________________________________________________________________________
pid_t FUShmBuffer::evtPrcId(unsigned int index) {
	stringstream details;
	details << "index<nRawCells_ assertion failed! Actual index is " << index;
	XCEPT_ASSERT(index < nRawCells_, evf::Exception, details.str());
	pid_t *prcid = (pid_t*) ((unsigned long) this + evtPrcIdOffset_);
	prcid += index;
	return *prcid;
}

//______________________________________________________________________________
time_t FUShmBuffer::evtTimeStamp(unsigned int index) {
	stringstream details;
	details << "index<nRawCells_ assertion failed! Actual index is " << index;
	XCEPT_ASSERT(index < nRawCells_, evf::Exception, details.str());
	time_t *ptstmp = (time_t*) ((unsigned long) this + evtTimeStampOffset_);
	ptstmp += index;
	return *ptstmp;
}

//______________________________________________________________________________
pid_t FUShmBuffer::clientPrcId(unsigned int index) {
	stringstream details;
	details << "index<nClientsMax_ assertion failed! Actual index is " << index;
	XCEPT_ASSERT(index < nClientsMax_, evf::Exception, details.str());
	pid_t *prcid = (pid_t*) ((unsigned long) this + clientPrcIdOffset_);
	prcid += index;
	return *prcid;
}

//______________________________________________________________________________
bool FUShmBuffer::setEvtState(unsigned int index, evt::State_t state) {
	stringstream details;
	details << "index<nRawCells_ assertion failed! Actual index is " << index;
	XCEPT_ASSERT(index < nRawCells_, evf::Exception, details.str());
	evt::State_t *pstate = (evt::State_t*) ((unsigned long) this
			+ evtStateOffset_);
	pstate += index;
	lock();
	*pstate = state;
	unlock();
	return true;
}

//______________________________________________________________________________
bool FUShmBuffer::setDqmState(unsigned int index, dqm::State_t state) {
	stringstream details;
	details << "index<nDqmCells_ assertion failed! Actual index is " << index;
	XCEPT_ASSERT(index < nDqmCells_, evf::Exception, details.str());
	dqm::State_t *pstate = (dqm::State_t*) ((unsigned long) this
			+ dqmStateOffset_);
	pstate += index;
	lock();
	*pstate = state;
	unlock();
	return true;
}

//______________________________________________________________________________
bool FUShmBuffer::setEvtDiscard(unsigned int index, unsigned int discard,
		bool checkValue) {
	stringstream details;
	details << "index<nRawCells_ assertion failed! Actual index is " << index;
	XCEPT_ASSERT(index < nRawCells_, evf::Exception, details.str());
	unsigned int *pcount = (unsigned int*) ((unsigned long) this
			+ evtDiscardOffset_);
	pcount += index;
	lock();
	if (checkValue) {
		if (*pcount < discard)
			*pcount = discard;
	} else
		*pcount = discard;
	unlock();
	return true;
}

//______________________________________________________________________________
int FUShmBuffer::incEvtDiscard(unsigned int index) {
	int result = 0;
	stringstream details;
	details << "index<nRawCells_ assertion failed! Actual index is " << index;
	XCEPT_ASSERT(index < nRawCells_, evf::Exception, details.str());
	unsigned int *pcount = (unsigned int*) ((unsigned long) this
			+ evtDiscardOffset_);
	pcount += index;
	lock();
	(*pcount)++;
	result = *pcount;
	unlock();
	return result;
}

//______________________________________________________________________________
bool FUShmBuffer::setEvtNumber(unsigned int index, unsigned int evtNumber) {
	stringstream details;
	details << "index<nRawCells_ assertion failed! Actual index is " << index;
	XCEPT_ASSERT(index < nRawCells_, evf::Exception, details.str());
	unsigned int *pevt = (unsigned int*) ((unsigned long) this
			+ evtNumberOffset_);
	pevt += index;
	lock();
	*pevt = evtNumber;
	unlock();
	return true;
}

//______________________________________________________________________________
bool FUShmBuffer::setEvtPrcId(unsigned int index, pid_t prcId) {
	stringstream details;
	details << "index<nRawCells_ assertion failed! Actual index is " << index;
	XCEPT_ASSERT(index < nRawCells_, evf::Exception, details.str());
	pid_t* prcid = (pid_t*) ((unsigned long) this + evtPrcIdOffset_);
	prcid += index;
	lock();
	*prcid = prcId;
	unlock();
	return true;
}

//______________________________________________________________________________
bool FUShmBuffer::setEvtTimeStamp(unsigned int index, time_t timeStamp) {
	stringstream details;
	details << "index<nRawCells_ assertion failed! Actual index is " << index;
	XCEPT_ASSERT(index < nRawCells_, evf::Exception, details.str());
	time_t *ptstmp = (time_t*) ((unsigned long) this + evtTimeStampOffset_);
	ptstmp += index;
	lock();
	*ptstmp = timeStamp;
	unlock();
	return true;
}

//______________________________________________________________________________
bool FUShmBuffer::setClientPrcId(pid_t prcId) {
	lock();
	stringstream details;
	details << "nClients_<nClientsMax_ assertion failed! Actual nClients is "
			<< nClients_ << " and nClientsMax is " << nClientsMax_;
	XCEPT_ASSERT(nClients_ < nClientsMax_, evf::Exception, details.str());
	pid_t *prcid = (pid_t*) ((unsigned long) this + clientPrcIdOffset_);
	for (unsigned int i = 0; i < nClients_; i++) {
		if ((*prcid) == prcId) {
			unlock();
			return false;
		}
		prcid++;
	}
	nClients_++;
	*prcid = prcId;
	unlock();
	return true;
}

//______________________________________________________________________________
bool FUShmBuffer::removeClientPrcId(pid_t prcId) {
	lock();
	pid_t *prcid = (pid_t*) ((unsigned long) this + clientPrcIdOffset_);
	unsigned int iClient(0);
	while (iClient < nClients_ && (*prcid) != prcId) {
		prcid++;
		iClient++;
	}
	if (iClient==nClients_) return false;
	//stringstream details;
	//details << "iClient!=nClients_ assertion failed! Actual iClient is "
	//		<< iClient;
	//XCEPT_ASSERT(iClient != nClients_, evf::Exception, details.str());
	pid_t* next = prcid;
	next++;
	while (iClient < nClients_ - 1) {
		*prcid = *next;
		prcid++;
		next++;
		iClient++;
	}
	nClients_--;
	unlock();
	return true;
}

//______________________________________________________________________________
FUShmRawCell* FUShmBuffer::rawCell(unsigned int iCell) {
	FUShmRawCell* result(0);

	if (iCell >= nRawCells_) {
		cout << "FUShmBuffer::rawCell(" << iCell << ") ERROR: " << "iCell="
				<< iCell << " >= nRawCells()=" << nRawCells_ << endl;
		return result;
	}

	if (segmentationMode_) {
		key_t shmkey = rawCellShmKey(iCell);
		int shmid = shm_get(shmkey, rawCellTotalSize_);
		void* cellAddr = shm_attach(shmid);
		result = new (cellAddr) FUShmRawCell(rawCellPayloadSize_);
	} else {
		result = (FUShmRawCell*) ((unsigned long) this + rawCellOffset_ + iCell
				* rawCellTotalSize_);
	}

	return result;
}

//______________________________________________________________________________
FUShmRecoCell* FUShmBuffer::recoCell(unsigned int iCell) {
	FUShmRecoCell* result(0);

	if (iCell >= nRecoCells_) {
		cout << "FUShmBuffer::recoCell(" << iCell << ") ERROR: " << "iCell="
				<< iCell << " >= nRecoCells=" << nRecoCells_ << endl;
		return result;
	}

	if (segmentationMode_) {
		key_t shmkey = recoCellShmKey(iCell);
		int shmid = shm_get(shmkey, recoCellTotalSize_);
		void* cellAddr = shm_attach(shmid);
		result = new (cellAddr) FUShmRecoCell(recoCellPayloadSize_);
	} else {
		result = (FUShmRecoCell*) ((unsigned long) this + recoCellOffset_
				+ iCell * recoCellTotalSize_);
	}

	return result;
}

//______________________________________________________________________________
FUShmDqmCell* FUShmBuffer::dqmCell(unsigned int iCell) {
	FUShmDqmCell* result(0);

	if (iCell >= nDqmCells_) {
		cout << "FUShmBuffer::dqmCell(" << iCell << ") ERROR: " << "iCell="
				<< iCell << " >= nDqmCells=" << nDqmCells_ << endl;
		return result;
	}

	if (segmentationMode_) {
		key_t shmkey = dqmCellShmKey(iCell);
		int shmid = shm_get(shmkey, dqmCellTotalSize_);
		void* cellAddr = shm_attach(shmid);
		result = new (cellAddr) FUShmDqmCell(dqmCellPayloadSize_);
	} else {
		result = (FUShmDqmCell*) ((unsigned long) this + dqmCellOffset_ + iCell
				* dqmCellTotalSize_);
	}

	return result;
}

//______________________________________________________________________________
bool FUShmBuffer::rawCellReadyForDiscard(unsigned int index) {
	stringstream details;
	details << "index<nRawCells_ assertion failed! Actual index is " << index;
	XCEPT_ASSERT(index < nRawCells_, evf::Exception, details.str());
	unsigned int *pcount = (unsigned int*) ((unsigned long) this
			+ evtDiscardOffset_);
	pcount += index;
	lock();
	stringstream details2;
	details2 << "*pcount>0 assertion failed! Value at pcount is " << *pcount << " for cell index " << index;
	XCEPT_ASSERT(*pcount > 0, evf::Exception, details2.str());
	--(*pcount);
	bool result = (*pcount == 0);
	unlock();
	return result;
}

//______________________________________________________________________________
key_t FUShmBuffer::shmKey(unsigned int iCell, unsigned int offset) {
	if (!segmentationMode_) {
		cout << "FUShmBuffer::shmKey() ERROR: only valid in segmentationMode!"
				<< endl;
		return -1;
	}
	key_t* addr = (key_t*) ((unsigned long) this + offset);
	for (unsigned int i = 0; i < iCell; i++)
		++addr;
	return *addr;
}

//______________________________________________________________________________
key_t FUShmBuffer::rawCellShmKey(unsigned int iCell) {
	if (iCell >= nRawCells_) {
		cout << "FUShmBuffer::rawCellShmKey() ERROR: " << "iCell>=nRawCells: "
				<< iCell << ">=" << nRawCells_ << endl;
		return -1;
	}
	return shmKey(iCell, rawCellOffset_);
}

//______________________________________________________________________________
key_t FUShmBuffer::recoCellShmKey(unsigned int iCell) {
	if (iCell >= nRecoCells_) {
		cout << "FUShmBuffer::recoCellShmKey() ERROR: "
				<< "iCell>=nRecoCells: " << iCell << ">=" << nRecoCells_
				<< endl;
		return -1;
	}
	return shmKey(iCell, recoCellOffset_);
}

//______________________________________________________________________________
key_t FUShmBuffer::dqmCellShmKey(unsigned int iCell) {
	if (iCell >= nDqmCells_) {
		cout << "FUShmBuffer::dqmCellShmKey() ERROR: " << "iCell>=nDqmCells: "
				<< iCell << ">=" << nDqmCells_ << endl;
		return -1;
	}
	return shmKey(iCell, dqmCellOffset_);
}

//______________________________________________________________________________
void FUShmBuffer::sem_init(int isem, int value) {
	if (semctl(semid(), isem, SETVAL, value) < 0) {
		cout << "FUShmBuffer: FATAL ERROR in semaphore initialization." << endl;
	}
}

//______________________________________________________________________________
int FUShmBuffer::sem_wait(int isem) {
	struct sembuf sops[1];
	sops[0].sem_num = isem;
	sops[0].sem_op = -1;
	sops[0].sem_flg = 0;
	if (semop(semid(), sops, 1) == -1) {
		cout << "FUShmBuffer: ERROR in semaphore operation sem_wait." << endl;
		return -1;
	}
	return 0;
}

//______________________________________________________________________________
void FUShmBuffer::sem_post(int isem) {
	struct sembuf sops[1];
	sops[0].sem_num = isem;
	sops[0].sem_op = 1;
	sops[0].sem_flg = 0;
	if (semop(semid(), sops, 1) == -1) {
		cout << "FUShmBuffer: ERROR in semaphore operation sem_post." << endl;
	}
}
