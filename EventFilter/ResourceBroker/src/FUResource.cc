////////////////////////////////////////////////////////////////////////////////
//
// FUResource
// ----------
//
//            12/10/2006 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
//            20/01/2012 Andrei Spataru <aspataru@cern.ch>
////////////////////////////////////////////////////////////////////////////////

#include "EventFilter/ResourceBroker/interface/FUResource.h"
#include "EventFilter/ResourceBroker/interface/ResourceChecker.h"

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "interface/shared/i2oXFunctionCodes.h"
#include "interface/evb/i2oEVBMsgs.h"
#include "EvffedFillerRB.h"
#include "toolbox/mem/Reference.h"
#include "xcept/tools.h"

#include <sstream>
#include <sys/shm.h>

using namespace std;
using namespace evf;

//#define DEBUG_FU_RES


////////////////////////////////////////////////////////////////////////////////
// initialize static members
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
bool FUResource::doFedIdCheck_ = true;
bool FUResource::useEvmBoard_ = true;
unsigned int FUResource::gtpEvmId_ = FEDNumbering::MINTriggerGTPFEDID;
unsigned int FUResource::gtpDaqId_ = FEDNumbering::MAXTriggerGTPFEDID;
unsigned int FUResource::gtpeId_ = FEDNumbering::MINTriggerEGTPFEDID;

////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
FUResource::FUResource(UInt_t fuResourceId, log4cplus::Logger logger,
		EvffedFillerRB *frb, xdaq::Application *app) :
	log_(logger), fuResourceId_(fuResourceId), superFragHead_(0),
			superFragTail_(0), nbBytes_(0), superFragSize_(0), shmCell_(0),
			frb_(frb), app_(app), nextEventWillHaveCRCError_(false) {
	//release();
}

//______________________________________________________________________________
FUResource::~FUResource() {

}

////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void FUResource::allocate(FUShmRawCell* shmCell) {
	//release();
	shmCell_ = shmCell;
	shmCell_->clear();
	shmCell_->setFuResourceId(fuResourceId_);
	//UPDATED
	shmCell_->setEventTypeData();
	eventPayloadSize_ = shmCell_->payloadSize();
	nFedMax_ = shmCell_->nFed();
	nSuperFragMax_ = shmCell_->nSuperFrag();
	/*
	 cout << "shmCell = " << shmCell_ << " shm cell furesourceId = "
	 << fuResourceId_ << " payload size = " << shmCell_->payloadSize()
	 << " nFed max = " << shmCell_->nFed() << " nSuperFragMax_ = "
	 << shmCell_->nSuperFrag() << endl;
	 */
}

//______________________________________________________________________________
void FUResource::release(bool detachResource) {
	doCrcCheck_ = false;
	fatalError_ = false;

	buResourceId_ = 0xffffffff;
	evtNumber_ = 0xffffffff;

	if (0 != superFragHead_) {
		try {
			superFragHead_->release();
		} catch (xcept::Exception& e) {
			LOG4CPLUS_ERROR(
					log_,
					"Failed to release superFragHead: "
							<< xcept::stdformat_exception_history(e));
		}
	}

	superFragHead_ = 0;
	superFragTail_ = 0;

	iBlock_ = 0;
	nBlock_ = 0xffffffff;
	iSuperFrag_ = 0;
	nSuperFrag_ = 0xffffffff;

	nbSent_ = 0;

	nbErrors_ = 0;
	nbCrcErrors_ = 0;

	for (UInt_t i = 0; i < 1024; i++)
		fedSize_[i] = 0;
	eventSize_ = 0;

	if (0 != shmCell_) {
		shmCell_ = 0;
		if (detachResource)
			shmdt(shmCell_);
	}

}

//______________________________________________________________________________
void FUResource::process(MemRef_t* bufRef) {
	if (fatalError()) {
		LOG4CPLUS_ERROR(log_, "THIS SHOULD *NEVER* HAPPEN!."); // DEBUG
		bufRef->release();
		return;
	}
#ifdef DEBUG_FU_RES
	std::cout << "Started process() for bufRef: " << bufRef<< std::endl;
#endif
	MemRef_t* itBufRef = bufRef;
	while (0 != itBufRef && !fatalError()) {
		MemRef_t* next = itBufRef->getNextReference();
		itBufRef->setNextReference(0);
		try {

			ResourceChecker resCheck(this);
			resCheck.processDataBlock(itBufRef);

		} catch (xcept::Exception& e) {
			LOG4CPLUS_ERROR(log_,
					"EVENT LOST:" << xcept::stdformat_exception_history(e));
			fatalError_ = true;
			itBufRef->setNextReference(next);
		}

		itBufRef = next;
	}
	if (isComplete()) {
		frb_->putHeader(evtNumber_, 0);
		frb_->putTrailer();
		fedSize_[frb_->fedId()] = frb_->size();
		UChar_t *startPos = shmCell_->writeData(frb_->getPayload(),
				frb_->size());
		superFragSize_ = frb_->size();
		if (!shmCell_->markSuperFrag(iSuperFrag_, superFragSize_, startPos)) {
			nbErrors_++;
			stringstream oss;
			oss << "Failed to mark super fragment in shared mem buffer."
					<< " fuResourceId:" << fuResourceId_ << " evtNumber:"
					<< evtNumber_ << " iSuperFrag:" << iSuperFrag_;
			XCEPT_RAISE(evf::Exception, oss.str());
		}

		if (!shmCell_->markFed(frb_->fedId(), frb_->size(), startPos)) {
			nbErrors_++;
			stringstream oss;
			oss << "Failed to mark fed in buffer." << " evtNumber:"
					<< evtNumber_ << " fedId:" << frb_->fedId() << " fedSize:"
					<< frb_->size() << " fedAddr:0x" << hex
					<< (unsigned long) frb_->getPayload() << dec;
			XCEPT_RAISE(evf::Exception, oss.str());
		}

	}
	return;
}

//______________________________________________________________________________
void FUResource::appendBlockToSuperFrag(MemRef_t* bufRef) {
	if (0 == superFragHead_) {
		superFragHead_ = bufRef;
		superFragTail_ = bufRef;
	} else {
		superFragTail_->setNextReference(bufRef);
		superFragTail_ = bufRef;
	}
	return;
}

//______________________________________________________________________________
void FUResource::removeLastAppendedBlockFromSuperFrag() {
	if (0 == superFragHead_) {
		//nothing to do... why did we get here then ???
	} else if (superFragHead_ == superFragTail_) {
		superFragHead_ = 0;
		superFragTail_ = 0;
	} else {
		MemRef_t *next = 0;
		MemRef_t *current = superFragHead_;
		while ((next = current->getNextReference()) != superFragTail_) {
			current = next;
			//get to the next-to-last block
		}
		superFragTail_ = current;
		current->setNextReference(0);
	}
	return;
}

//______________________________________________________________________________
void FUResource::superFragSize() throw (evf::Exception) {
	UChar_t *blockAddr = 0;
	UChar_t *frlHeaderAddr = 0;
	frlh_t *frlHeader = 0;

	superFragSize_ = 0;

	UInt_t frameSize = sizeof(I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME);
	MemRef_t* bufRef = superFragHead_;

	while (0 != bufRef) {
		blockAddr = (UChar_t*) bufRef->getDataLocation();
		frlHeaderAddr = blockAddr + frameSize;
		frlHeader = (frlh_t*) frlHeaderAddr;
		superFragSize_ += frlHeader->segsize & FRL_SEGSIZE_MASK;
		bufRef = bufRef->getNextReference();
	}

	eventSize_ += superFragSize_;

	if (eventSize_ > eventPayloadSize_) {
		nbErrors_++;
		stringstream oss;
		oss << "Event size exceeds maximum size." << " fuResourceId:"
				<< fuResourceId_ << " evtNumber:" << evtNumber_
				<< " iSuperFrag:" << iSuperFrag_ << " eventSize:" << eventSize_
				<< " eventPayloadSize:" << eventPayloadSize_;
		XCEPT_RAISE(evf::Exception, oss.str());
	}
}

//______________________________________________________________________________
void FUResource::fillSuperFragPayload() throw (evf::Exception) {
	UChar_t *blockAddr = 0;
	UChar_t *frlHeaderAddr = 0;
	UChar_t *fedAddr = 0;
	UInt_t nbBytes = 0;
	UInt_t nbBytesTot = 0;
	frlh_t *frlHeader = 0;
	UChar_t *bufferPos = 0;
	UChar_t *startPos = 0;

	MemRef_t* bufRef = superFragHead_;
	while (bufRef != 0) {
		blockAddr = (UChar_t*) bufRef->getDataLocation();
		frlHeaderAddr = blockAddr + sizeof(I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME);
		fedAddr = frlHeaderAddr + sizeof(frlh_t);
		frlHeader = (frlh_t*) frlHeaderAddr;
		nbBytes = frlHeader->segsize & FRL_SEGSIZE_MASK;
		nbBytesTot += nbBytes;

		// check if still within limits
		if (nbBytesTot > superFragSize_) {
			nbErrors_++;
			stringstream oss;
			oss << "Reached end of buffer." << " fuResourceId:"
					<< fuResourceId_ << " evtNumber:" << evtNumber_
					<< " iSuperFrag:" << iSuperFrag_;
			XCEPT_RAISE(evf::Exception, oss.str());
		}

		bufferPos = shmCell_->writeData(fedAddr, nbBytes);
		if (0 == startPos)
			startPos = bufferPos;

		nbBytes_ += nbBytes;
		bufRef = bufRef->getNextReference();
	}

	if (!shmCell_->markSuperFrag(iSuperFrag_, superFragSize_, startPos)) {
		nbErrors_++;
		stringstream oss;
		oss << "Failed to mark super fragment in shared mem buffer."
				<< " fuResourceId:" << fuResourceId_ << " evtNumber:"
				<< evtNumber_ << " iSuperFrag:" << iSuperFrag_;
		XCEPT_RAISE(evf::Exception, oss.str());
	}

	return;
}

//______________________________________________________________________________
void FUResource::releaseSuperFrag() {
	if (0 == superFragHead_)
		return;
	superFragHead_->release(); // throws xcept::Exception
	superFragHead_ = 0;
	superFragTail_ = 0;
	return;
}

//______________________________________________________________________________
UInt_t FUResource::nbErrors(bool reset) {
	UInt_t result = nbErrors_;
	if (reset)
		nbErrors_ = 0;
	return result;
}

//______________________________________________________________________________
UInt_t FUResource::nbCrcErrors(bool reset) {
	UInt_t result = nbCrcErrors_;
	if (reset)
		nbCrcErrors_ = 0;
	return result;
}

//______________________________________________________________________________
UInt_t FUResource::nbBytes(bool reset) {
	UInt_t result = nbBytes_;
	if (reset)
		nbBytes_ = 0;
	return result;
}
