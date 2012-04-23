////////////////////////////////////////////////////////////////////////////////
//
// IPCMethod.cc
// -------
//
// Contains common functionality for FUResourceTable and FUResourceQueue.
//
//  Created on: Oct 26, 2011
//      									Andrei Spataru : aspataru@cern.ch
////////////////////////////////////////////////////////////////////////////////

#include "EventFilter/ResourceBroker/interface/FUResourceTable.h"
#include "EventFilter/ResourceBroker/interface/IPCMethod.h"

#include "interface/evb/i2oEVBMsgs.h"

#include <iomanip>

using std::ofstream;
using std::endl;
using namespace evf;

////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
IPCMethod::IPCMethod(bool segmentationMode, UInt_t nbRawCells,
		UInt_t nbRecoCells, UInt_t nbDqmCells, UInt_t rawCellSize,
		UInt_t recoCellSize, UInt_t dqmCellSize, BUProxy *bu, SMProxy *sm,
		log4cplus::Logger logger, unsigned int timeout, EvffedFillerRB *frb,
		xdaq::Application*app) throw (evf::Exception) :
	bu_(bu), sm_(sm), log_(logger), nbDqmCells_(nbDqmCells),
			nbRawCells_(nbRawCells), nbRecoCells_(nbRecoCells),
			acceptSMDataDiscard_(0), acceptSMDqmDiscard_(0), doCrcCheck_(1),
			shutdownTimeout_(timeout), nbPending_(0), nbClientsToShutDown_(0),
			isReadyToShutDown_(true), isActive_(false), runNumber_(0xffffffff),
			frb_(frb), app_(app) {

	sem_init(&lock_, 0, 1);
}

IPCMethod::~IPCMethod() {

}

////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
UInt_t IPCMethod::allocateResource() {
	assert(!freeResourceIds_.empty());

	lock();
	UInt_t fuResourceId = freeResourceIds_.front();
	freeResourceIds_.pop();
	nbPending_++;
	nbAllocated_++;
	unlock();

	return fuResourceId;
}

//______________________________________________________________________________
void IPCMethod::dumpEvent(FUShmRawCell* cell) {
	std::ostringstream oss;
	oss << "/tmp/evt" << cell->evtNumber() << ".dump";
	ofstream fout(oss.str().c_str());
	fout.fill('0');

	fout << "#\n# evt " << cell->evtNumber() << "\n#\n" << endl;
	for (unsigned int i = 0; i < cell->nFed(); i++) {
		if (cell->fedSize(i) == 0)
			continue;
		fout << "# fedid " << i << endl;
		unsigned char* addr = cell->fedAddr(i);
		for (unsigned int j = 0; j < cell->fedSize(i); j++) {
			fout << std::setiosflags(std::ios::right) << std::setw(2)
					<< std::hex << (int) (*addr) << std::dec;
			if ((j + 1) % 8)
				fout << " ";
			else
				fout << endl;
			++addr;
		}
		fout << endl;
	}
	fout.close();
}

////////////////////////////////////////////////////////////////////////////////
// implementation of private member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void IPCMethod::sendAllocate() {
	UInt_t nbFreeSlots = this->nbFreeSlots();
	/*UInt_t nbFreeSlotsMax = 0*/ //reverting to larger chunk requests for BU
	UInt_t nbFreeSlotsMax = nbResources() / 2;
	if (nbFreeSlots > nbFreeSlotsMax) {
		UIntVec_t fuResourceIds;
		for (UInt_t i = 0; i < nbFreeSlots; i++)
			fuResourceIds.push_back(allocateResource());

		bu_->sendAllocate(fuResourceIds);

		nbAllocSent_++;
	}
}

//______________________________________________________________________________
void IPCMethod::resetPendingAllocates() {
	if (freeResourceIds_.size() < nbRawCells_) {
		LOG4CPLUS_INFO(
				log_,
				"There are " << nbRawCells_ - freeResourceIds_.size()
						<< " pending ALLOCATE messages! Forgetting...");
		while (!freeResourceIds_.empty())
			freeResourceIds_.pop();
		for (UInt_t i = 0; i < nbRawCells_; i++)
			freeResourceIds_.push(i);
	}
}

//______________________________________________________________________________
void IPCMethod::sendDiscard(UInt_t buResourceId) {
	bu_->sendDiscard(buResourceId);
	nbDiscarded_++;
}

//______________________________________________________________________________
void IPCMethod::sendInitMessage(UInt_t fuResourceId, UInt_t outModId,
		UInt_t fuProcessId, UInt_t fuGuid, UChar_t *data, UInt_t dataSize) {
	if (0 == sm_) {
		LOG4CPLUS_ERROR(log_, "No StorageManager, DROP INIT MESSAGE!");
	} else {
		acceptSMDataDiscard_[fuResourceId] = true;
		UInt_t nbBytes = sm_->sendInitMessage(fuResourceId, outModId,
				fuProcessId, fuGuid, data, dataSize);
		sumOfSquares_ += (uint64_t) nbBytes * (uint64_t) nbBytes;
		sumOfSizes_ += nbBytes;
	}
}

//______________________________________________________________________________
void IPCMethod::sendDataEvent(UInt_t fuResourceId, UInt_t runNumber,
		UInt_t evtNumber, UInt_t outModId, UInt_t fuProcessId, UInt_t fuGuid,
		UChar_t *data, UInt_t dataSize) {
	if (0 == sm_) {
		LOG4CPLUS_ERROR(log_, "No StorageManager, DROP DATA EVENT!");
	} else {
		acceptSMDataDiscard_[fuResourceId] = true;
		UInt_t nbBytes = sm_->sendDataEvent(fuResourceId, runNumber, evtNumber,
				outModId, fuProcessId, fuGuid, data, dataSize);
		sumOfSquares_ += (uint64_t) nbBytes * (uint64_t) nbBytes;
		sumOfSizes_ += nbBytes;
	}
}

//______________________________________________________________________________
void IPCMethod::sendErrorEvent(UInt_t fuResourceId, UInt_t runNumber,
		UInt_t evtNumber, UInt_t fuProcessId, UInt_t fuGuid, UChar_t *data,
		UInt_t dataSize) {
	if (0 == sm_) {
		LOG4CPLUS_ERROR(log_, "No StorageManager, DROP ERROR EVENT!");
	} else {
		acceptSMDataDiscard_[fuResourceId] = true;
		UInt_t nbBytes = sm_->sendErrorEvent(fuResourceId, runNumber,
				evtNumber, fuProcessId, fuGuid, data, dataSize);
		sumOfSquares_ += (uint64_t) nbBytes * (uint64_t) nbBytes;
		sumOfSizes_ += nbBytes;
	}

	//   if (0!=shmBuffer_) {
	//     UInt_t n=nbDqmCells_;

	//     for (UInt_t i=0;i<n;i++) {
	//       if(shmBuffer_->dqmCell(i)->fuProcessId()==fuProcessId)
	// 	{
	// 	  if(shmBuffer_->dqmState(i)!=dqm::SENT){
	// 	    shmBuffer_->setDqmState(i,dqm::SENT);
	// 	    shmBuffer_->discardDqmCell(i);
	// 	    acceptSMDqmDiscard_[i] = false;
	// 	  }
	// 	}
	//     }
	//     n=nbRecoCells_;
	//     for (UInt_t i=0;i<n;i++) {
	//       if(shmBuffer_->recoCell(i)->fuProcessId()==fuProcessId)
	// 	{
	// 	  shmBuffer_->discardOrphanedRecoCell(i);
	// 	}
	//     }

	//   }
}

//______________________________________________________________________________
void IPCMethod::sendDqmEvent(UInt_t fuDqmId, UInt_t runNumber,
		UInt_t evtAtUpdate, UInt_t folderId, UInt_t fuProcessId, UInt_t fuGuid,
		UChar_t* data, UInt_t dataSize) {
	if (0 == sm_) {
		LOG4CPLUS_WARN(log_, "No StorageManager, DROP DQM EVENT.");
	} else {
		sm_->sendDqmEvent(fuDqmId, runNumber, evtAtUpdate, folderId,
				fuProcessId, fuGuid, data, dataSize);

		nbPendingSMDqmDiscards_++;

		acceptSMDqmDiscard_[fuDqmId]++;
		if (acceptSMDqmDiscard_[fuDqmId] > 1)
			LOG4CPLUS_WARN(
					log_,
					"DQM Cell " << fuDqmId
							<< " being sent more than once for folder "
							<< folderId << " process " << fuProcessId
							<< " guid " << fuGuid);
		nbSentDqm_++;
	}
}

//______________________________________________________________________________
bool IPCMethod::isLastMessageOfEvent(MemRef_t* bufRef) {
	while (0 != bufRef->getNextReference())
		bufRef = bufRef->getNextReference();

	I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME *block =
			(I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME*) bufRef->getDataLocation();

	UInt_t iBlock = block->blockNb;
	UInt_t nBlock = block->nbBlocksInSuperFragment;
	UInt_t iSuperFrag = block->superFragmentNb;
	UInt_t nSuperFrag = block->nbSuperFragmentsInEvent;

	return ((iSuperFrag == nSuperFrag - 1) && (iBlock == nBlock - 1));
}

//______________________________________________________________________________
void IPCMethod::injectCRCError() {
	for (UInt_t i = 0; i < resources_.size(); i++) {
		resources_[i]->scheduleCRCError();
	}
}
