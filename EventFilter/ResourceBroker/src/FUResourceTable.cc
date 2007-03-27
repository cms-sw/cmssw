////////////////////////////////////////////////////////////////////////////////
//
// FUResourceTable
// ---------------
//
//            12/10/2006 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////


#include "EventFilter/ResourceBroker/interface/FUResourceTable.h"

#include "toolbox/include/toolbox/task/WorkLoopFactory.h"
#include "interface/evb/include/i2oEVBMsgs.h"
#include "xcept/include/xcept/tools.h"

#include <unistd.h>


using namespace evf;
using namespace std;


////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
FUResourceTable::FUResourceTable(bool              segmentationMode,
				 UInt_t            nbRawCells,
				 UInt_t            nbRecoCells,
				 UInt_t            nbDqmCells,
				 UInt_t            rawCellSize,
				 UInt_t            recoCellSize,
				 UInt_t            dqmCellSize,
				 BUProxy          *bu,
				 SMProxy          *sm,
				 log4cplus::Logger logger)
  : bu_(bu)
  , sm_(sm)
  , log_(logger)
  , wlSendData_(0)
  , asSendData_(0)
  , wlSendDqm_(0)
  , asSendDqm_(0)
  , wlDiscard_(0)
  , asDiscard_(0)
  , shmBuffer_(0)
  , doCrcCheck_(1)
  , nbClientsToShutDown_(0)
  , isReadyToShutDown_(true)
  , lock_(BSem::FULL)
{
  initialize(segmentationMode,
	     nbRawCells,nbRecoCells,nbDqmCells,
	     rawCellSize,recoCellSize,dqmCellSize);
}


//______________________________________________________________________________
FUResourceTable::~FUResourceTable()
{
  clear();
  shmdt(shmBuffer_);
  if (FUShmBuffer::releaseSharedMemory())
    LOG4CPLUS_INFO(log_,"SHARED MEMORY SUCCESSFULLY RELEASED.");
}


////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void FUResourceTable::initialize(bool   segmentationMode,
				 UInt_t nbRawCells,
				 UInt_t nbRecoCells,
				 UInt_t nbDqmCells,
				 UInt_t rawCellSize,
				 UInt_t recoCellSize,
				 UInt_t dqmCellSize)
{
  clear();
  
  shmBuffer_=FUShmBuffer::createShmBuffer(segmentationMode,
					  nbRawCells,nbRecoCells,nbDqmCells,
					  rawCellSize,recoCellSize,dqmCellSize);
  if (0==shmBuffer_) {
    LOG4CPLUS_FATAL(log_,"CREATION OF SHARED MEMORY SEGMENT FAILED!");
    return;
  }
  
  for (UInt_t i=0;i<nbRawCells;i++) resources_.push_back(new FUResource(log_));
}


//______________________________________________________________________________
void FUResourceTable::startSendDataWorkLoop() throw (evf::Exception)
{
  try {
    wlSendData_=
      toolbox::task::getWorkLoopFactory()->getWorkLoop("SendData","waiting");
    if (!wlSendData_->isActive()) wlSendData_->activate();
    asSendData_=toolbox::task::bind(this,&FUResourceTable::sendData,"SendData");
    wlSendData_->submit(asSendData_);
  }
  catch (xcept::Exception& e) {
    string msg = "Failed to start workloop 'SendData'.";
    XCEPT_RETHROW(evf::Exception,msg,e);
  }
}


//______________________________________________________________________________
bool FUResourceTable::sendData(toolbox::task::WorkLoop* /* wl */)
{
  bool reschedule=true;

  FUShmRecoCell* cell=shmBuffer_->recoCellToRead();
  nbAccepted_++;
  
  if (0==cell->eventSize()) {
    LOG4CPLUS_WARN(log_,"Don't reschedule sendData workloop.");
    shmBuffer_->discardRecoCell(cell->index());
    reschedule=false;
  }
  else {
    try {
      if (cell->type()==0) {
	sendInitMessage(cell->index(),cell->payloadAddr(),cell->eventSize());
      }
      else if (cell->type()==1) {
	sendDataEvent(cell->index(),cell->runNumber(),cell->evtNumber(),
		      cell->payloadAddr(),cell->eventSize());
      }
      else {
	string errmsg="Unknown RecoCell type (neither DATA nor INIT).";
	XCEPT_RAISE(evf::Exception,errmsg);
      }
    }
    catch (xcept::Exception& e) {
      LOG4CPLUS_FATAL(log_,"Failed to send EVENT DATA to StorageManager: "
		      <<xcept::stdformat_exception_history(e));
      reschedule=false;
    }
  }
  
  shmBuffer_->finishReadingRecoCell(cell);
  
  return reschedule;
}


//______________________________________________________________________________
void FUResourceTable::startSendDqmWorkLoop() throw (evf::Exception)
{
  try {
    wlSendDqm_=toolbox::task::getWorkLoopFactory()->getWorkLoop("SendDqm","waiting");
    if (!wlSendDqm_->isActive()) wlSendDqm_->activate();
    asSendDqm_=toolbox::task::bind(this,&FUResourceTable::sendDqm,"SendDqm");
    wlSendDqm_->submit(asSendDqm_);
  }
  catch (xcept::Exception& e) {
    string msg = "Failed to start workloop 'SendDqm'.";
    XCEPT_RETHROW(evf::Exception,msg,e);
  }
}


//______________________________________________________________________________
bool FUResourceTable::sendDqm(toolbox::task::WorkLoop* /* wl */)
{
  bool reschedule=true;
  
  FUShmDqmCell* cell=shmBuffer_->dqmCellToRead();
  dqm::State_t  state=shmBuffer_->dqmState(cell->index());
  
  if (state==dqm::EMPTY) {
    LOG4CPLUS_WARN(log_,"Don't reschedule sendDqm workloop.");
    shmBuffer_->discardDqmCell(cell->index());
    reschedule=false;
  }
  else {
    try {
      sendDqmEvent(cell->index(),
		   cell->runNumber(),cell->evtAtUpdate(),cell->folderId(),
		   cell->payloadAddr(),cell->eventSize());
    }
    catch (xcept::Exception& e) {
      LOG4CPLUS_FATAL(log_,"Failed to send DQM DATA to StorageManager: "
		      <<xcept::stdformat_exception_history(e));
      reschedule=false;
    }
  }
  
  shmBuffer_->finishReadingDqmCell(cell);
  
  return reschedule;
}


//______________________________________________________________________________
void FUResourceTable::startDiscardWorkLoop() throw (evf::Exception)
{
  try {
    wlDiscard_=toolbox::task::getWorkLoopFactory()->getWorkLoop("Discard","waiting");
    if (!wlDiscard_->isActive()) wlDiscard_->activate();
    asDiscard_=toolbox::task::bind(this,&FUResourceTable::discard,"Discard");
    wlDiscard_->submit(asDiscard_);
  }
  catch (xcept::Exception& e) {
    string msg = "Failed to start workloop 'Discard'.";
    XCEPT_RETHROW(evf::Exception,msg,e);
  }
  isReadyToShutDown_=false;
}


//______________________________________________________________________________
bool FUResourceTable::discard(toolbox::task::WorkLoop* /* wl */)
{
  FUShmRawCell* cell =shmBuffer_->rawCellToDiscard();
  evt::State_t  state=shmBuffer_->evtState(cell->index());

  bool   reschedule  =true;
  bool   shutDown    =(state==evt::EMPTY);
  UInt_t fuResourceId=cell->fuResourceId();
  UInt_t buResourceId=cell->buResourceId();

  if (shutDown) {
    LOG4CPLUS_WARN(log_,"nbClientsToShutDown = "<<nbClientsToShutDown_);
    if (nbClientsToShutDown_>0) --nbClientsToShutDown_;
    if (nbClientsToShutDown_==0) {
      LOG4CPLUS_WARN(log_,"Don't reschedule discard-workloop.");
      reschedule = false;
    }
  }
  
  resources_[fuResourceId]->release();
  shmBuffer_->discardRawCell(cell);
  if (!shutDown) {
    sendDiscard(buResourceId);
    sendAllocate();
  }
  
  if (!reschedule) {
    shmBuffer_->writeRecoEmptyEvent();
    shmBuffer_->writeDqmEmptyEvent();
    isReadyToShutDown_ = true;
  }
  
  return reschedule;
}



//______________________________________________________________________________
UInt_t FUResourceTable::allocateResource()
{
  FUShmRawCell* cell=shmBuffer_->rawCellToWrite();
  UInt_t fuResourceId=cell->fuResourceId();
  
  resources_[fuResourceId]->allocate(cell);
  nbPending_++;
  nbAllocated_++;
  
  if (0==nbAllocated_%doCrcCheck_) resources_[fuResourceId]->doCrcCheck(true);
  
  return fuResourceId;
}


//______________________________________________________________________________
bool FUResourceTable::buildResource(MemRef_t* bufRef)
{
  bool eventComplete=false;
  
  I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME *block=
    (I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME*)bufRef->getDataLocation();
  
  UInt_t      fuResourceId=(UInt_t)block->fuTransactionId;
  UInt_t      buResourceId=(UInt_t)block->buResourceId;
  FUResource* resource    =resources_[fuResourceId];
  
  // keep building this resource if it is healthy
  if (!resource->fatalError()) {
    resource->process(bufRef);
    lock();
    nbErrors_         +=resource->nbErrors();
    nbCrcErrors_      +=resource->nbCrcErrors();
    unlock();
    
    // make resource available for pick-up
    if (resource->isComplete()) {
      lock();
      UInt_t evtSize     =resource->shmCell()->eventSize();
      inputSumOfSquares_+=(uint64_t)evtSize*(uint64_t)evtSize;
      inputSumOfSizes_  +=evtSize;
      nbCompleted_++;
      nbPending_--;
      unlock();
      shmBuffer_->finishWritingRawCell(resource->shmCell());
      eventComplete=true;
    }
    
  }
  // bad event, release msg, and the whole resource if this was the last one
  else {
    bool lastMsg=isLastMessageOfEvent(bufRef);
    bufRef->release();
    if (lastMsg) {
      lock();
      bu_->sendDiscard(buResourceId);
      nbDiscarded_++;
      nbLost_++;
      nbPending_--;
      unlock();
      shmBuffer_->releaseRawCell(resource->shmCell());
    }
  }
  
  return eventComplete;
}


//______________________________________________________________________________
bool FUResourceTable::discardDataEvent(MemRef_t* bufRef)
{
  I2O_FU_DATA_DISCARD_MESSAGE_FRAME *msg;
  msg=(I2O_FU_DATA_DISCARD_MESSAGE_FRAME*)bufRef->getDataLocation();
  UInt_t recoIndex=msg->fuID;
  shmBuffer_->discardRecoCell(recoIndex);
  bufRef->release();
  return true;
}


//______________________________________________________________________________
bool FUResourceTable::discardDqmEvent(MemRef_t* bufRef)
{
  I2O_FU_DQM_DISCARD_MESSAGE_FRAME *msg;
  msg=(I2O_FU_DQM_DISCARD_MESSAGE_FRAME*)bufRef->getDataLocation();
  UInt_t dqmIndex=msg->fuID;
  shmBuffer_->discardDqmCell(dqmIndex);
  bufRef->release();
  return true;
}


//______________________________________________________________________________
void FUResourceTable::dropEvent()
{
  FUShmRawCell* cell=shmBuffer_->rawCellToRead();
  UInt_t fuResourceId=cell->fuResourceId();
  shmBuffer_->finishReadingRawCell(cell);
  shmBuffer_->scheduleRawCellForDiscard(fuResourceId);
}


//______________________________________________________________________________
void FUResourceTable::shutDownClients()
{
  nbClientsToShutDown_ = nbShmClients();
  isReadyToShutDown_   = false;
  
  if (nbClientsToShutDown_==0) {
    shmBuffer_->scheduleRawEmptyCellForDiscard();
  }
  else {
    for (UInt_t i=0;i<nbClientsToShutDown_;++i) {
      shmBuffer_->writeRawEmptyEvent();
    }
  }
}


//______________________________________________________________________________
void FUResourceTable::clear()
{
  for (UInt_t i=0;i<resources_.size();i++) {
    resources_[i]->release();
    delete resources_[i];
  }
  resources_.clear();
}


//______________________________________________________________________________
void FUResourceTable::reset()
{
  shmBuffer_->reset();
  resetCounters();
}


//______________________________________________________________________________
void FUResourceTable::resetCounters()
{
  nbAllocated_       =0;
  nbPending_         =0;
  nbCompleted_       =0;
  nbProcessed_       =0;
  nbAccepted_        =0;
  nbSent_            =0;
  nbDiscarded_       =0;
  nbLost_            =0;

  nbErrors_          =0;
  nbCrcErrors_       =0;
  nbAllocSent_       =0;

  inputSumOfSquares_ =0;
  outputSumOfSquares_=0;
  inputSumOfSizes_   =0;
  outputSumOfSizes_  =0;
}


//______________________________________________________________________________
UInt_t FUResourceTable::nbShmClients() const
{
  UInt_t result(0);
  if (0!=shmBuffer_) result=shmBuffer_->nbClients();
  return result;
}



////////////////////////////////////////////////////////////////////////////////
// implementation of private member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void FUResourceTable::sendAllocate()
{
  UInt_t nbFreeSlots    = this->nbFreeSlots();
  UInt_t nbFreeSlotsMax = resources_.size()/2;
  if (nbFreeSlots>nbFreeSlotsMax) {
    UIntVec_t fuResourceIds;
    for (UInt_t i=0;i<nbFreeSlots;i++)
      fuResourceIds.push_back(allocateResource());
    bu_->sendAllocate(fuResourceIds);
    nbAllocSent_++;
  }
}


//______________________________________________________________________________
void FUResourceTable::sendDiscard(UInt_t buResourceId)
{
  bu_->sendDiscard(buResourceId);
  nbDiscarded_++;
  nbProcessed_++;
}


//______________________________________________________________________________
void FUResourceTable::sendInitMessage(UInt_t   fuResourceId,
				      UChar_t *data,
				      UInt_t   dataSize)
{
  UInt_t   nbBytes    =sm_->sendInitMessage(fuResourceId,data,dataSize);
  uint64_t nbBytesSq  =(uint64_t)nbBytes*(uint64_t)nbBytes;
  outputSumOfSquares_+=nbBytesSq;
  outputSumOfSizes_  +=nbBytes;
}


//______________________________________________________________________________
void FUResourceTable::sendDataEvent(UInt_t   fuResourceId,
				    UInt_t   runNumber,
				    UInt_t   evtNumber,
				    UChar_t *data,
				    UInt_t   dataSize)
{
  UInt_t   nbBytes    =sm_->sendDataEvent(fuResourceId,
					  runNumber,evtNumber,data,dataSize);
  uint64_t nbBytesSq  =(uint64_t)nbBytes*(uint64_t)nbBytes;
  outputSumOfSquares_+=nbBytesSq;
  outputSumOfSizes_  +=nbBytes;
  nbSent_++;
}


//______________________________________________________________________________
void FUResourceTable::sendDqmEvent(UInt_t   fuDqmId,
				   UInt_t   runNumber,
				   UInt_t   evtAtUpdate,
				   UInt_t   folderId,
				   UChar_t* data,
				   UInt_t   dataSize)
{
  lock();
  sm_->sendDqmEvent(fuDqmId,runNumber,evtAtUpdate,folderId,data,dataSize);
  nbSentDqm_++;
  unlock();
}


//______________________________________________________________________________
bool FUResourceTable::isLastMessageOfEvent(MemRef_t* bufRef)
{
  while (0!=bufRef->getNextReference()) bufRef=bufRef->getNextReference();
  I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME *block=
    (I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME*)bufRef->getDataLocation();
  
  UInt_t iBlock    =block->blockNb;
  UInt_t nBlock    =block->nbBlocksInSuperFragment;
  UInt_t iSuperFrag=block->superFragmentNb;
  UInt_t nSuperFrag=block->nbSuperFragmentsInEvent;

  return ((iSuperFrag==nSuperFrag-1)&&(iBlock==nBlock-1));
}
