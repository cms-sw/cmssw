////////////////////////////////////////////////////////////////////////////////
//
// FUResourceTable
// ---------------
//
//            12/10/2006 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////


#include "EventFilter/ResourceBroker/interface/FUResourceTable.h"

#include "interface/evb/include/i2oEVBMsgs.h"


using namespace evf;
using namespace std;


////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
FUResourceTable::FUResourceTable(UInt_t nbResources,bool shmMode,
				 log4cplus::Logger logger)
  : log_(logger)
  , shmMode_(shmMode)
  , shmBuffer_(0)
  , doCrcCheck_(1)
  , lock_(BSem::FULL)
{
  initialize(nbResources);
}


//______________________________________________________________________________
FUResourceTable::~FUResourceTable()
{
  clear();
  if (FUShmBuffer::releaseSharedMemory())
    LOG4CPLUS_INFO(log_,"Shared Memory segments cleaned up successfully.");
}


////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void FUResourceTable::initialize(UInt_t nbResources)
{
  clear();
  
  if (shmMode_) {
    shmBuffer_=FUShmBuffer::createShmBuffer(nbResources);
    if (0==shmBuffer_) {
      LOG4CPLUS_WARN(log_,"creation of shared memory segment failed.");
    }
    else {
      for (UInt_t i=0;i<nbResources;i++) {
	resources_.push_back(new FUResource(shmBuffer_->cell(i),log_));
      }
    }
  }
  else {
    for (UInt_t i=0;i<nbResources;i++) {
      resources_.push_back(new FUResource(i,log_));
      freeResourceIds_.push(i);
    }
    sem_init(&writeSem_,0,nbResources);
    sem_init(&readSem_, 0,          0);
  }
  
  return;
}


//______________________________________________________________________________
void FUResourceTable::clear()
{
  for (UInt_t i=0;i<resources_.size();i++) delete resources_[i];
  
  resources_.clear();
  while (!freeResourceIds_.empty()) freeResourceIds_.pop();
  builtResourceIds_.clear();
  buIdsToBeDiscarded_.clear();
}


//______________________________________________________________________________
void FUResourceTable::resetCounters()
{
  nbAllocated_=0;
  nbPending_  =0;
  nbCompleted_=0;
  nbRequested_=0;
  nbLost_     =0;

  nbErrors_   =0;
  nbCrcErrors_=0;
  nbBytes_    =0;
}


//______________________________________________________________________________
UInt_t FUResourceTable::allocateResource()
{
  waitWriterSem();
  
  // determine fuResourceId for next available resource
  UInt_t fuResourceId;

  // shared memory mode: discard/dump at this stage!
  if (shmMode_) {
    shmBuffer_->lock();
    FUShmBufferCell* cell=shmBuffer_->currentWriterCell();
    fuResourceId=cell->fuResourceId();
    if (!cell->isEmpty()) {
      if (cell->isProcessed()) {
	lock();
	buIdsToBeDiscarded_.push_back(cell->buResourceId());
	nbRequested_++;
	unlock();
	resources_[fuResourceId]->release();
      }
      else {
	LOG4CPLUS_DEBUG(log_,"DROP EVENT: evtNumber="<<cell->evtNumber());
	//cell->dump(); //TODO
      }
    }
    shmBuffer_->unlock();
  }
  // standard mode: use queue to manage free resources
  else {
    fuResourceId=freeResourceIds_.front();
    freeResourceIds_.pop();
  }

  // initialize the new resource and update the relevant counters
  resources_[fuResourceId]->allocate();
  nbPending_++;
  nbAllocated_++;
  
  // set the resource to check the crc for each fed if requested
  if (0==nbAllocated_%doCrcCheck_)
    resources_[fuResourceId]->doCrcCheck(true);
  
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
    nbBytes_    +=resource->nbBytes();
    nbErrors_   +=resource->nbErrors();
    nbCrcErrors_+=resource->nbCrcErrors();
    unlock();
    
    // make resource available for pick-up
    if (resource->isComplete()) {
      if (shmMode_) {
	shmBuffer_->cell(fuResourceId)->setStateWritten();
      }
      else {
	resource->fillFEDs();
	builtResourceIds_.push_front(fuResourceId);
      }
      
      lock();
      nbCompleted_++;
      nbPending_--;
      unlock();

      postReaderSem();

      eventComplete=true;
    }
    
  }
  // bad event, release msg, and the whole resource if this was the last one
  else {
    bool lastMsg=isLastMessageOfEvent(bufRef);
    bufRef->release();
    if (lastMsg) {
      if (!shmMode_) {
	delete resource->fedData();
	freeResourceIds_.push(fuResourceId);
      }
      resource->release();
      lock();
      buIdsToBeDiscarded_.push_back(buResourceId);
      nbLost_++;
      nbPending_--;
      unlock();
      postWriterSem();
    }
  }
  
  return eventComplete;
}


//______________________________________________________________________________
FEDRawDataCollection* FUResourceTable::requestResource(UInt_t& evtNumber,
						       UInt_t& buResourceId)
{
  waitReaderSem();
  
  lock();
  
  FEDRawDataCollection* result(0);
  
  if (shmMode_) {
    FUShmBufferCell* cell=shmBuffer_->currentReaderCell();
    cell->setStateProcessed();
  }
  else {
    UInt_t      fuResourceId=builtResourceIds_.back();
    FUResource* resource    =resources_[fuResourceId];
  
    assert(resource->isComplete());
    assert(!resource->fatalError());
  
    result      =resource->fedData();
    evtNumber   =resource->evtNumber();
    buResourceId=resource->buResourceId();
  
    resource->release();
    builtResourceIds_.pop_back();
    freeResourceIds_.push(fuResourceId);
    nbRequested_++;
  }
  
  unlock();
  
  postWriterSem();
  
  return result;
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


//______________________________________________________________________________
UInt_t FUResourceTable::nbShmClients() const
{
  UInt_t result(0);
  if (shmMode_&&0!=shmBuffer_)
    result=FUShmBuffer::shm_nattch(shmBuffer_->shmid())-1;
  return result;
}
