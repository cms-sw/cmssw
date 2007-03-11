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

#include <unistd.h>


using namespace evf;
using namespace std;


////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
FUResourceTable::FUResourceTable(UInt_t nbResources,UInt_t eventBufferSize,
				 BUProxy* bu,log4cplus::Logger logger)
  : bu_(bu)
  , log_(logger)
  , workLoopDiscard_(0)
  , asDiscard_(0)
  , shmBuffer_(0)
  , doCrcCheck_(1)
  , nbClientsToShutDown_(0)
  , isReadyToShutDown_(true)
  , lock_(BSem::FULL)
{
  initialize(nbResources,eventBufferSize);
}


//______________________________________________________________________________
FUResourceTable::~FUResourceTable()
{
  clear();
  if (FUShmBuffer::releaseSharedMemory())
    LOG4CPLUS_INFO(log_,"SHARED MEMORY SUCCESSFULLY RELEASED.");
}


////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
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
void FUResourceTable::dropEvent()
{
  waitReaderSem();
  FUShmBufferCell* cell=shmBuffer_->currentReaderCell();
  cell->setStateProcessed();
  shmBuffer_->scheduleForDiscard(cell);
}


//______________________________________________________________________________
void FUResourceTable::shutDownClients()
{
  nbClientsToShutDown_ = nbShmClients();
  isReadyToShutDown_   = false;
  
  if (nbClientsToShutDown_==0) {
    waitWriterSem();
    FUShmBufferCell* cell=shmBuffer_->currentWriterCell();
    if (cell->isDead()) {
      LOG4CPLUS_ERROR(log_,"LOST EVENT: evtNumber="<<cell->evtNumber());
    }
    else if (!cell->isEmpty()) {
      LOG4CPLUS_ERROR(log_,"cell "<<cell->index()<<" is in unexpected state!");
    }
    cell->setStateEmpty();
    shmBuffer_->scheduleForDiscard(cell);
  }
  else {
    for (unsigned int i=0;i<nbClientsToShutDown_;++i) {
      waitWriterSem();
      FUShmBufferCell* cell=shmBuffer_->currentWriterCell();
      if (cell->isDead()) {
	LOG4CPLUS_ERROR(log_,"LOST EVENT: evtNumber="<<cell->evtNumber());
      }
      else if (!cell->isEmpty()) {
	LOG4CPLUS_ERROR(log_,"cell "<<cell->index()<<" is in unexpected state!");
      }
      cell->setStateEmpty();
      postReaderSem();
    }
  }
}


//______________________________________________________________________________
void FUResourceTable::initialize(UInt_t nbResources,UInt_t eventBufferSize)
{
  clear();
  
  shmBuffer_=FUShmBuffer::createShmBuffer(nbResources,eventBufferSize);
  if (0==shmBuffer_) {
    LOG4CPLUS_FATAL(log_,"CREATION OF SHARED MEMORY SEGMENT FAILED!");
    return;
  }
  
  for (UInt_t i=0;i<nbResources;i++)
    resources_.push_back(new FUResource(shmBuffer_->cell(i),log_));
}


//______________________________________________________________________________
void FUResourceTable::clear()
{
  for (UInt_t i=0;i<resources_.size();i++) delete resources_[i];
  resources_.clear();
}


//______________________________________________________________________________
void FUResourceTable::reset()
{
  shmBuffer_->initialize();
  resetCounters();
}


//______________________________________________________________________________
void FUResourceTable::resetCounters()
{
  nbAllocated_=0;
  nbPending_  =0;
  nbCompleted_=0;
  nbDiscarded_=0;
  nbProcessed_=0;
  nbLost_     =0;

  nbErrors_   =0;
  nbCrcErrors_=0;
  nbAllocSent_=0;
  nbBytes_    =0;
}


//______________________________________________________________________________
void FUResourceTable::startDiscardWorkLoop() throw (evf::Exception)
{
  try {
    workLoopDiscard_=
      toolbox::task::getWorkLoopFactory()->getWorkLoop("DiscardWorkLoop",
						       "waiting");
  }
  catch (xcept::Exception& e) {
    string msg = "Failed to create DiscardWorkLoop.";
    XCEPT_RETHROW(evf::Exception,msg,e);
  }
  
  if (!workLoopDiscard_->isActive()) workLoopDiscard_->activate();
  
  asDiscard_=toolbox::task::bind(this,&FUResourceTable::discard,"Discard");
  try {
    workLoopDiscard_->submit(asDiscard_);
  }
  catch (xcept::Exception& e) {
    string msg = "Failed to submit as Discard to workloop.";
    XCEPT_RETHROW(evf::Exception,msg,e);
  }
  isReadyToShutDown_ = false;
}


//______________________________________________________________________________
bool FUResourceTable::discard(toolbox::task::WorkLoop* /* wl */)
{
  FUShmBufferCell* cell = shmBuffer_->cellToBeDiscarded();
  assert(cell->isProcessed()||cell->isEmpty());
  
  bool reschedule = true;
  bool shutDown   = cell->isEmpty();
  
  if (shutDown) {
    LOG4CPLUS_WARN(log_,"nbClientsToShutDown = "<<nbClientsToShutDown_);
    if (nbClientsToShutDown_>0) --nbClientsToShutDown_;
    if (nbClientsToShutDown_==0) {
      LOG4CPLUS_WARN(log_,"Don't reschedule discard-workloop.");
      reschedule = false;
    }
  }
  
  cell->setStateEmpty();
  shmBuffer_->postWriterSem();
  shmBuffer_->postDiscardSem();
  lock();
  resources_[cell->fuResourceId()]->release();
  unlock();
  
  if (!shutDown) {
    sendDiscard(cell->buResourceId());
    sendAllocate();
  }
  
  if (!reschedule) isReadyToShutDown_ = true;
  
  return reschedule;
}


//______________________________________________________________________________
UInt_t FUResourceTable::allocateResource()
{
  // fuResourceId for next available resource
  UInt_t fuResourceId;
  
  waitWriterSem();
  
  FUShmBufferCell* cell=shmBuffer_->currentWriterCell();
  fuResourceId=cell->fuResourceId();
  if (cell->isDead()) {
    LOG4CPLUS_ERROR(log_,"DEAD EVENT: evtNumber="<<cell->evtNumber());
    //cell->dump(); //TODO
  }
  cell->setStateWriting();
  
  // initialize the new resource and update the relevant counters
  resources_[fuResourceId]->allocate();
  nbPending_++;
  nbAllocated_++;
  
  // set the resource to check the crc for each fed if requested
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
    nbBytes_    +=resource->nbBytes();
    nbErrors_   +=resource->nbErrors();
    nbCrcErrors_+=resource->nbCrcErrors();
    unlock();
    
    // make resource available for pick-up
    if (resource->isComplete()) {
      shmBuffer_->cell(fuResourceId)->setStateWritten();
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
      resource->release();
      lock();
      bu_->sendDiscard(buResourceId);
      nbDiscarded_++;
      nbLost_++;
      nbPending_--;
      unlock();
      postWriterSem();
    }
  }
  
  return eventComplete;
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
  if (0!=shmBuffer_)
    result=FUShmBuffer::shm_nattch(shmBuffer_->shmid())-1;
  return result;
}
