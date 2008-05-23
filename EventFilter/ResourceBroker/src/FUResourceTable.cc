////////////////////////////////////////////////////////////////////////////////
//
// FUResourceTable
// ---------------
//
//            12/10/2006 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////


#include "EventFilter/ResourceBroker/interface/FUResourceTable.h"

#include "toolbox/task/WorkLoopFactory.h"
#include "interface/evb/i2oEVBMsgs.h"
#include "xcept/tools.h"


#include <fstream>
#include <sstream>
#include <iomanip>
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
  throw (evf::Exception)
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
  , nbPending_(0)
  , nbClientsToShutDown_(0)
  , isReadyToShutDown_(true)
  , isActive_(false)
  , isHalting_(false)
  , lock_(toolbox::BSem::FULL)
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
  throw (evf::Exception)
{
  clear();
  
  shmBuffer_=FUShmBuffer::createShmBuffer(segmentationMode,
					  nbRawCells,nbRecoCells,nbDqmCells,
					  rawCellSize,recoCellSize,dqmCellSize);
  if (0==shmBuffer_) {
    string msg = "CREATION OF SHARED MEMORY SEGMENT FAILED!";
    LOG4CPLUS_FATAL(log_,msg);
    XCEPT_RAISE(evf::Exception,msg);
  }
  
  for (UInt_t i=0;i<nbRawCells;i++) {
    resources_.push_back(new FUResource(i,log_));
    freeResourceIds_.push(i);
  }
  
  resetCounters();
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
  
  if (0==cell->eventSize()) {
    LOG4CPLUS_INFO(log_,"Don't reschedule sendData workloop.");
    UInt_t cellIndex=cell->index();
    shmBuffer_->finishReadingRecoCell(cell);
    shmBuffer_->discardRecoCell(cellIndex);
    reschedule=false;
  }
  else if (isHalting_) {
    LOG4CPLUS_INFO(log_,"sendData: isHalting, discard recoCell.");
    UInt_t cellIndex=cell->index();
    shmBuffer_->finishReadingRecoCell(cell);
    shmBuffer_->discardRecoCell(cellIndex);
  }
  else {
    try {
      if (cell->type()==0) {
	UInt_t   cellIndex       = cell->index();
	UInt_t   cellOutModId    = cell->outModId();
	UChar_t* cellPayloadAddr = cell->payloadAddr();
	UInt_t   cellEventSize   = cell->eventSize();
	shmBuffer_->finishReadingRecoCell(cell);
	
	sendInitMessage(cellIndex,cellOutModId,cellPayloadAddr,cellEventSize);
      }
      else if (cell->type()==1) {
	lock();
	nbAccepted_++;
	unlock();
	UInt_t   cellIndex       = cell->index();
	UInt_t   cellRunNumber   = cell->runNumber();
	UInt_t   cellEvtNumber   = cell->evtNumber();
	UInt_t   cellOutModId    = cell->outModId();
	UChar_t *cellPayloadAddr = cell->payloadAddr();
	UInt_t   cellEventSize   = cell->eventSize();
	shmBuffer_->finishReadingRecoCell(cell);	

	sendDataEvent(cellIndex,cellRunNumber,cellEvtNumber,cellOutModId,
		      cellPayloadAddr,cellEventSize);
      }
      else if (cell->type()==2) {
	UInt_t cellIndex=cell->index();
	UInt_t   cellRunNumber   = cell->runNumber();
	UInt_t   cellEvtNumber   = cell->evtNumber();
	UChar_t *cellPayloadAddr = cell->payloadAddr();
	UInt_t   cellEventSize   = cell->eventSize();
	shmBuffer_->finishReadingRecoCell(cell);	

	sendErrorEvent(cellIndex,cellRunNumber,cellEvtNumber,
		       cellPayloadAddr,cellEventSize);
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
    UInt_t cellIndex=cell->index();
    shmBuffer_->finishReadingDqmCell(cell);
    shmBuffer_->discardDqmCell(cellIndex);
    reschedule=false;
  }
  else if (isHalting_) {
    UInt_t cellIndex=cell->index();
    shmBuffer_->finishReadingDqmCell(cell);
    shmBuffer_->discardDqmCell(cellIndex);
  }
  else {
    try {
      UInt_t   cellIndex       = cell->index();
      UInt_t   cellRunNumber   = cell->runNumber();
      UInt_t   cellEvtAtUpdate = cell->evtAtUpdate();
      UInt_t   cellFolderId    = cell->folderId();
      UChar_t *cellPayloadAddr = cell->payloadAddr();
      UInt_t   cellEventSize   = cell->eventSize();
      shmBuffer_->finishReadingDqmCell(cell);      

      sendDqmEvent(cellIndex,cellRunNumber,cellEvtAtUpdate,cellFolderId,
		   cellPayloadAddr,cellEventSize);
    }
    catch (xcept::Exception& e) {
      LOG4CPLUS_FATAL(log_,"Failed to send DQM DATA to StorageManager: "
		      <<xcept::stdformat_exception_history(e));
      reschedule=false;
    }
  }
  
  return reschedule;
}


//______________________________________________________________________________
void FUResourceTable::startDiscardWorkLoop() throw (evf::Exception)
{
  try {
    LOG4CPLUS_INFO(log_,"Start 'discard' workloop.");
    wlDiscard_=toolbox::task::getWorkLoopFactory()->getWorkLoop("Discard","waiting");
    if (!wlDiscard_->isActive()) wlDiscard_->activate();
    asDiscard_=toolbox::task::bind(this,&FUResourceTable::discard,"Discard");
    wlDiscard_->submit(asDiscard_);
    isActive_=true;
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
    LOG4CPLUS_INFO(log_,"nbClientsToShutDown = "<<nbClientsToShutDown_);
    if (nbClientsToShutDown_>0) --nbClientsToShutDown_;
    if (nbClientsToShutDown_==0) {
      LOG4CPLUS_INFO(log_,"Don't reschedule discard-workloop.");
      isActive_ =false;
      reschedule=false;
    }
  }
  
  shmBuffer_->discardRawCell(cell);
  
  if (!shutDown) {
    resources_[fuResourceId]->release();
    lock();
    freeResourceIds_.push(fuResourceId);
    assert(freeResourceIds_.size()<=resources_.size());
    unlock();
    
    if (!isHalting_) {
      sendDiscard(buResourceId);
      sendAllocate();
    }
  }
  
  if (!reschedule) {
    shmBuffer_->writeRecoEmptyEvent();
    shmBuffer_->writeDqmEmptyEvent();
    
    UInt_t count=0;
    while (count<10) {
      if (shmBuffer_->nClients()==0&&
	  FUShmBuffer::shm_nattch(shmBuffer_->shmid())==1) {
	isReadyToShutDown_ = true;
	break;
      }
      else {
	count++;
	LOG4CPLUS_DEBUG(log_,"FUResourceTable: Wait for all clients to detach,"
			<<" nClients="<<shmBuffer_->nClients()
			<<" nattch="<<FUShmBuffer::shm_nattch(shmBuffer_->shmid())
			<<" ("<<count<<")");
	::sleep(1);
      }
    }
    
  }
  
  return reschedule;
}



//______________________________________________________________________________
UInt_t FUResourceTable::allocateResource()
{
  assert(!freeResourceIds_.empty());

  lock();
  UInt_t fuResourceId=freeResourceIds_.front();
  freeResourceIds_.pop();
  nbPending_++;
  nbAllocated_++;
  unlock();
  
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
  
  // allocate resource
  if (!resource->fatalError()&&!resource->isAllocated()) {
    FUShmRawCell* cell=shmBuffer_->rawCellToWrite();
    resource->allocate(cell);
    if (doCrcCheck_>0&&0==nbAllocated_%doCrcCheck_)  resource->doCrcCheck(true);
    else                                             resource->doCrcCheck(false);
  }
  
  
  // keep building this resource if it is healthy
  if (!resource->fatalError()) {
    resource->process(bufRef);
    lock();
    nbErrors_   +=resource->nbErrors();
    nbCrcErrors_+=resource->nbCrcErrors();
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
      if (doDumpEvents_>0&&nbCompleted_%doDumpEvents_==0)
	dumpEvent(resource->shmCell());
      shmBuffer_->finishWritingRawCell(resource->shmCell());
      eventComplete=true;
    }
    
  }
  // bad event, release msg, and the whole resource if this was the last one
  //else {
  if (resource->fatalError()) {
    bool lastMsg=isLastMessageOfEvent(bufRef);
    if (lastMsg) {
      shmBuffer_->releaseRawCell(resource->shmCell());
      resource->release();
      lock();
      freeResourceIds_.push(fuResourceId);
      nbDiscarded_++;
      nbLost_++;
      nbPending_--;
      unlock();
      bu_->sendDiscard(buResourceId);
      sendAllocate();
    }
    else {
      bufRef->release();
    }
  }
  
  return eventComplete;
}


//______________________________________________________________________________
bool FUResourceTable::discardDataEvent(MemRef_t* bufRef)
{
  if (isHalting_) {
    bufRef->release();
    return false;
  }

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
  if (isHalting_) {
    bufRef->release();
    return false;
  }
  
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
void FUResourceTable::handleErrorEvent(UInt_t runNumber,pid_t pid)
{
  lockShm();
  vector<pid_t> pids=cellPrcIds();
  unlockShm();
  UInt_t iRawCell=pids.size();
  for (UInt_t i=0;i<pids.size();i++) { if (pid==pids[i]) { iRawCell=i; break; } }
  
  if (iRawCell<pids.size())
    shmBuffer_->writeErrorEventData(runNumber,iRawCell);
  
  shmBuffer_->removeClientPrcId(pid);
}


//______________________________________________________________________________
void FUResourceTable::dumpEvent(FUShmRawCell* cell)
{
  ostringstream oss; oss<<"/tmp/evt"<<cell->evtNumber()<<".dump";
  ofstream fout(oss.str().c_str());
  fout.fill('0');

  fout<<"#\n# evt "<<cell->evtNumber()<<"\n#\n"<<endl;
  for (unsigned int i=0;i<cell->nFed();i++) {
    if (cell->fedSize(i)==0) continue;
    fout<<"# fedid "<<i<<endl;
    unsigned char* addr=cell->fedAddr(i);
    for (unsigned int j=0;j<cell->fedSize(i);j++) {
      fout<<setiosflags(ios::right)<<setw(2)<<hex<<(int)(*addr)<<dec;
      if ((j+1)%8) fout<<" "; else fout<<endl;
      ++addr;
    }
    fout<<endl;
  }
  fout.close();
}


//______________________________________________________________________________
void FUResourceTable::stop()
{
  shutDownClients();
}


//______________________________________________________________________________
void FUResourceTable::halt()
{
  isHalting_=true;
  shutDownClients();
}


//______________________________________________________________________________
void FUResourceTable::shutDownClients()
{
  nbClientsToShutDown_ = nbClients();
  isReadyToShutDown_   = false;
  
  if (nbClientsToShutDown_==0) {
    shmBuffer_->scheduleRawEmptyCellForDiscard();
  }
  else {
    UInt_t n=nbClientsToShutDown_;
    for (UInt_t i=0;i<n;++i) shmBuffer_->writeRawEmptyEvent();
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
  while (!freeResourceIds_.empty()) freeResourceIds_.pop();
}


//______________________________________________________________________________
void FUResourceTable::resetCounters()
{
  nbAllocated_       =nbPending_;
  nbCompleted_       =0;
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
UInt_t FUResourceTable::nbClients() const
{
  UInt_t result(0);
  if (0!=shmBuffer_) result=shmBuffer_->nClients();
  return result;
}


//______________________________________________________________________________
vector<pid_t> FUResourceTable::clientPrcIds() const
{
  vector<pid_t> result;
  if (0!=shmBuffer_) {
    UInt_t n=nbClients();
    for (UInt_t i=0;i<n;i++) result.push_back(shmBuffer_->clientPrcId(i));
  }
  return result;
}


//______________________________________________________________________________
string FUResourceTable::clientPrcIdsAsString() const
{
  stringstream ss;
  if (0!=shmBuffer_) {
    UInt_t n=nbClients();
    for (UInt_t i=0;i<n;i++) {
      if (i>0) ss<<",";
      ss<<shmBuffer_->clientPrcId(i);
    }
  }
  return ss.str();
}


//______________________________________________________________________________
vector<string> FUResourceTable::cellStates() const
{
  vector<string> result;
  if (0!=shmBuffer_) {
    UInt_t n=nbResources();
    for (UInt_t i=0;i<n;i++) {
      evt::State_t state=shmBuffer_->evtState(i);
      if      (state==evt::EMPTY)      result.push_back("EMPTY");
      else if (state==evt::RAWWRITING) result.push_back("RAWWRITING");
      else if (state==evt::RAWWRITTEN) result.push_back("RAWWRITTEN");
      else if (state==evt::RAWREADING) result.push_back("RAWREADING");
      else if (state==evt::RAWREAD)    result.push_back("RAWREAD");
      else if (state==evt::PROCESSING) result.push_back("PROCESSING");
      else if (state==evt::PROCESSED)  result.push_back("PROCESSED");
      else if (state==evt::RECOWRITING)result.push_back("RECOWRITING");
      else if (state==evt::RECOWRITTEN)result.push_back("RECOWRITTEN");
      else if (state==evt::SENDING)    result.push_back("SENDING");
      else if (state==evt::SENT)       result.push_back("SENT");
      else if (state==evt::DISCARDING) result.push_back("DISCARDING");
    }
  }
  return result;
}


//______________________________________________________________________________
vector<UInt_t> FUResourceTable::cellEvtNumbers() const
{
  vector<UInt_t> result;
  if (0!=shmBuffer_) {
    UInt_t n=nbResources();
    for (UInt_t i=0;i<n;i++) result.push_back(shmBuffer_->evtNumber(i));
  }
  return result;
}


//______________________________________________________________________________
vector<pid_t> FUResourceTable::cellPrcIds() const
{
  vector<pid_t> result;
  if (0!=shmBuffer_) {
    UInt_t n=nbResources();
    for (UInt_t i=0;i<n;i++) result.push_back(shmBuffer_->evtPrcId(i));
  }
  return result;
}


//______________________________________________________________________________
vector<time_t> FUResourceTable::cellTimeStamps() const
{
  vector<time_t> result;
  if (0!=shmBuffer_) {
    UInt_t n=nbResources();
    for (UInt_t i=0;i<n;i++) result.push_back(shmBuffer_->evtTimeStamp(i));
  }
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
}


//______________________________________________________________________________
void FUResourceTable::sendInitMessage(UInt_t   fuResourceId,
				      UInt_t   outModId,
				      UChar_t *data,
				      UInt_t   dataSize)
{
  if (0==sm_) {
    LOG4CPLUS_ERROR(log_,"No StorageManager, DROP INIT MESSAGE!");
  }
  else {
    UInt_t   nbBytes    =sm_->sendInitMessage(fuResourceId,outModId,data,dataSize);
    outputSumOfSquares_+=(uint64_t)nbBytes*(uint64_t)nbBytes;
    outputSumOfSizes_  +=nbBytes;
  }
}


//______________________________________________________________________________
void FUResourceTable::sendDataEvent(UInt_t   fuResourceId,
				    UInt_t   runNumber,
				    UInt_t   evtNumber,
				    UInt_t   outModId,
				    UChar_t *data,
				    UInt_t   dataSize)
{
  if (0==sm_) {
    LOG4CPLUS_ERROR(log_,"No StorageManager, DROP DATA EVENT!");
  }
  else {
    UInt_t   nbBytes    =sm_->sendDataEvent(fuResourceId,
					    runNumber,evtNumber,outModId,
					    data,dataSize);
    outputSumOfSquares_+=(uint64_t)nbBytes*(uint64_t)nbBytes;
    outputSumOfSizes_  +=nbBytes;
    nbSent_++;
  }
}


//______________________________________________________________________________
void FUResourceTable::sendErrorEvent(UInt_t   fuResourceId,
				     UInt_t   runNumber,
				     UInt_t   evtNumber,
				     UChar_t *data,
				     UInt_t   dataSize)
{
  if (0==sm_) {
    LOG4CPLUS_ERROR(log_,"No StorageManager, DROP ERROR EVENT!");
  }
  else {
    UInt_t nbBytes=sm_->sendErrorEvent(fuResourceId,
				       runNumber,evtNumber,data,dataSize);
    outputSumOfSquares_+=(uint64_t)nbBytes*(uint64_t)nbBytes;
    outputSumOfSizes_  +=nbBytes;
    nbSent_++;
  }
}


//______________________________________________________________________________
void FUResourceTable::sendDqmEvent(UInt_t   fuDqmId,
				   UInt_t   runNumber,
				   UInt_t   evtAtUpdate,
				   UInt_t   folderId,
				   UChar_t* data,
				   UInt_t   dataSize)
{
  if (0==sm_) {
    LOG4CPLUS_WARN(log_,"No StorageManager, DROP DQM EVENT.");
  }
  else {
    sm_->sendDqmEvent(fuDqmId,runNumber,evtAtUpdate,folderId,data,dataSize);
    nbSentDqm_++;
  }
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
