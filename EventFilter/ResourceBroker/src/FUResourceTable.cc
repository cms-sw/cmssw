////////////////////////////////////////////////////////////////////////////////
//
// FUResourceTable
// ---------------
//
//            12/10/2006 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////


#include "EventFilter/ResourceBroker/interface/FUResourceTable.h"
#include "FWCore/Utilities/interface/CRC16.h"
#include "EvffedFillerRB.h"

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
				 log4cplus::Logger logger,
				 unsigned int      timeout,
				 EvffedFillerRB   *frb,
				 xdaq::Application*app)
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
  , nbDqmCells_(nbDqmCells)
  , nbRawCells_(nbRawCells)
  , nbRecoCells_(nbRecoCells)
  , acceptSMDataDiscard_(0)
  , acceptSMDqmDiscard_(0)
  , doCrcCheck_(1)
  , shutdownTimeout_(timeout)
  , nbPending_(0)
  , nbClientsToShutDown_(0)
  , isReadyToShutDown_(true)
  , isActive_(false)
  , isHalting_(false)
  , isStopping_(false)
  , runNumber_(0xffffffff)
  , lock_(toolbox::BSem::FULL)
  , frb_(frb)
  , app_(app)
{
  initialize(segmentationMode,
	     nbRawCells,nbRecoCells,nbDqmCells,
	     rawCellSize,recoCellSize,dqmCellSize);
}


//______________________________________________________________________________
FUResourceTable::~FUResourceTable()
{
  clear();
  if(wlSendData_){
    wlSendData_->cancel();
    toolbox::task::getWorkLoopFactory()->removeWorkLoop("SendData","waiting");
  }
  if(wlSendDqm_){
    wlSendDqm_->cancel();
    toolbox::task::getWorkLoopFactory()->removeWorkLoop("SendDqm","waiting");
  }
  if(wlDiscard_){
    wlDiscard_->cancel();
    toolbox::task::getWorkLoopFactory()->removeWorkLoop("Discard","waiting");
  }
  shmdt(shmBuffer_);
  if (FUShmBuffer::releaseSharedMemory())
    LOG4CPLUS_INFO(log_,"SHARED MEMORY SUCCESSFULLY RELEASED.");
  if (0!=acceptSMDataDiscard_) delete [] acceptSMDataDiscard_;
  if (0!= acceptSMDqmDiscard_) delete [] acceptSMDqmDiscard_;
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
  
  for (UInt_t i=0;i<nbRawCells_;i++) {
    resources_.push_back(new FUResource(i,log_,frb_,app_));
    freeResourceIds_.push(i);
  }

  acceptSMDataDiscard_ = new bool[nbRecoCells];
  acceptSMDqmDiscard_  = new int[nbDqmCells];
  
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
	UInt_t   cellFUProcId    = cell->fuProcessId();
	UInt_t   cellFUGuid      = cell->fuGuid();
	UChar_t* cellPayloadAddr = cell->payloadAddr();
	UInt_t   cellEventSize   = cell->eventSize();
	shmBuffer_->finishReadingRecoCell(cell);

	lock();
	nbPendingSMDiscards_++;
	unlock();

	sendInitMessage(cellIndex,cellOutModId,cellFUProcId,cellFUGuid,
			cellPayloadAddr,cellEventSize);
      }
      else if (cell->type()==1) {
	UInt_t   cellIndex       = cell->index();
	UInt_t   cellRawIndex    = cell->rawCellIndex();
	UInt_t   cellRunNumber   = cell->runNumber();
	UInt_t   cellEvtNumber   = cell->evtNumber();
	UInt_t   cellOutModId    = cell->outModId();
	UInt_t   cellFUProcId    = cell->fuProcessId();
	UInt_t   cellFUGuid      = cell->fuGuid();
	UChar_t *cellPayloadAddr = cell->payloadAddr();
	UInt_t   cellEventSize   = cell->eventSize();
	shmBuffer_->finishReadingRecoCell(cell);	

	lock();
	nbPendingSMDiscards_++;
	resources_[cellRawIndex]->incNbSent();
	if (resources_[cellRawIndex]->nbSent()==1) nbSent_++;
	unlock();

	sendDataEvent(cellIndex,cellRunNumber,cellEvtNumber,cellOutModId,
		      cellFUProcId,cellFUGuid,cellPayloadAddr,cellEventSize);
      }
      else if (cell->type()==2) {
	UInt_t   cellIndex       = cell->index();
	UInt_t   cellRawIndex    = cell->rawCellIndex();
	//UInt_t   cellRunNumber   = cell->runNumber();
	UInt_t   cellEvtNumber   = cell->evtNumber();
	UInt_t   cellFUProcId    = cell->fuProcessId();
	UInt_t   cellFUGuid      = cell->fuGuid();
	UChar_t *cellPayloadAddr = cell->payloadAddr();
	UInt_t   cellEventSize   = cell->eventSize();
	shmBuffer_->finishReadingRecoCell(cell);	

	lock();
	nbPendingSMDiscards_++;
	resources_[cellRawIndex]->incNbSent();
	if (resources_[cellRawIndex]->nbSent()==1) { nbSent_++; nbSentError_++; }
	unlock();
	
	sendErrorEvent(cellIndex,runNumber_,cellEvtNumber,
		       cellFUProcId,cellFUGuid,cellPayloadAddr,cellEventSize);
      }
      else {
	string errmsg="Unknown RecoCell type (neither INIT/DATA/ERROR).";
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
    std::cout << "shut down dqm workloop " << std::endl;
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
      UInt_t   cellFUProcId    = cell->fuProcessId();
      UInt_t   cellFUGuid      = cell->fuGuid();
      UChar_t *cellPayloadAddr = cell->payloadAddr();
      UInt_t   cellEventSize   = cell->eventSize();
      sendDqmEvent(cellIndex,cellRunNumber,cellEvtAtUpdate,cellFolderId,
		   cellFUProcId,cellFUGuid,cellPayloadAddr,cellEventSize);
      shmBuffer_->finishReadingDqmCell(cell);
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
  bool   shutDown    =(state==evt::STOP);
  bool   isLumi      =(state==evt::LUMISECTION);
  UInt_t fuResourceId=cell->fuResourceId();
  UInt_t buResourceId=cell->buResourceId();

  //  std::cout << "discard loop, state, shutDown, isLumi " << state << " " 
  //	    << shutDown << " " << isLumi << std::endl;
  //  std::cout << "resource ids " << fuResourceId << " " << buResourceId << std::endl;

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
  
  if (!shutDown && !isLumi) {
    resources_[fuResourceId]->release();
    lock();
    freeResourceIds_.push(fuResourceId);
    assert(freeResourceIds_.size()<=resources_.size());
    unlock();
    
    if (!isHalting_) {
      sendDiscard(buResourceId);
      if(!isStopping_)sendAllocate();
    }
  }
  
  if (!reschedule) {
    std::cout << " entered shutdown cycle " << std::endl;
    shmBuffer_->writeRecoEmptyEvent();
    UInt_t count=0;
    while (count<100) {
      std::cout << " shutdown cycle " <<shmBuffer_->nClients() << " "
		<<  FUShmBuffer::shm_nattch(shmBuffer_->shmid()) << std::endl;
      if (shmBuffer_->nClients()==0&&
	  FUShmBuffer::shm_nattch(shmBuffer_->shmid())==1) {
	//	isReadyToShutDown_ = true;
	break;
      }
      else {
	count++;
	std::cout << " shutdown cycle attempt " << count << std::endl;
	LOG4CPLUS_DEBUG(log_,"FUResourceTable: Wait for all clients to detach,"
			<<" nClients="<<shmBuffer_->nClients()
			<<" nattch="<<FUShmBuffer::shm_nattch(shmBuffer_->shmid())
			<<" ("<<count<<")");
	::usleep(shutdownTimeout_);
	if(count*shutdownTimeout_ > 10000000)
	  LOG4CPLUS_WARN(log_,"FUResourceTable:LONG Wait (>10s) for all clients to detach,"
			  <<" nClients="<<shmBuffer_->nClients()
			  <<" nattch="<<FUShmBuffer::shm_nattch(shmBuffer_->shmid())
			  <<" ("<<count<<")");

      }
    }
    bool allEmpty = false;
    std::cout << "Checking if all dqm cells are empty " << std::endl;
    while(!allEmpty){
      UInt_t n=nbDqmCells_;
      allEmpty = true;
      shmBuffer_->lock();
      for (UInt_t i=0;i<n;i++) {
	dqm::State_t state=shmBuffer_->dqmState(i);
	if(state!=dqm::EMPTY) allEmpty = false; 
      } 
      shmBuffer_->unlock();
    }
    std::cout << "Making sure there are no dqm pending discards " << std::endl;
    if(nbPendingSMDqmDiscards_ != 0)
      {
	LOG4CPLUS_WARN(log_,"FUResourceTable: pending DQM discards not zero: ="
		       << nbPendingSMDqmDiscards_ << " while cells are all empty. This may cause problems at next start ");

      }
    shmBuffer_->writeDqmEmptyEvent();
    isReadyToShutDown_ = true; // moved here from within the first while loop to make sure the 
                               // sendDqm loop has been shut down as well
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
    timeval now;
    gettimeofday(&now,0);

    frb_->setRBTimeStamp(((uint64_t)(now.tv_sec) << 32) + (uint64_t)(now.tv_usec));

    frb_->setRBEventCount(nbCompleted_);

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
    bufRef->release(); // this should now be safe re: appendToSuperFrag as corrupted blocks will be removed... 
  }
  
  return eventComplete;
}


//______________________________________________________________________________
bool FUResourceTable::discardDataEvent(MemRef_t* bufRef)
{
  I2O_FU_DATA_DISCARD_MESSAGE_FRAME *msg;
  msg=(I2O_FU_DATA_DISCARD_MESSAGE_FRAME*)bufRef->getDataLocation();
  UInt_t recoIndex=msg->rbBufferID;
  
  if (acceptSMDataDiscard_[recoIndex]) {
    lock();
    nbPendingSMDiscards_--;
    unlock();
    acceptSMDataDiscard_[recoIndex] = false;
    
    if (!isHalting_) {
      shmBuffer_->discardRecoCell(recoIndex);
      bufRef->release();
    }
  }
  else {
    LOG4CPLUS_ERROR(log_,"Spurious DATA discard by StorageManager, skip!");
  }
  
  if (isHalting_) {
    bufRef->release();
    return false;
  }
  
  return true;
}


//______________________________________________________________________________
bool FUResourceTable::discardDqmEvent(MemRef_t* bufRef)
{
  I2O_FU_DQM_DISCARD_MESSAGE_FRAME *msg;
  msg=(I2O_FU_DQM_DISCARD_MESSAGE_FRAME*)bufRef->getDataLocation();
  UInt_t dqmIndex=msg->rbBufferID;
  unsigned int ntries = 0;
  while(shmBuffer_->dqmState(dqmIndex)!=dqm::SENT){
    LOG4CPLUS_WARN(log_,"DQM discard for cell "<< dqmIndex << " which is not yer in SENT state - waiting");
    ::usleep(10000);
    if(ntries++>10){
      LOG4CPLUS_ERROR(log_,"DQM cell " << dqmIndex 
		      << " discard timed out while cell still in state " << shmBuffer_->dqmState(dqmIndex) );
      bufRef->release();
      return true;
    }
  }
  if (acceptSMDqmDiscard_[dqmIndex]>0) {
    acceptSMDqmDiscard_[dqmIndex]--;
    if(nbPendingSMDqmDiscards_>0){
      nbPendingSMDqmDiscards_--;
    }
    else {
      LOG4CPLUS_WARN(log_,"Spurious??? DQM discard by StorageManager, index " << dqmIndex 
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
  
  return true;
}


//______________________________________________________________________________
void FUResourceTable::postEndOfLumiSection(MemRef_t* bufRef)
{
  I2O_EVM_END_OF_LUMISECTION_MESSAGE_FRAME *msg = 
    (I2O_EVM_END_OF_LUMISECTION_MESSAGE_FRAME *)bufRef->getDataLocation();
  //make sure to fill up the shmem so no process will miss it
  // but processes will have to handle duplicates

  for(unsigned int i = 0; i < nbRawCells_; i++) 
    shmBuffer_->writeRawLumiSectionEvent(msg->lumiSection);
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
bool FUResourceTable::handleCrashedEP(UInt_t runNumber,pid_t pid)
{
  bool retval = false;
  vector<pid_t> pids=cellPrcIds();
  UInt_t iRawCell=pids.size();
  for (UInt_t i=0;i<pids.size();i++) { if (pid==pids[i]) { iRawCell=i; break; } }
  
  if (iRawCell<pids.size()){
    shmBuffer_->writeErrorEventData(runNumber,pid,iRawCell);
    retval = true;
  }
  else
    LOG4CPLUS_WARN(log_,"No raw data to send to error stream for process " << pid);
  shmBuffer_->removeClientPrcId(pid);
  return retval;
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
  isStopping_ = true;
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
    LOG4CPLUS_INFO(log_,"No clients to shut down. Checking if there are raw cells not assigned to any process yet");
    UInt_t n=nbResources();
    for (UInt_t i=0;i<n;i++) {
      evt::State_t state=shmBuffer_->evtState(i);
      if (state!=evt::EMPTY){
	LOG4CPLUS_WARN(log_,"Schedule discard at STOP for orphaned event in state " 
		       << state);
	shmBuffer_->scheduleRawCellForDiscardServerSide(i);
      }
    }
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
  if (0!=shmBuffer_) {
    for (UInt_t i=0;i<shmBuffer_->nRecoCells();i++) acceptSMDataDiscard_[i]= false;
    for (UInt_t i=0;i<shmBuffer_->nDqmCells();i++)  acceptSMDqmDiscard_[i] = 0;
  }
  
  nbAllocated_           =nbPending_;
  nbCompleted_           =0;
  nbSent_                =0;
  nbSentError_           =0;
  nbSentDqm_             =0;
  nbPendingSMDiscards_   =0;
  nbPendingSMDqmDiscards_=0;
  nbDiscarded_           =0;
  nbLost_                =0;

  nbErrors_              =0;
  nbCrcErrors_           =0;
  nbAllocSent_           =0;

  sumOfSquares_          =0;
  sumOfSizes_            =0;
  isStopping_            =false;
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
    shmBuffer_->lock();
    for (UInt_t i=0;i<n;i++) {
      evt::State_t state=shmBuffer_->evtState(i);
      if      (state==evt::EMPTY)      result.push_back("EMPTY");
      else if (state==evt::STOP)       result.push_back("STOP");
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
    shmBuffer_->unlock();
  }
  return result;
}

vector<string> FUResourceTable::dqmCellStates() const
{
  vector<string> result;
  if (0!=shmBuffer_) {
    UInt_t n=nbDqmCells_;
    shmBuffer_->lock();
    for (UInt_t i=0;i<n;i++) {
      dqm::State_t state=shmBuffer_->dqmState(i);
      if      (state==dqm::EMPTY)      result.push_back("EMPTY");
      else if (state==dqm::WRITING) result.push_back("WRITING");
      else if (state==dqm::WRITTEN) result.push_back("WRITTEN");
      else if (state==dqm::SENDING)    result.push_back("SENDING");
      else if (state==dqm::SENT)       result.push_back("SENT");
      else if (state==dqm::DISCARDING) result.push_back("DISCARDING");
    }
    shmBuffer_->unlock();
  }
  return result;
}


//______________________________________________________________________________
vector<UInt_t> FUResourceTable::cellEvtNumbers() const
{
  vector<UInt_t> result;
  if (0!=shmBuffer_) {
    UInt_t n=nbResources();
    shmBuffer_->lock();
    for (UInt_t i=0;i<n;i++) result.push_back(shmBuffer_->evtNumber(i));
    shmBuffer_->unlock();
  }
  return result;
}


//______________________________________________________________________________
vector<pid_t> FUResourceTable::cellPrcIds() const
{
  vector<pid_t> result;
  if (0!=shmBuffer_) {
    UInt_t n=nbResources();
    shmBuffer_->lock();
    for (UInt_t i=0;i<n;i++) result.push_back(shmBuffer_->evtPrcId(i));
    shmBuffer_->unlock();
  }
  return result;
}


//______________________________________________________________________________
vector<time_t> FUResourceTable::cellTimeStamps() const
{
  vector<time_t> result;
  if (0!=shmBuffer_) {
    UInt_t n=nbResources();
    shmBuffer_->lock();
    for (UInt_t i=0;i<n;i++) result.push_back(shmBuffer_->evtTimeStamp(i));
    shmBuffer_->unlock();
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
				      UInt_t   fuProcessId,
				      UInt_t   fuGuid,
				      UChar_t *data,
				      UInt_t   dataSize)
{
  if (0==sm_) {
    LOG4CPLUS_ERROR(log_,"No StorageManager, DROP INIT MESSAGE!");
  }
  else {
    acceptSMDataDiscard_[fuResourceId] = true;
    UInt_t nbBytes=sm_->sendInitMessage(fuResourceId,outModId,fuProcessId,
					fuGuid,data,dataSize);
    sumOfSquares_+=(uint64_t)nbBytes*(uint64_t)nbBytes;
    sumOfSizes_  +=nbBytes;
  }
}


//______________________________________________________________________________
void FUResourceTable::sendDataEvent(UInt_t   fuResourceId,
				    UInt_t   runNumber,
				    UInt_t   evtNumber,
				    UInt_t   outModId,
				    UInt_t   fuProcessId,
				    UInt_t   fuGuid,
				    UChar_t *data,
				    UInt_t   dataSize)
{
  if (0==sm_) {
    LOG4CPLUS_ERROR(log_,"No StorageManager, DROP DATA EVENT!");
  }
  else {
    acceptSMDataDiscard_[fuResourceId] = true;
    UInt_t nbBytes=sm_->sendDataEvent(fuResourceId,runNumber,evtNumber,
				      outModId,fuProcessId,fuGuid,
				      data,dataSize);
    sumOfSquares_+=(uint64_t)nbBytes*(uint64_t)nbBytes;
    sumOfSizes_  +=nbBytes;
  }
}


//______________________________________________________________________________
void FUResourceTable::sendErrorEvent(UInt_t   fuResourceId,
				     UInt_t   runNumber,
				     UInt_t   evtNumber,
				     UInt_t   fuProcessId,
				     UInt_t   fuGuid,
				     UChar_t *data,
				     UInt_t   dataSize)
{
  if (0==sm_) {
    LOG4CPLUS_ERROR(log_,"No StorageManager, DROP ERROR EVENT!");
  }
  else {
    acceptSMDataDiscard_[fuResourceId] = true;
    UInt_t nbBytes=sm_->sendErrorEvent(fuResourceId,runNumber,evtNumber,
				       fuProcessId,fuGuid,data,dataSize);
    sumOfSquares_+=(uint64_t)nbBytes*(uint64_t)nbBytes;
    sumOfSizes_  +=nbBytes;
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
void FUResourceTable::sendDqmEvent(UInt_t   fuDqmId,
				   UInt_t   runNumber,
				   UInt_t   evtAtUpdate,
				   UInt_t   folderId,
				   UInt_t   fuProcessId,
				   UInt_t   fuGuid,
				   UChar_t* data,
				   UInt_t   dataSize)
{
  if (0==sm_) {
    LOG4CPLUS_WARN(log_,"No StorageManager, DROP DQM EVENT.");
  }
  else {
    sm_->sendDqmEvent(fuDqmId,runNumber,evtAtUpdate,folderId,
		      fuProcessId,fuGuid,data,dataSize);

    nbPendingSMDqmDiscards_++;

    acceptSMDqmDiscard_[fuDqmId]++;
    if(acceptSMDqmDiscard_[fuDqmId]>1)
      LOG4CPLUS_WARN(log_,"DQM Cell " << fuDqmId << " being sent more than once for folder " 
		     << folderId << " process " << fuProcessId << " guid " << fuGuid);
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

//______________________________________________________________________________
void FUResourceTable::injectCRCError()
{
  for (UInt_t i=0;i<resources_.size();i++) {
    resources_[i]->scheduleCRCError();
  }
}
void FUResourceTable::printWorkLoopStatus()
{
  std::cout << "Workloop status===============" << std::endl;
  std::cout << "==============================" << std::endl;
  if(wlSendData_!=0)
    std::cout << "SendData -> " << wlSendData_->isActive() << std::endl;
  if(wlSendDqm_!=0)
    std::cout << "SendDqm  -> " << wlSendDqm_->isActive()  << std::endl;
  if(wlDiscard_!=0)
    std::cout << "Discard  -> " << wlDiscard_->isActive()  << std::endl;
  std::cout << "Workloops Active  -> " << isActive_  << std::endl;

}


void FUResourceTable::lastResort()
{
  std::cout << "lastResort: " << shmBuffer_->nbRawCellsToRead() 
	    << " more rawcells to read " << std::endl;
  while(shmBuffer_->nbRawCellsToRead()!=0){
    FUShmRawCell* newCell=shmBuffer_->rawCellToRead();
    std::cout << "lastResort: " << shmBuffer_->nbRawCellsToRead() << std::endl;
    shmBuffer_->scheduleRawEmptyCellForDiscardServerSide(newCell);
    std::cout << "lastResort: schedule raw cell for discard" << std::endl;
  }
}
