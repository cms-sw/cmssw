///////////////////////////////////////////////////////////////////////////////
//
// FUShmBuffer
// -----------
//
//            15/11/2006 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////


#include "EventFilter/ShmBuffer/interface/FUShmBuffer.h"

#include <iostream>
#include <string>


#define SHM_KEYPATH          "/dev/null" /* Path used on ftok for shmget key  */
#define SHM_DESCRIPTOR_KEYID           1 /* Id used on ftok for 1. shmget key */
#define SHM_KEYID                      2 /* Id used on ftok for 2. shmget key */
#define SEM_KEYPATH          "/dev/null" /* Path used on ftok for semget key  */
#define SEM_KEYID                      1 /* Id used on ftok for semget key    */

#define NSKIP_MAX                    100


using namespace std;
using namespace evf;


////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
FUShmBuffer::FUShmBuffer(bool         segmentationMode,
			 unsigned int nRawCells,
			 unsigned int nRecoCells,
			 unsigned int nDqmCells,
			 unsigned int rawCellSize,
			 unsigned int recoCellSize,
			 unsigned int dqmCellSize)
  : segmentationMode_(segmentationMode)
  , nRawCells_(nRawCells)
  , rawCellPayloadSize_(rawCellSize)
  , nRecoCells_(nRecoCells)
  , recoCellPayloadSize_(recoCellSize)
  , nDqmCells_(nDqmCells)
  , dqmCellPayloadSize_(dqmCellSize)
{
  rawCellTotalSize_ =FUShmRawCell::size(rawCellPayloadSize_);
  recoCellTotalSize_=FUShmRecoCell::size(recoCellPayloadSize_);
  dqmCellTotalSize_ =FUShmDqmCell::size(dqmCellPayloadSize_);

  void* addr;
  
  rawWriteOffset_=sizeof(FUShmBuffer);
  addr=(void*)((unsigned int)this+rawWriteOffset_);
  new (addr) unsigned int[nRawCells_];
  
  rawReadOffset_=rawWriteOffset_+nRawCells_*sizeof(unsigned int);
  addr=(void*)((unsigned int)this+rawReadOffset_);
  new (addr) unsigned int[nRawCells_];
 
  recoWriteOffset_=rawReadOffset_+nRawCells_*sizeof(unsigned int);
  addr=(void*)((unsigned int)this+recoWriteOffset_);
  new (addr) unsigned int[nRecoCells_];

  recoReadOffset_=recoWriteOffset_+nRecoCells_*sizeof(unsigned int);
  addr=(void*)((unsigned int)this+recoReadOffset_);
  new (addr) unsigned int[nRecoCells_];

  dqmWriteOffset_=recoReadOffset_+nRecoCells_*sizeof(unsigned int);
  addr=(void*)((unsigned int)this+dqmWriteOffset_);
  new (addr) unsigned int[nDqmCells_];

  dqmReadOffset_=dqmWriteOffset_+nDqmCells_*sizeof(unsigned int);
  addr=(void*)((unsigned int)this+dqmReadOffset_);
  new (addr) unsigned int[nDqmCells_];

  evtStateOffset_=dqmReadOffset_+nDqmCells_*sizeof(unsigned int);
  addr=(void*)((unsigned int)this+evtStateOffset_);
  new (addr) evt::State_t[nRawCells_];
  
  evtDiscardOffset_=evtStateOffset_+nRawCells_*sizeof(evt::State_t);
  addr=(void*)((unsigned int)this+evtDiscardOffset_);
  new (addr) unsigned int[nRawCells_];
  
  evtNumberOffset_=evtDiscardOffset_+nRawCells_*sizeof(unsigned int);
  addr=(void*)((unsigned int)this+evtNumberOffset_);
  new (addr) unsigned int[nRawCells_];
  
  evtRecoIdOffset_=evtNumberOffset_+nRawCells_*sizeof(unsigned int);
  addr=(void*)((unsigned int)this+evtRecoIdOffset_);
  new (addr) unsigned int[nRawCells_];

  dqmStateOffset_=evtRecoIdOffset_+nRawCells_*sizeof(unsigned int);
  addr=(void*)((unsigned int)this+dqmStateOffset_);
  new (addr) dqm::State_t[nDqmCells_];
  
  rawCellOffset_=dqmStateOffset_+nDqmCells_*sizeof(dqm::State_t);
  
  if (segmentationMode_) {
    recoCellOffset_=rawCellOffset_+nRawCells_*sizeof(key_t);
    dqmCellOffset_ =recoCellOffset_+nRecoCells_*sizeof(key_t);
    addr=(void*)((unsigned int)this+rawCellOffset_);
    new (addr) key_t[nRawCells_];
    addr=(void*)((unsigned int)this+recoCellOffset_);
    new (addr) key_t[nRecoCells_];
    addr=(void*)((unsigned int)this+dqmCellOffset_);
    new (addr) key_t[nDqmCells_];
  }
  else {
    recoCellOffset_=rawCellOffset_+nRawCells_*rawCellTotalSize_;
    dqmCellOffset_ =recoCellOffset_+nRecoCells_*recoCellTotalSize_;
    for (unsigned int i=0;i<nRawCells_;i++) {
      addr=(void*)((unsigned int)this+rawCellOffset_+i*rawCellTotalSize_);
      new (addr) FUShmRawCell(rawCellSize);
    }
    for (unsigned int i=0;i<nRecoCells_;i++) {
      addr=(void*)((unsigned int)this+recoCellOffset_+i*recoCellTotalSize_);
      new (addr) FUShmRecoCell(recoCellSize);
    }
    for (unsigned int i=0;i<nDqmCells_;i++) {
      addr=(void*)((unsigned int)this+dqmCellOffset_+i*dqmCellTotalSize_);
      new (addr) FUShmDqmCell(dqmCellSize);
    }
  }
}


//______________________________________________________________________________
FUShmBuffer::~FUShmBuffer()
{
  
}


////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void FUShmBuffer::initialize(unsigned int shmid,unsigned int semid)
{
  shmid_=shmid;
  semid_=semid;
  
  if (segmentationMode_) {
    int    shmKeyId=666;
    key_t* keyAddr =(key_t*)((unsigned int)this+rawCellOffset_);
    for (unsigned int i=0;i<nRawCells_;i++) {
      *keyAddr     =ftok("/dev/null",shmKeyId++);
      int   shmid  =shm_create(*keyAddr,rawCellTotalSize_);
      void* shmAddr=shm_attach(shmid);
      new (shmAddr) FUShmRawCell(rawCellPayloadSize_);
      shmdt(shmAddr);
      ++keyAddr;
    }
    keyAddr =(key_t*)((unsigned int)this+recoCellOffset_);
    for (unsigned int i=0;i<nRecoCells_;i++) {
      *keyAddr     =ftok("/dev/null",shmKeyId++);
      int   shmid  =shm_create(*keyAddr,recoCellTotalSize_);
      void* shmAddr=shm_attach(shmid);
      new (shmAddr) FUShmRecoCell(recoCellPayloadSize_);
      shmdt(shmAddr);
      ++keyAddr;
    }
    keyAddr =(key_t*)((unsigned int)this+dqmCellOffset_);
    for (unsigned int i=0;i<nDqmCells_;i++) {
      *keyAddr     =ftok("/dev/null",shmKeyId++);
      int   shmid  =shm_create(*keyAddr,dqmCellTotalSize_);
      void* shmAddr=shm_attach(shmid);
      new (shmAddr) FUShmDqmCell(dqmCellPayloadSize_);
      shmdt(shmAddr);
      ++keyAddr;
    }
  }
  
  for (unsigned int i=0;i<nRawCells_;i++) {
    FUShmRawCell* cell=rawCell(i);
    cell->initialize(i);
    if (segmentationMode_) shmdt(cell);
  }
  
  for (unsigned int i=0;i<nRecoCells_;i++) {
    FUShmRecoCell* cell=recoCell(i);
    cell->initialize(i);
    if (segmentationMode_) shmdt(cell);
  }
  
  for (unsigned int i=0;i<nDqmCells_;i++) {
    FUShmDqmCell* cell=dqmCell(i);
    cell->initialize(i);
    if (segmentationMode_) shmdt(cell);
  }

  reset();
}


//______________________________________________________________________________
void FUShmBuffer::reset()
{
  // setup ipc semaphores
  sem_init(0,1);          // lock (binary)
  sem_init(1,nRawCells_); // raw  write semaphore
  sem_init(2,0);          // raw  read  semaphore
  sem_init(3,1);          // binary semaphore to schedule raw event for discard
  sem_init(4,0);          // binary semaphore to discard raw event
  sem_init(5,nRecoCells_);// reco write semaphore
  sem_init(6,0);          // reco send (read) semaphore
  sem_init(7,nDqmCells_); // dqm  write semaphore
  sem_init(8,0);          // dqm  send (read) semaphore

  sem_print();

  unsigned int *iWrite,*iRead;
  
  rawWriteNext_=0; rawWriteLast_=0; rawReadNext_ =0; rawReadLast_ =0;
  iWrite=(unsigned int*)((unsigned int)this+rawWriteOffset_);
  iRead =(unsigned int*)((unsigned int)this+rawReadOffset_);
  for (unsigned int i=0;i<nRawCells_;i++) { *iWrite++=i; *iRead++ =0xffffffff; }

  recoWriteNext_=0; recoWriteLast_=0; recoReadNext_ =0; recoReadLast_ =0;
  iWrite=(unsigned int*)((unsigned int)this+recoWriteOffset_);
  iRead =(unsigned int*)((unsigned int)this+recoReadOffset_);
  for (unsigned int i=0;i<nRecoCells_;i++) { *iWrite++=i; *iRead++ =0xffffffff; }

  dqmWriteNext_=0; dqmWriteLast_=0; dqmReadNext_ =0; dqmReadLast_ =0;
  iWrite=(unsigned int*)((unsigned int)this+dqmWriteOffset_);
  iRead =(unsigned int*)((unsigned int)this+dqmReadOffset_);
  for (unsigned int i=0;i<nDqmCells_;i++) { *iWrite++=i; *iRead++ =0xffffffff; }
  
  for (unsigned int i=0;i<nRawCells_;i++) {
    setEvtState(i,evt::EMPTY);
    setEvtDiscard(i,0);
    setEvtNumber(i,0xffffffff);
    setRecoCellId(i,0xffffffff);
  }

  for (unsigned int i=0;i<nDqmCells_;i++) setDqmState(i,dqm::EMPTY);
}


//______________________________________________________________________________
unsigned int FUShmBuffer::nbClients()
{
  return FUShmBuffer::shm_nattch(shmid_)-1;
}


//______________________________________________________________________________
int FUShmBuffer::nbRawCellsToWrite() const
{
  return semctl(semid(),1,GETVAL);
}


//______________________________________________________________________________
int FUShmBuffer::nbRawCellsToRead() const
{
  return semctl(semid(),2,GETVAL);
}


//______________________________________________________________________________
FUShmRawCell* FUShmBuffer::rawCellToWrite()
{
  waitRawWrite();
  unsigned int  iCell=nextRawWriteIndex();
  FUShmRawCell* cell =rawCell(iCell);
  evt::State_t  state=evtState(iCell);
  assert(state==evt::EMPTY);
  setEvtState(iCell,evt::RAWWRITING);
  setEvtDiscard(iCell,1);
  return cell;
}


//______________________________________________________________________________
FUShmRawCell* FUShmBuffer::rawCellToRead()
{
  waitRawRead();
  unsigned int iCell=nextRawReadIndex();
  FUShmRawCell* cell=rawCell(iCell);
  evt::State_t  state=evtState(iCell);
  assert(state==evt::RAWWRITTEN||state==evt::EMPTY);
  if (state==evt::RAWWRITTEN) setEvtState(iCell,evt::RAWREADING);
  return cell;
}


//______________________________________________________________________________
FUShmRecoCell* FUShmBuffer::recoCellToRead()
{
  waitRecoRead();
  unsigned int   iCell   =nextRecoReadIndex();
  FUShmRecoCell* cell    =recoCell(iCell);
  unsigned int   iRawCell=cell->rawCellIndex();
  if (iRawCell<nRawCells_) {
    evt::State_t   state=evtState(iRawCell);
    assert(state==evt::RECOWRITTEN);
    setEvtState(iRawCell,evt::SENDING);
  }
  return cell;
}


//______________________________________________________________________________
FUShmDqmCell* FUShmBuffer::dqmCellToRead()
{
  waitDqmRead();
  unsigned int  iCell=nextDqmReadIndex();
  FUShmDqmCell* cell=dqmCell(iCell);
  dqm::State_t  state=dqmState(iCell);
  assert(state==dqm::WRITTEN||state==dqm::EMPTY);
  if (state==dqm::WRITTEN) setDqmState(iCell,dqm::SENDING);
  return cell;
}


//______________________________________________________________________________
FUShmRawCell* FUShmBuffer::rawCellToDiscard()
{
  waitRawDiscarded();
  FUShmRawCell* cell=rawCell(rawDiscardIndex_);
  evt::State_t  state=evtState(cell->index());
  assert(state==evt::PROCESSED||state==evt::SENT||state==evt::EMPTY);
  if (state!=evt::EMPTY) setEvtState(cell->index(),evt::DISCARDING);
  return cell;
}


//______________________________________________________________________________
void FUShmBuffer::finishWritingRawCell(FUShmRawCell* cell)
{
  evt::State_t state=evtState(cell->index());
  assert(state==evt::RAWWRITING);
  setEvtState(cell->index(),evt::RAWWRITTEN);
  setEvtNumber(cell->index(),cell->evtNumber());
  postRawIndexToRead(cell->index());
  if (segmentationMode_) shmdt(cell);
  postRawRead();
}


//______________________________________________________________________________
void FUShmBuffer::finishReadingRawCell(FUShmRawCell* cell)
{
  evt::State_t state=evtState(cell->index());
  assert(state==evt::RAWREADING);
  setEvtState(cell->index(),evt::RAWREAD);
  setEvtState(cell->index(),evt::PROCESSING);
  if (segmentationMode_) shmdt(cell);
}


//______________________________________________________________________________
void FUShmBuffer::finishReadingRecoCell(FUShmRecoCell* cell)
{
  unsigned int iRawCell=cell->rawCellIndex();
  if (iRawCell<nRawCells_) {
    evt::State_t state=evtState(cell->rawCellIndex());
    assert(state==evt::SENDING);
    setEvtState(cell->rawCellIndex(),evt::SENT);
  }
  if (segmentationMode_) shmdt(cell);
}


//______________________________________________________________________________
void FUShmBuffer::finishReadingDqmCell(FUShmDqmCell* cell)
{
  dqm::State_t state=dqmState(cell->index());
  assert(state==dqm::SENDING||state==dqm::EMPTY);
  if (state==dqm::SENDING) setDqmState(cell->index(),dqm::SENT);
  if (segmentationMode_) shmdt(cell);
}


//______________________________________________________________________________
void FUShmBuffer::scheduleRawCellForDiscard(unsigned int iCell)
{
  waitRawDiscard();
  if (rawCellReadyForDiscard(iCell)) {
    rawDiscardIndex_=iCell;
    evt::State_t  state=evtState(iCell);
    assert(state==evt::PROCESSING||state==evt::SENT||state==evt::EMPTY);
    if (state==evt::PROCESSING) setEvtState(iCell,evt::PROCESSED);
    postRawDiscarded();
  }
  else postRawDiscard();
}


//______________________________________________________________________________
void FUShmBuffer::discardRawCell(FUShmRawCell* cell)
{
  releaseRawCell(cell);
  postRawDiscard();
}


//______________________________________________________________________________
void FUShmBuffer::discardRecoCell(unsigned int iCell)
{
  FUShmRecoCell* cell=recoCell(iCell);
  unsigned int iRawCell=cell->rawCellIndex();
  if (iRawCell<nRawCells_) {
    evt::State_t state=evtState(iRawCell);
    assert(state==evt::SENT);
    scheduleRawCellForDiscard(iRawCell);
  }
  cell->clear();
  if (segmentationMode_) shmdt(cell);
  postRecoIndexToWrite(iCell);
  postRecoWrite();
}


//______________________________________________________________________________
void FUShmBuffer::discardDqmCell(unsigned int iCell)
{
  dqm::State_t state=dqmState(iCell);
  assert(state==dqm::EMPTY||state==dqm::SENT);
  setDqmState(iCell,dqm::DISCARDING);
  FUShmDqmCell* cell=dqmCell(iCell);
  cell->clear();
  if (segmentationMode_) shmdt(cell);
  setDqmState(iCell,dqm::EMPTY);
  postDqmIndexToWrite(iCell);
  postDqmWrite();
}


//______________________________________________________________________________
void FUShmBuffer::releaseRawCell(FUShmRawCell* cell)
{
  evt::State_t state=evtState(cell->index());
  assert(state==evt::DISCARDING||state==evt::RAWWRITING||state==evt::EMPTY);
  setEvtState(cell->index(),evt::EMPTY);
  setEvtDiscard(cell->index(),0);
  setEvtNumber(cell->index(),0xffffffff);
  setRecoCellId(cell->index(),0xffffffff);
  cell->clear();
  postRawIndexToWrite(cell->index());
  if (segmentationMode_) shmdt(cell);
  postRawWrite();
}


//______________________________________________________________________________
void FUShmBuffer::writeRawEmptyEvent()
{
  FUShmRawCell* cell=rawCellToWrite();
  evt::State_t state=evtState(cell->index());
  assert(state==evt::RAWWRITING);
  setEvtState(cell->index(),evt::EMPTY);
  postRawIndexToRead(cell->index());
  if (segmentationMode_) shmdt(cell);
  postRawRead();
}


//______________________________________________________________________________
void FUShmBuffer::writeRecoEmptyEvent()
{
  waitRecoWrite();
  unsigned int   iCell=nextRecoWriteIndex();
  FUShmRecoCell* cell =recoCell(iCell);
  cell->clear();
  postRecoIndexToRead(iCell);
  if (segmentationMode_) shmdt(cell);
  postRecoRead();
}


//______________________________________________________________________________
void FUShmBuffer::writeDqmEmptyEvent()
{
  waitDqmWrite();
  unsigned int  iCell=nextDqmWriteIndex();
  FUShmDqmCell* cell=dqmCell(iCell);
  cell->clear();
  postDqmIndexToRead(iCell);
  if (segmentationMode_) shmdt(cell);
  postDqmRead();
}


//______________________________________________________________________________
void FUShmBuffer::scheduleRawEmptyCellForDiscard()
{
  FUShmRawCell* cell=rawCellToWrite();
  scheduleRawEmptyCellForDiscard(cell);
}


//______________________________________________________________________________
void FUShmBuffer::scheduleRawEmptyCellForDiscard(FUShmRawCell* cell)
{
  waitRawDiscard();
  if (rawCellReadyForDiscard(cell->index())) {
    rawDiscardIndex_=cell->index();
    setEvtState(cell->index(),evt::EMPTY);
    setEvtNumber(cell->index(),0xffffffff);
    setRecoCellId(cell->index(),0xffffffff);
    if (segmentationMode_) shmdt(cell);
    postRawDiscarded();
  }
  else postRawDiscard();
}


//______________________________________________________________________________
bool FUShmBuffer::writeRecoEventData(unsigned int   runNumber,
				     unsigned int   evtNumber,
				     unsigned char *data,
				     unsigned int   dataSize)
{
  if (dataSize>recoCellPayloadSize_) {
    cout<<"FUShmBuffer::writeRecoEventData() ERROR: buffer overflow."<<endl;
    return false;
  }
  
  waitRecoWrite();
  unsigned int   iCell=nextRecoWriteIndex();
  FUShmRecoCell* cell =recoCell(iCell);
  unsigned int rawCellIndex=indexForEvtNumber(evtNumber);
  evt::State_t state=evtState(rawCellIndex);
  assert(state==evt::PROCESSING);
  setEvtState(rawCellIndex,evt::RECOWRITING);
  setEvtDiscard(rawCellIndex,2);
  setRecoCellId(rawCellIndex,iCell);
  cell->writeEventData(rawCellIndex,runNumber,evtNumber,data,dataSize);
  setEvtState(rawCellIndex,evt::RECOWRITTEN);
  postRecoIndexToRead(iCell);
  if (segmentationMode_) shmdt(cell);
  postRecoRead();
  return true;
}


//______________________________________________________________________________
bool FUShmBuffer::writeRecoInitMsg(unsigned char *data,
				   unsigned int   dataSize)
{
  if (dataSize>recoCellPayloadSize_) {
    cout<<"FUShmBuffer::writeRecoInitMsg() ERROR: buffer overflow."<<endl;
    return false;
  }
  
  waitRecoWrite();
  unsigned int   iCell=nextRecoWriteIndex();
  FUShmRecoCell* cell =recoCell(iCell);
  cell->writeInitMsg(data,dataSize);
  postRecoIndexToRead(iCell);
  if (segmentationMode_) shmdt(cell);
  postRecoRead();
  return true;
}


//______________________________________________________________________________
bool FUShmBuffer::writeDqmEventData(unsigned int   runNumber,
				    unsigned int   evtAtUpdate,
				    unsigned int   folderId,
				    unsigned char *data,
				    unsigned int   dataSize)
{
  if (dataSize>dqmCellPayloadSize_) {
    cout<<"FUShmBuffer::writeDqmEventData() ERROR: buffer overflow."<<endl;
    return false;
  }
  
  waitDqmWrite();
  unsigned int  iCell=nextDqmWriteIndex();
  FUShmDqmCell* cell=dqmCell(iCell);
  dqm::State_t state=dqmState(iCell);
  assert(state==dqm::EMPTY);
  setDqmState(iCell,dqm::WRITING);
  cell->writeData(runNumber,evtAtUpdate,folderId,data,dataSize);
  setDqmState(iCell,dqm::WRITTEN);
  postDqmIndexToRead(iCell);
  if (segmentationMode_) shmdt(cell);
  postDqmRead();
  return true;
}


//______________________________________________________________________________
void FUShmBuffer::sem_print()
{
  cout<<"--> current sem values:"
      <<endl
      <<" lock="<<semctl(semid(),0,GETVAL)
      <<endl
      <<" wraw="<<semctl(semid(),1,GETVAL)
      <<" rraw="<<semctl(semid(),2,GETVAL)
      <<endl
      <<" wdsc="<<semctl(semid(),3,GETVAL)
      <<" rdsc="<<semctl(semid(),4,GETVAL)
      <<endl
      <<" wrec="<<semctl(semid(),5,GETVAL)
      <<" rrec="<<semctl(semid(),6,GETVAL)
      <<endl
      <<" wdqm="<<semctl(semid(),7,GETVAL)
      <<" rdqm="<<semctl(semid(),8,GETVAL)
      <<endl;
}


//______________________________________________________________________________
void FUShmBuffer::printEvtState(unsigned int index)
{
  evt::State_t state=evtState(index);
  std::string stateName;
  if      (state==evt::EMPTY)      stateName="EMPTY";
  else if (state==evt::RAWWRITING) stateName="RAWWRITING";
  else if (state==evt::RAWWRITTEN) stateName="RAWRITTEN";
  else if (state==evt::RAWREADING) stateName="RAWREADING";
  else if (state==evt::RAWREAD)    stateName="RAWREAD";
  else if (state==evt::PROCESSING) stateName="PROCESSING";
  else if (state==evt::PROCESSED)  stateName="PROCESSED";
  else if (state==evt::RECOWRITING)stateName="RECOWRITING";
  else if (state==evt::RECOWRITTEN)stateName="RECOWRITTEN";
  else if (state==evt::SENDING)    stateName="SENDING";
  else if (state==evt::SENT)       stateName="SENT";
  else if (state==evt::DISCARDING) stateName="DISCARDING";
  cout<<"evt "<<index<<" in state '"<<stateName<<"'."<<endl;
}


//______________________________________________________________________________
void FUShmBuffer::printDqmState(unsigned int index)
{
  dqm::State_t state=dqmState(index);
  cout<<"dqm evt "<<index<<" in state '"<<state<<"'."<<endl;
}


//______________________________________________________________________________
FUShmBuffer* FUShmBuffer::createShmBuffer(bool         segmentationMode,
					  unsigned int nRawCells,
					  unsigned int nRecoCells,
					  unsigned int nDqmCells,
					  unsigned int rawCellSize,
					  unsigned int recoCellSize,
					  unsigned int dqmCellSize)
{
  // if necessary, release shared memory first!
  if (FUShmBuffer::releaseSharedMemory())
    cout<<"FUShmBuffer::createShmBuffer: "
	<<"REMOVAL OF OLD SHARED MEM SEGMENTS SUCCESSFULL."
	<<endl;
  
  // create bookkeeping shared memory segment
  int  size =sizeof(unsigned int)*7;
  int  shmid=shm_create(FUShmBuffer::getShmDescriptorKey(),size);if(shmid<0)return 0;
  void*shmAddr=shm_attach(shmid); if(0==shmAddr)return 0;
  
  if(1!=shm_nattch(shmid)) {
    cout<<"FUShmBuffer::createShmBuffer() FAILED: nattch="<<shm_nattch(shmid)<<endl;
    shmdt(shmAddr);
    return 0;
  }
  
  unsigned int* p=(unsigned int*)shmAddr;
  *p++=segmentationMode;
  *p++=nRawCells;
  *p++=nRecoCells;
  *p++=nDqmCells;
  *p++=rawCellSize;
  *p++=recoCellSize;
  *p++=dqmCellSize;
  shmdt(shmAddr);
  
  // create the 'real' shared memory buffer
  size     =FUShmBuffer::size(segmentationMode,
			      nRawCells,nRecoCells,nDqmCells,
			      rawCellSize,recoCellSize,dqmCellSize);
  shmid    =shm_create(FUShmBuffer::getShmKey(),size); if (shmid<0)    return 0;
  int semid=sem_create(FUShmBuffer::getSemKey(),9);    if (semid<0)    return 0;
  shmAddr  =shm_attach(shmid);                         if (0==shmAddr) return 0;
  
  if (1!=shm_nattch(shmid)) {
    cout<<"FUShmBuffer::createShmBuffer FAILED: nattch="<<shm_nattch(shmid)<<endl;
    shmdt(shmAddr);
    return 0;
  }
  FUShmBuffer* buffer=new(shmAddr) FUShmBuffer(segmentationMode,
					       nRawCells,nRecoCells,nDqmCells,
					       rawCellSize,recoCellSize,dqmCellSize);
  
  cout<<"FUShmBuffer::createShmBuffer(): CREATED shared memory buffer."<<endl;
  cout<<"                                segmentationMode="<<segmentationMode<<endl;
  
  buffer->initialize(shmid,semid);
  
  return buffer;
}


//______________________________________________________________________________
FUShmBuffer* FUShmBuffer::getShmBuffer()
{
  // get bookkeeping shared memory segment
  int   size   =sizeof(unsigned int)*7;
  int   shmid  =shm_get(FUShmBuffer::getShmDescriptorKey(),size);if(shmid<0)return 0;
  void* shmAddr=shm_attach(shmid); if (0==shmAddr) return 0;
  
  unsigned int *p=(unsigned int*)shmAddr;
  bool          segmentationMode=*p++;
  unsigned int  nRawCells       =*p++;
  unsigned int  nRecoCells      =*p++;
  unsigned int  nDqmCells       =*p++;
  unsigned int  rawCellSize     =*p++;
  unsigned int  recoCellSize    =*p++;
  unsigned int  dqmCellSize     =*p++;
  shmdt(shmAddr);

  cout<<"FUShmBuffer::getShmBuffer():"
      <<" segmentationMode="<<segmentationMode
      <<" nRawCells="<<nRawCells
      <<" nRecoCells="<<nRecoCells
      <<" nDqmCells="<<nDqmCells
      <<" rawCellSize="<<rawCellSize
      <<" recoCellSize="<<recoCellSize
      <<" dqmCellSize="<<dqmCellSize
      <<endl;
  
  // get the 'real' shared memory buffer
  size     =FUShmBuffer::size(segmentationMode,
			      nRawCells,nRecoCells,nDqmCells,
			      rawCellSize,recoCellSize,dqmCellSize);
  shmid    =shm_get(FUShmBuffer::getShmKey(),size); if (shmid<0)    return 0;
  int semid=sem_get(FUShmBuffer::getSemKey(),9);    if (semid<0)    return 0;
  shmAddr  =shm_attach(shmid);                      if (0==shmAddr) return 0;
  
  if (0==shm_nattch(shmid)) {
    cout<<"FUShmBuffer::getShmBuffer() FAILED: nattch="<<shm_nattch(shmid)<<endl;
    return 0;
  }
  
  FUShmBuffer* buffer=new(shmAddr) FUShmBuffer(segmentationMode,
					       nRawCells,nRecoCells,nDqmCells,
					       rawCellSize,recoCellSize,dqmCellSize);
  
  cout<<"FUShmBuffer::getShmBuffer(): shared memory buffer RETRIEVED."<<endl;
  cout<<"                             segmentationMode="<<segmentationMode<<endl;
  
  return buffer;
}


//______________________________________________________________________________
bool FUShmBuffer::releaseSharedMemory()
{
  // get bookkeeping shared memory segment
  int   size   =sizeof(unsigned int)*7;
  int   shmidd =shm_get(FUShmBuffer::getShmDescriptorKey(),size);if(shmidd<0)return 0;
  void* shmAddr=shm_attach(shmidd); if (0==shmAddr) return false;
  
  unsigned int*p=(unsigned int*)shmAddr;
  bool         segmentationMode=*p++;
  unsigned int nRawCells       =*p++;
  unsigned int nRecoCells      =*p++;
  unsigned int nDqmCells       =*p++;
  unsigned int rawCellSize     =*p++;
  unsigned int recoCellSize    =*p++;
  unsigned int dqmCellSize     =*p++;
  shmdt(shmAddr);

  
  // get the 'real' shared memory segment
  size     =FUShmBuffer::size(segmentationMode,
			      nRawCells,nRecoCells,nDqmCells,
			      rawCellSize,recoCellSize,dqmCellSize);
  int shmid=shm_get(FUShmBuffer::getShmKey(),size);if (shmid<0)    return false;
  int semid=sem_get(FUShmBuffer::getSemKey(),9);   if (semid<0)    return false;
  shmAddr  =shm_attach(shmid);                     if (0==shmAddr) return false;
  
  if (shm_nattch(shmid)>1) {
    cout<<"FUShmBuffer::releaseSharedMemory(): nattch="<<shm_nattch(shmid)
	<<", don't release shared memory."<<endl;
    return false;
  }
  
  if (segmentationMode) {
    FUShmBuffer* buffer=
      new (shmAddr) FUShmBuffer(segmentationMode,
				nRawCells,nRecoCells,nDqmCells,
				rawCellSize,recoCellSize,dqmCellSize);
    int cellid;
    for (unsigned int i=0;i<nRawCells;i++) {
      cellid=shm_get(buffer->rawCellShmKey(i),FUShmRawCell::size(rawCellSize));
      if ((shm_destroy(cellid)==-1)) return false;
    }
    for (unsigned int i=0;i<nRecoCells;i++) {
      cellid=shm_get(buffer->recoCellShmKey(i),FUShmRecoCell::size(recoCellSize));
      if ((shm_destroy(cellid)==-1)) return false;
    }
    for (unsigned int i=0;i<nDqmCells;i++) {
      cellid=shm_get(buffer->dqmCellShmKey(i),FUShmDqmCell::size(dqmCellSize));
      if ((shm_destroy(cellid)==-1)) return false;
    }
  }
  shmdt(shmAddr);

  if (sem_destroy(semid)==-1)  return false;
  if (shm_destroy(shmid)==-1)  return false;
  if (shm_destroy(shmidd)==-1) return false;

  return true;
}


//______________________________________________________________________________
unsigned int FUShmBuffer::size(bool         segmentationMode,
			       unsigned int nRawCells,
			       unsigned int nRecoCells,
			       unsigned int nDqmCells,
			       unsigned int rawCellSize,
			       unsigned int recoCellSize,
			       unsigned int dqmCellSize)
{
  unsigned int offset=
    sizeof(FUShmBuffer)+
    sizeof(unsigned int)*4*nRawCells+
    sizeof(evt::State_t)*nRawCells+
    sizeof(dqm::State_t)*nDqmCells;
  
  unsigned int rawCellTotalSize=FUShmRawCell::size(rawCellSize);
  unsigned int recoCellTotalSize=FUShmRecoCell::size(recoCellSize);
  unsigned int dqmCellTotalSize=FUShmDqmCell::size(dqmCellSize);
  
  unsigned int realSize =
    (segmentationMode) ?
    offset
    +sizeof(key_t)*(nRawCells+nRecoCells+nDqmCells)
    :
    offset
    +rawCellTotalSize*nRawCells
    +recoCellTotalSize*nRecoCells
    +dqmCellTotalSize*nDqmCells;

  unsigned int result=realSize/0x10*0x10 + (realSize%0x10>0)*0x10;
  
  return result;
}


//______________________________________________________________________________
key_t FUShmBuffer::getShmDescriptorKey()
{
  key_t result=ftok(SHM_KEYPATH,SHM_DESCRIPTOR_KEYID);
  if (result==(key_t)-1)
    cout<<"FUShmBuffer::getShmDescriptorKey: ftok() failed!"<<endl;
  return result;
}


//______________________________________________________________________________
key_t FUShmBuffer::getShmKey()
{
  key_t result=ftok(SHM_KEYPATH,SHM_KEYID);
  if (result==(key_t)-1) cout<<"FUShmBuffer::getShmKey: ftok() failed!"<<endl;
  return result;
}


//______________________________________________________________________________
key_t FUShmBuffer::getSemKey()
{
  key_t result=ftok(SEM_KEYPATH,SEM_KEYID);
  if (result==(key_t)-1) cout<<"FUShmBuffer::getSemKey: ftok() failed!"<<endl;
  return result;
}


//______________________________________________________________________________
int FUShmBuffer::shm_create(key_t key,int size)
{
  int shmid=shmget(key,size,IPC_CREAT|0644);
  if (shmid==-1) {
    int err=errno;
    cout<<"FUShmBuffer::shm_create("<<key<<","<<size<<") failed: "
	<<strerror(err)<<endl;
  }
  return shmid;
}


//______________________________________________________________________________
int FUShmBuffer::shm_get(key_t key,int size)
{
  int shmid=shmget(key,size,0644);
  if (shmid==-1) {
    int err=errno;
    cout<<"FUShmBuffer::shm_get("<<key<<","<<size<<") failed: "
	<<strerror(err)<<endl;
  }
  return shmid;
}


//______________________________________________________________________________
void* FUShmBuffer::shm_attach(int shmid)
{
  void* result=shmat(shmid,NULL,0);
  if (0==result) {
    int err=errno;
    cout<<"FUShmBuffer::shm_attach("<<shmid<<") failed: "
	<<strerror(err)<<endl;
  }
  return result;
}


//______________________________________________________________________________
int FUShmBuffer::shm_nattch(int shmid)
{
  shmid_ds shmstat;
  shmctl(shmid,IPC_STAT,&shmstat);
  return shmstat.shm_nattch;
}


//______________________________________________________________________________
int FUShmBuffer::shm_destroy(int shmid)
{
  shmid_ds shmstat;
  int result=shmctl(shmid,IPC_RMID,&shmstat);
  if (result==-1) cout<<"FUShmBuffer::shm_destroy(shmid="<<shmid<<") failed."<<endl;
  return result;
}


//______________________________________________________________________________
int FUShmBuffer::sem_create(key_t key,int nsem)
{
  int semid=semget(key,nsem,IPC_CREAT|0666);
  if (semid<0) {
    int err=errno;
    cout<<"FUShmBuffer::sem_create(key="<<key<<",nsem="<<nsem<<") failed: "
	<<strerror(err)<<endl;
  }
  return semid;
}


//______________________________________________________________________________
int FUShmBuffer::sem_get(key_t key,int nsem)
{
  int semid=semget(key,nsem,0666);
  if (semid<0) {
    int err=errno;
    cout<<"FUShmBuffer::sem_get(key="<<key<<",nsem="<<nsem<<") failed: "
	<<strerror(err)<<endl;
  }
  return semid;
}


//______________________________________________________________________________
int FUShmBuffer::sem_destroy(int semid)
{
  int result=semctl(semid,0,IPC_RMID);
  if (result==-1) cout<<"FUShmBuffer::sem_destroy(semid="<<semid<<") failed."<<endl;
  return result;
}



////////////////////////////////////////////////////////////////////////////////
// implementation of private member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
unsigned int FUShmBuffer::nextIndex(unsigned int  offset,
				    unsigned int  nCells,
				    unsigned int& iNext)
{
  lock();
  unsigned int* pindex=(unsigned int*)((unsigned int)this+offset);
  pindex+=iNext;
  iNext=(iNext+1)%nCells;
  unsigned int result=*pindex;
  unlock();
  return result;
}


//______________________________________________________________________________
void FUShmBuffer::postIndex(unsigned int  index,
			    unsigned int  offset,
			    unsigned int  nCells,
			    unsigned int& iLast)
{
  lock();
  unsigned int* pindex=(unsigned int*)((unsigned int)this+offset);
  pindex+=iLast;
  *pindex=index;
  iLast=(iLast+1)%nCells;
  unlock();
}


//______________________________________________________________________________
unsigned int FUShmBuffer::nextRawWriteIndex()
{
  return nextIndex(rawWriteOffset_,nRawCells_,rawWriteNext_);
}


//______________________________________________________________________________
unsigned int FUShmBuffer::nextRawReadIndex()
{
  return nextIndex(rawReadOffset_,nRawCells_,rawReadNext_);
}


//______________________________________________________________________________
void FUShmBuffer::postRawIndexToWrite(unsigned int index)
{
  postIndex(index,rawWriteOffset_,nRawCells_,rawWriteLast_);
}


//______________________________________________________________________________
void FUShmBuffer::postRawIndexToRead(unsigned int index)
{
  postIndex(index,rawReadOffset_,nRawCells_,rawReadLast_);
}


//______________________________________________________________________________
unsigned int FUShmBuffer::nextRecoWriteIndex()
{
  return nextIndex(recoWriteOffset_,nRecoCells_,recoWriteNext_);
}


//______________________________________________________________________________
unsigned int FUShmBuffer::nextRecoReadIndex()
{
  return nextIndex(recoReadOffset_,nRecoCells_,recoReadNext_);
}


//______________________________________________________________________________
void FUShmBuffer::postRecoIndexToWrite(unsigned int index)
{
  postIndex(index,recoWriteOffset_,nRecoCells_,recoWriteLast_);
}


//______________________________________________________________________________
void FUShmBuffer::postRecoIndexToRead(unsigned int index)
{
  postIndex(index,recoReadOffset_,nRecoCells_,recoReadLast_);
}


//______________________________________________________________________________
unsigned int FUShmBuffer::nextDqmWriteIndex()
{
  return nextIndex(dqmWriteOffset_,nDqmCells_,dqmWriteNext_);
}


//______________________________________________________________________________
unsigned int FUShmBuffer::nextDqmReadIndex()
{
  return nextIndex(dqmReadOffset_,nDqmCells_,dqmReadNext_);
}


//______________________________________________________________________________
void FUShmBuffer::postDqmIndexToWrite(unsigned int index)
{
  postIndex(index,dqmWriteOffset_,nDqmCells_,dqmWriteLast_);
}


//______________________________________________________________________________
void FUShmBuffer::postDqmIndexToRead(unsigned int index)
{
  postIndex(index,dqmReadOffset_,nDqmCells_,dqmReadLast_);
}


//______________________________________________________________________________
unsigned int FUShmBuffer::indexForEvtNumber(unsigned int evtNumber)
{
  unsigned int *pevt=(unsigned int*)((unsigned int)this+evtNumberOffset_);
  for (unsigned int i=0;i<nRawCells_;i++) {
    if ((*pevt++)==evtNumber) return i;
  }
  assert(false);
  return 0xffffffff;
}


//______________________________________________________________________________
evt::State_t FUShmBuffer::evtState(unsigned int index)
{
  assert(index<nRawCells_);
  evt::State_t *pstate=(evt::State_t*)((unsigned int)this+evtStateOffset_);
  pstate+=index;
  return *pstate;
}


//______________________________________________________________________________
unsigned int FUShmBuffer::evtNumber(unsigned int index)
{
  assert(index<nRawCells_);
  unsigned int *pevt=(unsigned int*)((unsigned int)this+evtNumberOffset_);
  pevt+=index;
  return *pevt;
}


//______________________________________________________________________________
unsigned int FUShmBuffer::evtRecoId(unsigned int index)
{
  assert(index<nRawCells_);
  unsigned int *precoid=(unsigned int*)((unsigned int)this+evtRecoIdOffset_);
  precoid+=index;
  return *precoid;
}


//______________________________________________________________________________
dqm::State_t FUShmBuffer::dqmState(unsigned int index)
{
  assert(index<nDqmCells_);
  dqm::State_t *pstate=(dqm::State_t*)((unsigned int)this+dqmStateOffset_);
  pstate+=index;
  return *pstate;
}


//______________________________________________________________________________
bool FUShmBuffer::setEvtState(unsigned int index,evt::State_t state)
{
  assert(index<nRawCells_);
  evt::State_t *pstate=(evt::State_t*)((unsigned int)this+evtStateOffset_);
  pstate+=index;
  lock();
  *pstate=state;
  unlock();
  return true;
}


//______________________________________________________________________________
bool FUShmBuffer::setEvtDiscard(unsigned int index,unsigned int discard)
{
  assert(index<nRawCells_);
  unsigned int *pcount=(unsigned int*)((unsigned int)this+evtDiscardOffset_);
  pcount+=index;
  lock();
  *pcount=discard;
  unlock();
  return true;
}


//______________________________________________________________________________
bool FUShmBuffer::setEvtNumber(unsigned int index,unsigned int evtNumber)
{
  assert(index<nRawCells_);
  unsigned int *pevt=(unsigned int*)((unsigned int)this+evtNumberOffset_);
  pevt+=index;
  lock();
  *pevt=evtNumber;
  unlock();
  return true;
}


//______________________________________________________________________________
bool FUShmBuffer::setRecoCellId(unsigned int index,unsigned int recoCellId)
{
  assert(index<nRawCells_);
  unsigned int *precoid=(unsigned int*)((unsigned int)this+evtRecoIdOffset_);
  precoid+=index;
  lock();
  *precoid=recoCellId;
  unlock();
  return true;
}


//______________________________________________________________________________
bool FUShmBuffer::setDqmState(unsigned int index,dqm::State_t state)
{
  assert(index<nDqmCells_);
  dqm::State_t *pstate=(dqm::State_t*)((unsigned int)this+dqmStateOffset_);
  pstate+=index;
  lock();
  *pstate=state;
  unlock();
  return true;
}


//______________________________________________________________________________
FUShmRawCell* FUShmBuffer::rawCell(unsigned int iCell)
{
  FUShmRawCell* result(0);
  
  if (iCell>=nRawCells_) {
    cout<<"FUShmBuffer::rawCell("<<iCell<<") ERROR: "
	<<"iCell="<<iCell<<" >= nRawCells()="<<nRawCells_<<endl;
    return result;
  }
  
  if (segmentationMode_) {
    key_t         shmkey  =rawCellShmKey(iCell);
    int           shmid   =shm_get(shmkey,rawCellTotalSize_);
    void*         cellAddr=shm_attach(shmid);
    result=new (cellAddr) FUShmRawCell(rawCellPayloadSize_);
  }
  else {
    result=
      (FUShmRawCell*)((unsigned int)this+rawCellOffset_+iCell*rawCellTotalSize_);
  }
  
  return result;
}


//______________________________________________________________________________
FUShmRecoCell* FUShmBuffer::recoCell(unsigned int iCell)
{
  FUShmRecoCell* result(0);
  
  if (iCell>=nRecoCells_) {
    cout<<"FUShmBuffer::recoCell("<<iCell<<") ERROR: "
	<<"iCell="<<iCell<<" >= nRecoCells="<<nRecoCells_<<endl;
    return result;
  }
  
  if (segmentationMode_) {
    key_t         shmkey  =recoCellShmKey(iCell);
    int           shmid   =shm_get(shmkey,recoCellTotalSize_);
    void*         cellAddr=shm_attach(shmid);
    result=new (cellAddr) FUShmRecoCell(recoCellPayloadSize_);
  }
  else {
    result=
      (FUShmRecoCell*)((unsigned int)this+recoCellOffset_+iCell*recoCellTotalSize_);
  }
  
  return result;
}


//______________________________________________________________________________
FUShmDqmCell* FUShmBuffer::dqmCell(unsigned int iCell)
{
  FUShmDqmCell* result(0);
  
  if (iCell>=nDqmCells_) {
    cout<<"FUShmBuffer::dqmCell("<<iCell<<") ERROR: "
	<<"iCell="<<iCell<<" >= nDqmCells="<<nDqmCells_<<endl;
    return result;
  }
  
  if (segmentationMode_) {
    key_t         shmkey  =dqmCellShmKey(iCell);
    int           shmid   =shm_get(shmkey,dqmCellTotalSize_);
    void*         cellAddr=shm_attach(shmid);
    result=new (cellAddr) FUShmDqmCell(dqmCellPayloadSize_);
  }
  else {
    result=
      (FUShmDqmCell*)((unsigned int)this+dqmCellOffset_+iCell*dqmCellTotalSize_);
  }
  
  return result;
}


//______________________________________________________________________________
bool FUShmBuffer::rawCellReadyForDiscard(unsigned int index)
{
  assert(index<nRawCells_);
  unsigned int *pcount=(unsigned int*)((unsigned int)this+evtDiscardOffset_);
  pcount+=index;
  lock();
  assert(*pcount>0);
  --(*pcount);
  bool result=(*pcount==0);
  unlock();
  return result;
}


//______________________________________________________________________________
key_t FUShmBuffer::shmKey(unsigned int iCell,unsigned int offset)
{
  if (!segmentationMode_) {
    cout<<"FUShmBuffer::shmKey() ERROR: only valid in segmentationMode!"<<endl;
    return -1;
  }
  key_t* addr=(key_t*)((unsigned int)this+offset);
  for (unsigned int i=0;i<iCell;i++) ++addr;
  return *addr;
}


//______________________________________________________________________________
key_t FUShmBuffer::rawCellShmKey(unsigned int iCell)
{
  if (iCell>=nRawCells_) {
    cout<<"FUShmBuffer::rawCellShmKey() ERROR: "
	<<"iCell>=nRawCells: "<<iCell<<">="<<nRawCells_<<endl;
    return -1;
  }
  return shmKey(iCell,rawCellOffset_);
}


//______________________________________________________________________________
key_t FUShmBuffer::recoCellShmKey(unsigned int iCell)
{
  if (iCell>=nRecoCells_) {
    cout<<"FUShmBuffer::recoCellShmKey() ERROR: "
	<<"iCell>=nRecoCells: "<<iCell<<">="<<nRecoCells_<<endl;
    return -1;
  }
  return shmKey(iCell,recoCellOffset_);
}


//______________________________________________________________________________
key_t FUShmBuffer::dqmCellShmKey(unsigned int iCell)
{
  if (iCell>=nDqmCells_) {
    cout<<"FUShmBuffer::dqmCellShmKey() ERROR: "
	<<"iCell>=nDqmCells: "<<iCell<<">="<<nDqmCells_<<endl;
    return -1;
  }
  return shmKey(iCell,dqmCellOffset_);
}


//______________________________________________________________________________
void FUShmBuffer::sem_init(int isem,int value)
{
  if (semctl(semid(),isem,SETVAL,value)<0) {
    cout<<"FUShmBuffer: FATAL ERROR in semaphore initialization."<<endl;
  }
}


//______________________________________________________________________________
void FUShmBuffer::sem_wait(int isem)
{
  struct sembuf sops[1];
  sops[0].sem_num=isem;
  sops[0].sem_op =  -1;
  sops[0].sem_flg=   0;
  if (semop(semid(),sops,1)==-1) {
    cout<<"FUShmBuffer: ERROR in semaphore operation sem_wait."<<endl;
  }
}


//______________________________________________________________________________
void FUShmBuffer::sem_post(int isem)
{
  struct sembuf sops[1];
  sops[0].sem_num=isem;
  sops[0].sem_op =   1;
  sops[0].sem_flg=   0;
  if (semop(semid(),sops,1)==-1) {
    cout<<"FUShmBuffer: ERROR in semaphore operation sem_post."<<endl;
  }
}
