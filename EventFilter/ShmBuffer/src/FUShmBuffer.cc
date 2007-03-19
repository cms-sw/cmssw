////////////////////////////////////////////////////////////////////////////////
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
  
  if (segmentationMode_) {
    rawCellOffset_ =sizeof(FUShmBuffer);
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
    rawCellOffset_ =sizeof(FUShmBuffer);
    recoCellOffset_=rawCellOffset_+nRawCells_*rawCellTotalSize_;
    recoCellOffset_=recoCellOffset_+nRecoCells_*recoCellTotalSize_;
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
  cout<<"FUShmBuffer::intialize"<<endl;
  shmid_         =shmid;
  semid_         =semid;
  rawWriteIndex_ =    0;
  rawReadIndex_  =    0;
  recoWriteIndex_=    0;
  recoReadIndex_ =    0;
  dqmWriteIndex_ =    0;
  dqmReadIndex_  =    0;
  
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
  lock();
  FUShmRawCell* result=rawCell(rawWriteIndex_);
  while (result->isWritten()||result->isProcessing()) {
    result->skip();
    if (result->nSkip()>NSKIP_MAX) result->setStateDead();
    else {
      if (segmentationMode_) shmdt(result);
      rawWriteIndex_=(rawWriteIndex_+1)%nRawCells_;
      result        =rawCell(rawWriteIndex_);
    }
  }
  result->resetSkip();
  rawWriteIndex_=(rawWriteIndex_+1)%nRawCells_;
  unlock();
  return result;
}


//______________________________________________________________________________
FUShmRawCell* FUShmBuffer::rawCellToRead()
{
  waitRawRead();
  lock();
  FUShmRawCell* result=rawCell(rawReadIndex_);
  result->setStateProcessing();
  rawReadIndex_=(rawReadIndex_+1)%nRawCells_;
  unlock();
  return result;
}


//______________________________________________________________________________
FUShmRawCell* FUShmBuffer::rawCellToDiscard()
{
  waitRawDiscarded();
  lock();
  FUShmRawCell* result=rawCell(rawDiscardIndex_);
  unlock();
  return result;
}


//______________________________________________________________________________
void FUShmBuffer::finishWritingRawCell(FUShmRawCell* cell)
{
  lock();
  cell->setStateWritten();
  if (segmentationMode_) shmdt(cell);
  unlock();
  postRawRead();
}

//______________________________________________________________________________
void FUShmBuffer::finishReadingRawCell(FUShmRawCell* cell)
{
  if (segmentationMode_) shmdt(cell);
}

//______________________________________________________________________________
void FUShmBuffer::scheduleRawCellForDiscard(unsigned int iCell)
{
  waitRawDiscard();
  lock();
  rawDiscardIndex_=iCell;
  FUShmRawCell* cell=rawCell(iCell);
  cell->setStateProcessed();
  if (segmentationMode_) shmdt(cell);
  unlock();
  postRawDiscarded();
}


//______________________________________________________________________________
void FUShmBuffer::discardRawCell(FUShmRawCell* cell)
{
  lock();
  cell->setStateEmpty();
  if (segmentationMode_) shmdt(cell);
  unlock();
  postRawWrite();
  postRawDiscard();
}


//______________________________________________________________________________
int FUShmBuffer::nbRecoCellsToWrite() const
{
  return semctl(semid(),5,GETVAL);
}


//______________________________________________________________________________
int FUShmBuffer::nbRecoCellsToRead() const
{
  return semctl(semid(),6,GETVAL);
}


//______________________________________________________________________________
FUShmRecoCell* FUShmBuffer::recoCellToWrite()
{
  waitRecoWrite();
  lock();
  FUShmRecoCell* result=recoCell(recoWriteIndex_);
  result->setStateWriting();
  recoWriteIndex_=(recoWriteIndex_+1)%nRecoCells_;
  unlock();
  return result;
}


//______________________________________________________________________________
FUShmRecoCell* FUShmBuffer::recoCellToRead()
{
  waitRecoRead();
  lock();
  FUShmRecoCell* result=recoCell(recoReadIndex_);
  result->setStateSending();
  recoReadIndex_=(recoReadIndex_+1)%nRecoCells_;
  unlock();
  return result;
}


//______________________________________________________________________________
int FUShmBuffer::nbDqmCellsToWrite() const
{
  return semctl(semid(),7,GETVAL);
}


//______________________________________________________________________________
int FUShmBuffer::nbDqmCellsToRead() const
{
  return semctl(semid(),8,GETVAL);
}


//______________________________________________________________________________
FUShmDqmCell* FUShmBuffer::dqmCellToWrite()
{
  waitDqmWrite();
  lock();
  FUShmDqmCell* result=dqmCell(dqmWriteIndex_);
  result->setStateWriting();
  dqmWriteIndex_=(dqmWriteIndex_+1)%nDqmCells_;
  unlock();
  return result;
}


//______________________________________________________________________________
FUShmDqmCell* FUShmBuffer::dqmCellToRead()
{
  waitDqmRead();
  lock();
  FUShmDqmCell* result=dqmCell(dqmReadIndex_);
  result->setStateSending();
  dqmReadIndex_=(dqmReadIndex_+1)%nDqmCells_;
  unlock();
  return result;
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
  
  if (1==shm_nattch(shmid)) {
    cout<<"FUShmBuffer::getShmBuffer FAILED: nattch="<<shm_nattch(shmid)<<endl;
    shmdt(shmAddr);
    return 0;
  }
  
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
  int   shmid  =shm_get(FUShmBuffer::getShmDescriptorKey(),size);if(shmid<0)return 0;
  void* shmAddr=shm_attach(shmid); if (0==shmAddr) return false;
  
  unsigned int*p=(unsigned int*)shmAddr;
  bool         segmentationMode=*p++;
  unsigned int nRawCells       =*p++;
  unsigned int nRecoCells      =*p++;
  unsigned int nDqmCells       =*p++;
  unsigned int rawCellSize     =*p++;
  unsigned int recoCellSize    =*p++;
  unsigned int dqmCellSize     =*p++;
  shmdt(shmAddr);
  if (shm_destroy(shmid)==-1) return false;
  
  // get the 'real' shared memory segment
  size     =FUShmBuffer::size(segmentationMode,
			      nRawCells,nRecoCells,nDqmCells,
			      rawCellSize,recoCellSize,dqmCellSize);
  shmid    =shm_get(FUShmBuffer::getShmKey(),size);if (shmid<0)    return false;
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
  
  if (sem_destroy(semid)==-1)  return false;
  if (shm_destroy(shmid)==-1)  return false;

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
  unsigned int offset=sizeof(FUShmBuffer);
  unsigned int rawCellTotalSize=FUShmRawCell::size(rawCellSize);
  unsigned int recoCellTotalSize=FUShmRecoCell::size(recoCellSize);
  unsigned int dqmCellTotalSize=FUShmDqmCell::size(dqmCellSize);
  unsigned int realSize =
    (segmentationMode) ?
    offset+sizeof(key_t)*(nRawCells+nRecoCells+nDqmCells) :
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
  int shmid=shmget(key,size,IPC_CREAT|0666);
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
  int shmid=shmget(key,size,0666);
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
key_t FUShmBuffer::shmKey(unsigned int iCell,unsigned int offset)
{
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
