////////////////////////////////////////////////////////////////////////////////
//
// FUShmBuffer
// -----------
//
//            15/11/2006 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////


#include "EventFilter/ResourceBroker/interface/FUShmBuffer.h"

#include <iostream>
#include <string>


#define SHM_KEYPATH     "/dev/null"    /* Path used on ftok for semget key  */
#define SHM_KEYID1      1              /* Id used on ftok for 1. semget key */
#define SHM_KEYID2      2              /* Id used on ftok for 2. semget key */
#define SEM_KEYPATH     "/dev/null"    /* Path used on ftok for semget key  */
#define SEM_KEYID       1              /* Id used on ftok for semget key    */

#define NSKIP_MAX       100

using namespace std;
using namespace evf;


////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
FUShmBuffer::FUShmBuffer(int shmid,int semid,
			 unsigned int  nCell,
			 unsigned int  cellBufferSize,
			 unsigned int  nFed,
			 unsigned int  nSuperFrag,
			 bool          ownsMemory)
  : ownsMemory_(ownsMemory)
  , shmid_(shmid)
  , semid_(semid)
  , nCell_(nCell)
  , cellBufferSize_(cellBufferSize)
  , nFed_(nFed)
  , nSuperFrag_(nSuperFrag)
  , cellOffset_(0)
  , cellSize_(0)
  // don't initialize readIndex_/writeIndex_, might have been set by another prc!
{
  cellSize_=FUShmBufferCell::size(cellBufferSize_,nFed_,nSuperFrag_);
  
  if (ownsMemory_) {
    unsigned char* buffer=new unsigned char[nCell_*cellSize_];
    cellOffset_=(unsigned int)buffer-(unsigned int)this;
  }
  else {
    cellOffset_=sizeof(FUShmBuffer);
  }
  
  for (unsigned int i=0;i<nCell_;i++) {
    void* addr=(void*)((unsigned int)this+cellOffset_+i*cellSize_);
    new(addr) FUShmBufferCell(i,cellBufferSize_,nFed_,nSuperFrag_,false);
  }
}


//______________________________________________________________________________
FUShmBuffer::~FUShmBuffer()
{
  if (ownsMemory()) {
    unsigned char* buffer=(unsigned char*)((unsigned int)this+cellOffset_);
    if (0!=buffer) delete [] buffer;
  }
  else {
    shmdt((void*)this);
  }
}


////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
FUShmBufferCell* FUShmBuffer::cell(unsigned int i)
{
  if (i>=nCell()) {
    cout<<"FUShmBuffer::cell("<<i<<") Error. nCell()="<<nCell()<<endl;
    return 0;
  }
  
  FUShmBufferCell*
    cell=(FUShmBufferCell*)((unsigned int)this+cellOffset_+i*cellSize_);
  return cell;
}


//______________________________________________________________________________
FUShmBufferCell* FUShmBuffer::currentWriterCell()
{
  FUShmBufferCell* result=cell(writeIndex_);
  while (result->isWritten()||result->isProcessing()) {
    result->skip();
    if (result->nSkip()>NSKIP_MAX) result->setStateDead();
    else {
      writeIndex_=(writeIndex_+1)%nCell_;
      result     = cell(writeIndex_);
    }
  }
  result->resetSkip();
  writeIndex_=(writeIndex_+1)%nCell_;
  return result;
}


//______________________________________________________________________________
FUShmBufferCell* FUShmBuffer::currentReaderCell()
{
  FUShmBufferCell* result=cell(readIndex_);
  readIndex_=(readIndex_+1)%nCell_;
  return result;
}


//______________________________________________________________________________
FUShmBufferCell* FUShmBuffer::cellToBeDiscarded()
{
  waitDiscardedSem();
  return cell(cellIndexToBeDiscarded_);
}


//______________________________________________________________________________
void FUShmBuffer::scheduleForDiscard(FUShmBufferCell* cell)
{
  waitDiscardSem();
  cellIndexToBeDiscarded_ = cell->index();
  postDiscardedSem();
}


//______________________________________________________________________________
void FUShmBuffer::initialize()
{
  // this is supposed to be called by the writer prc only, initialize!
  readIndex_ =0;
  writeIndex_=0;
  
  for (unsigned int i=0;i<nCell();i++) cell(i)->clear();
  
  // setup semaphores
  sem_init(0,nCell()); // writer semaphore
  sem_init(1,0);       // reader semaphore
  sem_init(2,1);       // lock (binary)
  sem_init(3,0);       // binary semaphore to read  next buID to be discarded
  sem_init(4,1);       // binary semaphore to write next buID to be discarded
  sem_print();         // print values
}


//______________________________________________________________________________
int FUShmBuffer::writerSemValue() const
{
  return semctl(semid(),0,GETVAL);
}


//______________________________________________________________________________
int FUShmBuffer::readerSemValue() const
{
  return semctl(semid(),1,GETVAL);
}


//______________________________________________________________________________
void FUShmBuffer::sem_print()
{
  cout<<"--> current sem values:"
      <<" wsem="<<semctl(semid(),0,GETVAL)
      <<" rsem="<<semctl(semid(),1,GETVAL)
      <<" mutx="<<semctl(semid(),2,GETVAL)
      <<" dscr="<<semctl(semid(),3,GETVAL)
      <<" dscw="<<semctl(semid(),4,GETVAL)
      <<endl;
}


//______________________________________________________________________________
void FUShmBuffer::print(int verbose)
{
  unsigned int size=FUShmBuffer::size(nCell(),
				      nFed(),
				      nSuperFrag(),
				      cellBufferSize());

  cout<<"shmid="      <<shmid()
      <<" semid="     <<semid()
      <<" ownsMeory=" <<ownsMemory()
      <<" readIndex=" <<readIndex()
      <<" writeIndex="<<writeIndex()
      <<endl
      <<"nCell="          <<nCell()
      <<" nFed="          <<nFed()
      <<" nSuperFrag="    <<nSuperFrag()
      <<" cellBufferSize="<<cellBufferSize()
      <<" cellOffset="    <<cellOffset()
      <<" cellSize="      <<cellSize()
      <<endl
      <<"this=0x"<<hex<<(int)this<<dec
      <<" size="<<size
      <<endl;

  if (verbose>0)
    for (unsigned int i=0;i<nCell_;i++) cell(i)->print();
}


//______________________________________________________________________________
unsigned int FUShmBuffer::size(unsigned int nCell,
			       unsigned int cellBufferSize,	
			       unsigned int nFed,
			       unsigned int nSuperFrag)
{
  unsigned int offset  =sizeof(FUShmBuffer);
  unsigned int cellSize=FUShmBufferCell::size(cellBufferSize,nFed,nSuperFrag);
  unsigned int realSize=offset+cellSize*nCell;
  unsigned int result  =realSize/0x10*0x10 + (realSize%0x10>0)*0x10;
  return result;
}


//______________________________________________________________________________
FUShmBuffer* FUShmBuffer::createShmBuffer(unsigned int nCell,
					  unsigned int cellBufferSize,
					  unsigned int nFed,
					  unsigned int nSuperFrag)
  
{
  // if necessary, release shared memory first!
  if (FUShmBuffer::releaseSharedMemory())
    cout<<"FUShmBuffer::createShmBuffer: "
	<<"REMOVAL OF OLD SHARED MEM SEGMENTS SUCCESSFULL."
	<<endl;

  // create bookkeeping shared memory segment
  int size1 =sizeof(unsigned int)*4;
  int shmid1=shm_create(FUShmBuffer::getShmKey1(),size1); if (shmid1<0) return 0;

  // attach bookeeping segment to the address space of this prc
  void* shmAddr1=shm_attach(shmid1); if (0==shmAddr1) return 0;

  // check if this is indeed the creator of the segment
  if(1!=shm_nattch(shmid1)) {
    cout<<"FUShmBuffer::createShmBuffer() FAILED: nattch="<<shm_nattch(shmid1)<<endl;
    shmdt(shmAddr1);
    return 0;
  }
  
  // store buffer parameters in segment
  cout<<"FUShmBuffer::createShmBuffer():"
      <<" nCell="<<nCell
      <<" cellBufferSize="<<cellBufferSize
      <<" nFed="<<nFed
      <<" nSuperFrag="<<nSuperFrag
      <<endl;
  unsigned int* p=(unsigned int*)shmAddr1;
  *p++=nCell; *p++=cellBufferSize; *p++=nFed; *p=nSuperFrag;
  
  // create the 'real' shared memory buffer
  int size =FUShmBuffer::size(nCell,cellBufferSize,nFed,nSuperFrag);
  int shmid=shm_create(FUShmBuffer::getShmKey(),size); if (shmid<0) return 0;
  
  // attach the 'real' segment to the address space of this prc
  unsigned char* shmAddr=(unsigned char*)shm_attach(shmid);if (0==shmAddr) return 0;
  
  // check that this is indeed the creator of the segment
  if (1!=shm_nattch(shmid)) {
    cout<<"FUShmBuffer::createShmBuffer FAILED: nattch="<<shm_nattch(shmid1)<<endl;
    shmdt(shmAddr);
    return 0;
  }
  
  // create semaphore set to control buffer access
  int semid=sem_create(FUShmBuffer::getSemKey(),5); if (semid<0) return 0;
  
  // allocate the shared memory buffer using a 'placement new'
  FUShmBuffer* buffer=new(shmAddr) FUShmBuffer(shmid,semid,
					       nCell,
					       cellBufferSize,
					       nFed,
					       nSuperFrag,
					       false);
  cout<<"FUShmBuffer::createShmBuffer(): created shared memory buffer."<<endl;
  
  // as the creator, intitialize the buffer
  buffer->initialize();
  
  return buffer;
}


//______________________________________________________________________________
FUShmBuffer* FUShmBuffer::getShmBuffer()
{
  // get bookkeeping shared memory segment
  int size1 =sizeof(unsigned int)*4;
  int shmid1=shm_get(FUShmBuffer::getShmKey1(),size1); if (shmid1<0) return 0;

  // attach bookeeping segment to the address space of this prc
  void* shmAddr1=shm_attach(shmid1); if (0==shmAddr1) return 0;

  // check that a creator is atached to the segment
  if (1==shm_nattch(shmid1)) {
    cout<<"FUShmBuffer::getShmBuffer FAILED: nattch="<<shm_nattch(shmid1)<<endl;
    shmdt(shmAddr1);
    return 0;
  }
  
  // retrieve buffer parameters
  unsigned int*p             =(unsigned int*)shmAddr1;
  unsigned int nCell         =*p++;
  unsigned int cellBufferSize=*p++;
  unsigned int nFed          =*p++;
  unsigned int nSuperFrag    =*p;
  cout<<"FUShmBuffer::getShmBuffer():"
      <<" nCell="<<nCell
      <<" cellBufferSize="<<cellBufferSize  
      <<" nFed="<<nFed
      <<" nSuperFrag="<<nSuperFrag
      <<endl;
  
  // no reason to stay attached to this segment
  shmdt(shmAddr1);
  
  // get the 'real' shared memory buffer
  int size =FUShmBuffer::size(nCell,cellBufferSize,nFed,nSuperFrag);
  int shmid=shm_get(FUShmBuffer::getShmKey(),size); if (shmid<0) return 0;
  
  // attach the 'real' segment to the address space of this prc
  unsigned char* shmAddr=(unsigned char*)shm_attach(shmid);if (0==shmAddr) return 0;
  
  // check that a creator is atached to the segment
  if (1==shm_nattch(shmid)) {
    cout<<"FUShmBuffer::getShmBuffer() FAILED: nattch="<<shm_nattch(shmid)<<endl;
    shmdt(shmAddr);
    return 0;
  }
  
  // get semaphore set to control buffer access
  int semid=sem_get(FUShmBuffer::getSemKey(),5); if (semid<0) return 0;
  
  // allocate the shared memory buffer using a 'placement new'
  FUShmBuffer* buffer=new(shmAddr) FUShmBuffer(shmid,semid,
					       nCell,
					       cellBufferSize,
					       nFed,
					       nSuperFrag,
					       false);
  cout<<"FUShmBuffer::getShmBuffer(): got shared memory buffer."<<endl;
  
  return buffer;
}


//______________________________________________________________________________
bool FUShmBuffer::releaseSharedMemory()
{
  // get bookkeeping shared memory segment
  int size1 =sizeof(unsigned int)*4;
  int shmid1=shm_get(FUShmBuffer::getShmKey1(),size1);
  if (shmid1<0) return false;
  
  // attach bookeeping segment to the address space of this prc
  void* shmAddr1=shm_attach(shmid1);
  if (0==shmAddr1) return false;
  
  // check that a creator is atached to the segment
  if (1!=shm_nattch(shmid1)) {
    cout<<"FUShmBuffer::releaseSharedMemory(): nattch="<<shm_nattch(shmid1)
	<<", don't release shared memory."<<endl;
    shmdt(shmAddr1);
    return false;
  }
  
  // retrieve buffer parameters
  unsigned int*p             =(unsigned int*)shmAddr1;
  unsigned int nCell         =*p++;
  unsigned int cellBufferSize=*p++;
  unsigned int nFed          =*p++;
  unsigned int nSuperFrag    =*p;
  cout<<"FUShmBuffer::releaseSharedMemory():"
      <<" nCell="<<nCell
      <<" cellBufferSize="<<cellBufferSize  
      <<" nFed="<<nFed
      <<" nSuperFrag="<<nSuperFrag
      <<endl;
  
  // no reason to stay attached to this segment
  shmdt(shmAddr1);
  
  // get the 'real' shared memory buffer
  int size =FUShmBuffer::size(nCell,cellBufferSize,nFed,nSuperFrag);
  int shmid=shm_get(FUShmBuffer::getShmKey(),size);
  if (shmid<0) return false;
  
  // check that a creator is atached to the segment
  if (0!=shm_nattch(shmid)) {
    cout<<"FUShmBuffer::releaseSharedMemory(): nattch="<<shm_nattch(shmid)
	<<", don't release shared memory."<<endl;
    return false;
  }
  
  // get semaphore set to control buffer access
  int semid=sem_get(FUShmBuffer::getSemKey(),3);
  if (semid<0) return false;
  
  if (sem_destroy(semid)==-1)  return false;
  if (shm_destroy(shmid)==-1)  return false;
  if (shm_destroy(shmid1)==-1) return false;

  return true;
}


//______________________________________________________________________________
key_t FUShmBuffer::getShmKey1()
{
  key_t result=ftok(SHM_KEYPATH,SHM_KEYID1);
  if (result==(key_t)-1) {
    cout<<"FUShmBuffer::getShmKey1: ftok() failed!"<<endl;
    return -1;
  }
  return result;
}


//______________________________________________________________________________
key_t FUShmBuffer::getShmKey()
{
  key_t result=ftok(SHM_KEYPATH,SHM_KEYID2);
  if (result==(key_t)-1) {
    cout<<"FUShmBuffer::getShmKey: ftok() failed!"<<endl;
    return -1;
  }
  return result;
}


//______________________________________________________________________________
key_t FUShmBuffer::getSemKey()
{
  key_t result=ftok(SEM_KEYPATH,SEM_KEYID);
  if (result==(key_t)-1) {
    cout<<"FUShmBuffer::getSemKey: ftok() failed!"<<endl;
    return -1;
  }
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
  else {
    cout<<"FUShmBuffer::shm_create(key="<<key<<",size="<<size<<") successful. "
	<<"shmid="<<shmid<<endl;
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
  else {
    cout<<"FUShmBuffer::shm_get(key="<<key<<",size="<<size<<") successful. "
	<<"shmid="<<shmid<<endl;
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
  else {
    cout<<"FUShmBuffer::shm_attach(shmid="<<shmid<<") successful. "
	<<"addr="<<hex<<result<<dec<<endl;
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
  else {
    cout<<"FUShmBuffer::sem_create(key="<<key<<",nsem="<<nsem<<") successful. "
	<<"semid="<<semid<<endl;
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
  else {
    cout<<"FUShmBuffer::sem_get(key="<<key<<",nsem="<<nsem<<") successful. "
	<<"semid="<<semid<<endl;
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
    cout<<"FUShmBuffer: FATAL ERROR in semaphore operation."<<endl;
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
    cout<<"FUShmBuffer: FATAL ERROR in semaphore operation."<<endl;
  }
}
