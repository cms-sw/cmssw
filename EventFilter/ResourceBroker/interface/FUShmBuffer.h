#ifndef FUSHMBUFFER_H
#define FUSHMBUFFER_H 1


#include "EventFilter/ResourceBroker/interface/FUShmBufferCell.h"

#include <sys/ipc.h>
#include <sys/types.h>
#include <sys/shm.h>
#include <sys/sem.h>
#include <errno.h>


namespace evf {
  
  class FUShmBuffer
  {
    //
    // construction/destruction [-> static 'createShmBuffer'/'getShmBuffer']
    //
  private:
    FUShmBuffer(int shmid,int semid,
		unsigned int  nCell,
		unsigned int  cellBufferSize,
		unsigned int  nFed,
		unsigned int  nSuperFrag,
		bool          ownsMemory=true);
  public:
    ~FUShmBuffer();
    
    
    //
    // member functions
    //
    bool             ownsMemory()     const { return ownsMemory_; }
    int              shmid()          const { return shmid_; }
    int              semid()          const { return semid_; }
    unsigned int     nCell()          const { return nCell_; }
    unsigned int     cellBufferSize() const { return cellBufferSize_; }
    unsigned int     nFed()           const { return nFed_; }
    unsigned int     nSuperFrag()     const { return nSuperFrag_; }
    unsigned int     cellOffset()     const { return cellOffset_; }
    unsigned int     cellSize()       const { return cellSize_; }
    unsigned int     writeIndex()     const { return writeIndex_; }
    unsigned int     readIndex()      const { return readIndex_; }
    
    FUShmBufferCell* cell(unsigned int i);
    FUShmBufferCell* currentWriterCell();
    FUShmBufferCell* currentReaderCell();
    FUShmBufferCell* cellToBeDiscarded();
    
    void             scheduleForDiscard(FUShmBufferCell* cell);
    
    void             initialize();

    void             lock()             { sem_wait(2); }
    void             unlock()           { sem_post(2); }
    void             waitWriterSem()    { sem_wait(0); }
    void             postWriterSem()    { sem_post(0); }
    void             waitReaderSem()    { sem_wait(1); }
    void             postReaderSem()    { sem_post(1); }
    // semaphores to discard events
    void             waitDiscardedSem() { sem_wait(3); }
    void             postDiscardedSem() { sem_post(3); }
    void             waitDiscardSem()   { sem_wait(4); }
    void             postDiscardSem()   { sem_post(4); }

    int              writerSemValue() const;
    int              readerSemValue() const;
    
    void             sem_print();
    void             print(int verbose=0);

    
    
    //
    // static member functions
    //
    static unsigned int size(unsigned int nCell,
			     unsigned int cellBufferSize,
			     unsigned int nFed,
			     unsigned int nSuperFrag);
    static FUShmBuffer* createShmBuffer(unsigned int nCell,
					unsigned int cellBufferSize=4096000, //4MB
					unsigned int nFed=1024,
					unsigned int nSuperFrag=64);

    static FUShmBuffer* getShmBuffer();

    static bool         releaseSharedMemory();
    
    static key_t        getShmKey1();
    static key_t        getShmKey();
    static key_t        getSemKey();
    
    static int          shm_create(key_t key,int size);
    static int          shm_get(key_t key,int size);
    static void*        shm_attach(int shmid);
    static int          shm_nattch(int shmid);
    static int          shm_destroy(int shmid);
    
    static int          sem_create(key_t key,int nsem);
    static int          sem_get(key_t key,int nsem);
    static int          sem_destroy(int semid);
    
    
    //
    // private member functions
    //
    void  sem_init(int isem,int value);
    void  sem_wait(int isem);
    void  sem_post(int isem);
    
    
  private:
    //
    // member data
    //
    bool            ownsMemory_;
    int             shmid_;
    int             semid_;
    unsigned int    writeIndex_;
    unsigned int    readIndex_;
    unsigned int    cellIndexToBeDiscarded_;
    unsigned int    nCell_;
    unsigned int    cellBufferSize_;
    unsigned int    nFed_;
    unsigned int    nSuperFrag_;
    unsigned int    cellOffset_;
    unsigned int    cellSize_;
    
  };

  
} // namespace evf


#endif
