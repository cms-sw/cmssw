#ifndef FUSHMBUFFER_H
#define FUSHMBUFFER_H 1


#include "EventFilter/ShmBuffer/interface/FUShmRawCell.h"
#include "EventFilter/ShmBuffer/interface/FUShmRecoCell.h"
#include "EventFilter/ShmBuffer/interface/FUShmDqmCell.h"


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
    FUShmBuffer(bool         segmentationMode,
		unsigned int nRawCells,
		unsigned int nRecoCells,
		unsigned int nDqmCells,
		unsigned int rawCellSize,
		unsigned int recoCellSize,
		unsigned int dqmCellSize);
  public:
    ~FUShmBuffer();
    
    
  public:
    //
    // public member functions
    //
    void           initialize(unsigned int shmid,unsigned int semid);
    
    int            nbRawCellsToWrite()  const;
    int            nbRawCellsToRead()   const;
    FUShmRawCell*  rawCellToWrite();
    FUShmRawCell*  rawCellToRead();
    FUShmRawCell*  rawCellToDiscard();
    void           finishWritingRawCell(FUShmRawCell* cell);
    void           finishReadingRawCell(FUShmRawCell* cell);
    void           scheduleRawCellForDiscard(unsigned int iCell);
    void           discardRawCell(FUShmRawCell* cell);
    
    int            nbRecoCellsToWrite() const;
    int            nbRecoCellsToRead()  const;
    FUShmRecoCell* recoCellToWrite();
    FUShmRecoCell* recoCellToRead();
    void           finishWritingRecoCell(FUShmRecoCell* cell);
    void           finishReadingRecoCell(FUShmRecoCell* cell);
    
    int            nbDqmCellsToWrite() const;
    int            nbDqmCellsToRead()  const;
    FUShmDqmCell*  dqmCellToWrite();
    FUShmDqmCell*  dqmCellToRead();
    void           finishWritingDqmCell(FUShmDqmCell* cell);
    void           finishReadingDqmCell(FUShmDqmCell* cell);
    
    void           sem_print();
    
    
    //
    // static member functions
    //
    static FUShmBuffer* createShmBuffer(bool         semgmentationMode,
					unsigned int nRawCells,
					unsigned int nRecoCells,
					unsigned int nDqmCells,
					unsigned int rawCellSize =4096000,  //4MB
					unsigned int recoCellSize=4096000,  //4MB
					unsigned int dqmCellSize =4096000); //4MB
    static FUShmBuffer* getShmBuffer();
    static bool         releaseSharedMemory();
    
    static unsigned int size(bool         segmentationMode,
			     unsigned int nRawCells,
			     unsigned int nRecoCells,
			     unsigned int nDqmCells,
			     unsigned int rawCellSize,
			     unsigned int recoCellSize,
			     unsigned int dqmCellSize);
    
    static key_t getShmDescriptorKey();
    static key_t getShmKey();
    static key_t getSemKey();
    
    static int   shm_create(key_t key,int size);
    static int   shm_get(key_t key,int size);
    static void* shm_attach(int shmid);
    static int   shm_nattch(int shmid);
    static int   shm_destroy(int shmid);
    
    static int   sem_create(key_t key,int nsem);
    static int   sem_get(key_t key,int nsem);
    static int   sem_destroy(int semid);
    
    
  private:
    //
    // private member functions
    //
    int            shmid()       const { return shmid_; }
    int            semid()       const { return semid_; }
    
    FUShmRawCell*  rawCell(unsigned int iCell);
    FUShmRecoCell* recoCell(unsigned int iCell);
    FUShmDqmCell*  dqmCell(unsigned int iCell);
    
    key_t          shmKey(unsigned int iCell,unsigned int offset);
    key_t          rawCellShmKey(unsigned int iCell);
    key_t          recoCellShmKey(unsigned int iCell);
    key_t          dqmCellShmKey(unsigned int iCell);

    void           sem_init(int isem,int value);
    void           sem_wait(int isem);
    void           sem_post(int isem);

    void           lock()             { sem_wait(0); }
    void           unlock()           { sem_post(0); }
    void           waitRawWrite()     { sem_wait(1); }
    void           postRawWrite()     { sem_post(1); }
    void           waitRawRead()      { sem_wait(2); }
    void           postRawRead()      { sem_post(2); }
    void           waitRawDiscard()   { sem_wait(3); }
    void           postRawDiscard()   { sem_post(3); }
    void           waitRawDiscarded() { sem_wait(4); }
    void           postRawDiscarded() { sem_post(4); }
    void           waitRecoWrite()    { sem_wait(5); }
    void           postRecoWrite()    { sem_post(5); }
    void           waitRecoRead()     { sem_post(6); }
    void           postRecoRead()     { sem_post(6); }
    void           waitDqmWrite()     { sem_wait(7); }
    void           postDqmWrite()     { sem_post(7); }
    void           waitDqmRead()      { sem_post(8); }
    void           postDqmRead()      { sem_post(8); }

    
  private:
    //
    // member data
    //
    bool         segmentationMode_;
    int          shmid_;
    int          semid_;

    unsigned int rawWriteIndex_;
    unsigned int rawReadIndex_;
    unsigned int rawDiscardIndex_;
    unsigned int nRawCells_;
    unsigned int rawCellPayloadSize_;
    unsigned int rawCellTotalSize_;
    unsigned int rawCellOffset_;
    
    unsigned int recoWriteIndex_;
    unsigned int recoReadIndex_;
    unsigned int nRecoCells_;
    unsigned int recoCellPayloadSize_;
    unsigned int recoCellTotalSize_;
    unsigned int recoCellOffset_;
    
    unsigned int dqmWriteIndex_;
    unsigned int dqmReadIndex_;
    unsigned int nDqmCells_;
    unsigned int dqmCellPayloadSize_;
    unsigned int dqmCellTotalSize_;
    unsigned int dqmCellOffset_;
    
  };

  
} // namespace evf


#endif
