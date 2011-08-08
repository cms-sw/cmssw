#ifndef FURESOURCETABLE_H
#define FURESOURCETABLE_H 1


#include "EventFilter/ResourceBroker/interface/FUResource.h"
#include "EventFilter/ResourceBroker/interface/BUProxy.h"
#include "EventFilter/ResourceBroker/interface/SMProxy.h"
#include "EventFilter/ResourceBroker/interface/FUTypes.h"
#include "EventFilter/ShmBuffer/interface/FUShmBuffer.h"
#include "EventFilter/Utilities/interface/Exception.h"

#include "log4cplus/logger.h"
#include "toolbox/lang/Class.h"
#include "toolbox/task/Action.h"
#include "toolbox/task/WorkLoop.h"
#include "toolbox/BSem.h"

#include <sys/types.h>
#include <string>
#include <vector>
#include <queue>


namespace evf {

  class EvffedFillerRB;
  
  class FUResourceTable : public toolbox::lang::Class
  {
  public:
    //
    // construction/destruction
    //
    FUResourceTable(bool   segmentationMode,
		    UInt_t nbRawCells, UInt_t nbRecoCells, UInt_t nbDqmCells,
		    UInt_t rawCellSize,UInt_t recoCellSize,UInt_t dqmCellSize,
		    BUProxy* bu,SMProxy* sm,
		    log4cplus::Logger logger, 
		    unsigned int, 
		    EvffedFillerRB*frb,
		    xdaq::Application*) throw (evf::Exception);
    virtual ~FUResourceTable();
    
    
    //
    // member functions
    //
    
    // set fed filler
    //    void setFedFiller(){frb_ = frb;}

    // set the run number
    void   setRunNumber(UInt_t runNumber) { runNumber_ = runNumber; }

    // initialization of the resource queue
    void   initialize(bool   segmentationMode,
		      UInt_t nbRawCells, UInt_t nbRecoCells, UInt_t nbDqmCells,
		      UInt_t rawCellSize,UInt_t recoCellSize,UInt_t dqmCellSize)
      throw (evf::Exception);
    
    // work loop to send data events to storage manager
    void   startSendDataWorkLoop() throw (evf::Exception);
    bool   sendData(toolbox::task::WorkLoop* workLoop);
    
    // work loop to send dqm events to storage manager
    void   startSendDqmWorkLoop() throw (evf::Exception);
    bool   sendDqm(toolbox::task::WorkLoop* workLoop);
    
    // work loop to discard events to builder unit
    void   startDiscardWorkLoop() throw (evf::Exception);
    bool   discard(toolbox::task::WorkLoop* workLoop);
    
    // returns the fuResourceId of the allocated resource
    UInt_t allocateResource();
    
    // process buffer received via I2O_FU_TAKE message
    bool   buildResource(MemRef_t* bufRef);
    
    // process buffer received via I2O_SM_DATA_DISCARD message
    bool   discardDataEvent(MemRef_t* bufRef);
    
    // process buffer received via I2O_SM_DQM_DISCARD message
    bool   discardDqmEvent(MemRef_t* bufRef);
    
    // post end-of-ls event to shmem
    void   postEndOfLumiSection(MemRef_t* bufRef);

    // drop next available event
    void   dropEvent();

    // send event belonging to crashed process to error stream (return false
    // if no event is found)    
    bool   handleCrashedEP(UInt_t runNumber,pid_t pid);

    // dump event to ascii file
    void   dumpEvent(evf::FUShmRawCell* cell);
    
    // send empty events to notify clients to shutdown
    void   stop();
    void   halt();
    void   shutDownClients();
    
    // emtpy all containers (resources & ids)
    void   clear();

    // reset event & error counters
    void   resetCounters();

    // tell resources wether to check the crc
    void   setDoCrcCheck(UInt_t doCrcCheck) { doCrcCheck_=doCrcCheck; }

    // tell resources wether to dump events to an ascii file
    void   setDoDumpEvents(UInt_t doDumpEvents) { doDumpEvents_=doDumpEvents; }

    // check if resource table is active (enabled)
    bool   isActive() const { return isActive_; }

    // check if resource table can be savely destroyed
    bool   isReadyToShutDown() const { return isReadyToShutDown_; }

    // various counters
    UInt_t   nbResources()        const { return resources_.size(); }
    UInt_t   nbFreeSlots()        const { return freeResourceIds_.size(); }
    UInt_t   nbAllocated()        const { return nbAllocated_; }
    UInt_t   nbPending()          const { return nbPending_; }
    UInt_t   nbCompleted()        const { return nbCompleted_; }
    UInt_t   nbSent()             const { return nbSent_; }
    UInt_t   nbSentError()        const { return nbSentError_; }
    UInt_t   nbSentDqm()          const { return nbSentDqm_; }
    UInt_t   nbPendingSMDiscards()const { return nbPendingSMDiscards_; }
    UInt_t   nbPendingSMDqmDiscards()const { return nbPendingSMDqmDiscards_; }
    UInt_t   nbDiscarded()        const { return nbDiscarded_; }
    UInt_t   nbLost()             const { return nbLost_; }
    
    UInt_t   nbErrors()           const { return nbErrors_; }
    UInt_t   nbCrcErrors()        const { return nbCrcErrors_; }
    UInt_t   nbAllocSent()        const { return nbAllocSent_; }
    
    uint64_t sumOfSquares()       const { return sumOfSquares_; }
    UInt_t   sumOfSizes()         const { return sumOfSizes_; }
    
    
    // information about (raw) shared memory cells
    UInt_t                   nbClients()                           const;
    std::vector<pid_t>       clientPrcIds()                        const;
    std::string              clientPrcIdsAsString()                const;
    std::vector<std::string> cellStates()                          const;
    std::vector<std::string> dqmCellStates()                       const;
    std::vector<UInt_t>      cellEvtNumbers()                      const;
    std::vector<pid_t>       cellPrcIds()                          const;
    std::vector<time_t>      cellTimeStamps()                      const;
    

    
    //
    // helpers
    //
    void   sendAllocate();
    void   sendDiscard(UInt_t buResourceId);
    
    void   sendInitMessage(UInt_t  fuResourceId,
			   UInt_t  outModId,
			   UInt_t  fuProcessId,
			   UInt_t  fuGuid,
			   UChar_t*data,
			   UInt_t  dataSize);

    void   sendDataEvent(UInt_t  fuResourceId,
			 UInt_t  runNumber,
			 UInt_t  evtNumber,
			 UInt_t  outModId,
			 UInt_t  fuProcessId,
			 UInt_t  fuGuid,
			 UChar_t*data,
			 UInt_t  dataSize);

    void   sendErrorEvent(UInt_t  fuResourceId,
			  UInt_t  runNumber,
			  UInt_t  evtNumber,
			  UInt_t  fuProcessId,
			  UInt_t  fuGuid,
			  UChar_t*data,
			  UInt_t  dataSize);
    
    void   sendDqmEvent(UInt_t  fuDqmId,
			UInt_t  runNumber,
			UInt_t  evtAtUpdate,
			UInt_t  folderId,
			UInt_t  fuProcessId,
			UInt_t  fuGuid,
			UChar_t*data,
			UInt_t  dataSize);
    
    bool   isLastMessageOfEvent(MemRef_t* bufRef);
    
    void   injectCRCError();
    
    void   lock()      { lock_.take(); }
    void   unlock()    { lock_.give(); }
    //void   lockShm()   { shmBuffer_->lock(); }
    //void   unlockShm() { shmBuffer_->unlock(); }
    void   printWorkLoopStatus();

    void   lastResort();

  private:
    //
    // member data
    //
    typedef toolbox::task::WorkLoop        WorkLoop_t;
    typedef toolbox::task::ActionSignature ActionSignature_t;

    BUProxy           *bu_;
    SMProxy           *sm_;

    log4cplus::Logger  log_;
    
    WorkLoop_t        *wlSendData_;
    ActionSignature_t *asSendData_;

    WorkLoop_t        *wlSendDqm_;
    ActionSignature_t *asSendDqm_;

    WorkLoop_t        *wlDiscard_;
    ActionSignature_t *asDiscard_;

    FUShmBuffer       *shmBuffer_;
    FUResourceVec_t    resources_;
    UInt_t             nbDqmCells_;
    UInt_t             nbRawCells_;
    UInt_t             nbRecoCells_;
    std::queue<UInt_t> freeResourceIds_;
    
    bool              *acceptSMDataDiscard_;
    int               *acceptSMDqmDiscard_;
    
    UInt_t             doCrcCheck_;
    UInt_t             doDumpEvents_;
    unsigned int       shutdownTimeout_;

    UInt_t             nbAllocated_;
    UInt_t             nbPending_;
    UInt_t             nbCompleted_;
    UInt_t             nbSent_;
    UInt_t             nbSentError_;
    UInt_t             nbSentDqm_;
    UInt_t             nbPendingSMDiscards_;
    UInt_t             nbPendingSMDqmDiscards_;
    UInt_t             nbDiscarded_;
    UInt_t             nbLost_;
    
    UInt_t             nbClientsToShutDown_;
    bool               isReadyToShutDown_;
    bool               isActive_;
    bool               isHalting_;
    bool               isStopping_;
    
    UInt_t             nbErrors_;
    UInt_t             nbCrcErrors_;
    UInt_t             nbAllocSent_;
    
    uint64_t           sumOfSquares_;
    UInt_t             sumOfSizes_;

    UInt_t             runNumber_;
    
    toolbox::BSem      lock_;
    EvffedFillerRB    *frb_;    
    xdaq::Application *app_;
  };
  
} // namespace evf


#endif
