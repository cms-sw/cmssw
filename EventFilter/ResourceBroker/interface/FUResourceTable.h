#ifndef FURESOURCETABLE_H
#define FURESOURCETABLE_H 1


#include "EventFilter/ResourceBroker/interface/FEDProvider.h"
#include "EventFilter/ResourceBroker/interface/FUResource.h"
#include "EventFilter/ResourceBroker/interface/FUShmBuffer.h"
#include "EventFilter/ResourceBroker/interface/BUProxy.h"
#include "EventFilter/ResourceBroker/interface/FUTypes.h"
#include "EventFilter/Utilities/interface/Exception.h"

#include "extern/log4cplus/linuxx86/include/log4cplus/logger.h"
#include "toolbox/include/toolbox/lang/Class.h"
#include "toolbox/include/toolbox/task/Action.h"
#include "toolbox/include/toolbox/task/WorkLoop.h"
#include "toolbox/include/BSem.h"

#include <vector>
#include <semaphore.h>


namespace evf {

  class FUResourceTable : public toolbox::lang::Class,
			  public FEDProvider
  {
  public:
    //
    // construction/destruction
    //
    FUResourceTable(UInt_t nbResources,UInt_t eventBufferSize,bool shmMode,
		    BUProxy* bu,log4cplus::Logger logger);
    virtual ~FUResourceTable();

    
    //
    // member functions
    //
    
    // FEDProvider interface, rqstEvent for shmMode=false only!
    FEDRawDataCollection* rqstEvent(UInt_t& evtNumber,UInt_t& buResourceId);
    void   sendDiscard(UInt_t buResourceId);
    
    // allocate new events from builder unit
    void   sendAllocate();
    
    // initialization of the resource queue
    void   initialize(UInt_t nbResources,UInt_t eventBufferSize);
    
    // emtpy all containers (resources & ids)
    void   clear();

    // reset the resource table to start over (in its current configuration)
    void   reset();

    // reset event & error counters
    void   resetCounters();

    // work loop to discard/allocate events in shmMode
    void   startWorkLoop() throw (evf::Exception);
    bool   workLoopAction(toolbox::task::WorkLoop* workLoop);
    
    // tell resources wether to check the crc
    void   setDoCrcCheck(UInt_t doCrcCheck)   { doCrcCheck_  =doCrcCheck; }

    // returns the fuResourceId of the allocated resource
    UInt_t allocateResource();
    
    // process buffer received via I2O_FU_TAKE message
    bool   buildResource(MemRef_t* bufRef);
    
    // return pointer to event FED data
    FEDRawDataCollection* requestResource(UInt_t& evtNumber,UInt_t& buResourceId);
    
    // check if this is the last message of the events without processing the msg
    bool   isLastMessageOfEvent(MemRef_t* bufRef);

    // various counters
    UInt_t nbResources() const { return resources_.size(); }
    UInt_t nbFreeSlots() const;
    UInt_t nbShmClients()const;
    UInt_t nbAllocated() const { return nbAllocated_; }
    UInt_t nbPending()   const { return nbPending_; }
    UInt_t nbCompleted() const { return nbCompleted_; }
    UInt_t nbProcessed() const { return nbProcessed_; }
    UInt_t nbDiscarded() const { return nbDiscarded_; }
    UInt_t nbLost()      const { return nbLost_; }
    
    UInt_t nbErrors()    const { return nbErrors_; }
    UInt_t nbCrcErrors() const { return nbCrcErrors_; }
    UInt_t nbAllocSent() const { return nbAllocSent_; }
    
    UInt_t nbBytes(bool reset=true);
    
  private:
    //
    // private member functions
    //
    void   lock();
    void   unlock();
    void   waitWriterSem();
    void   postWriterSem();
    void   waitReaderSem();
    void   postReaderSem();
    

  private:
    //
    // member data
    //
    typedef toolbox::task::WorkLoop        WorkLoop_t;
    typedef toolbox::task::ActionSignature ActionSignature_t;

    BUProxy*          bu_;
    log4cplus::Logger log_;
    
    WorkLoop_t*       workLoop_;
    ActionSignature_t*workLoopActionSignature_;

    bool              shmMode_;
    FUShmBuffer*      shmBuffer_;

    FUResourceVec_t   resources_;
    UIntQueue_t       freeResourceIds_; 
    UIntDeque_t       builtResourceIds_;

    UInt_t            doCrcCheck_;
    
    UInt_t            nbAllocated_;
    UInt_t            nbPending_;
    UInt_t            nbCompleted_;
    UInt_t            nbDiscarded_;
    UInt_t            nbProcessed_;
    UInt_t            nbLost_;

    UInt_t            nbErrors_;
    UInt_t            nbCrcErrors_;
    UInt_t            nbAllocSent_;
    
    UInt_t            nbBytes_;

    BSem              lock_;
    sem_t             writeSem_;
    sem_t             readSem_;
    
  };
  
} // namespace evf


//
// implementation of inline member functions
//
#include "EventFilter/ResourceBroker/interface/FUResourceTable.icc"


#endif
