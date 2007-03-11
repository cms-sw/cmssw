#ifndef FURESOURCETABLE_H
#define FURESOURCETABLE_H 1


#include "EventFilter/ResourceBroker/interface/FUResource.h"
#include "EventFilter/ResourceBroker/interface/BUProxy.h"
#include "EventFilter/ResourceBroker/interface/FUTypes.h"
#include "EventFilter/ShmBuffer/interface/FUShmBuffer.h"
#include "EventFilter/Utilities/interface/Exception.h"

#include "extern/log4cplus/linuxx86/include/log4cplus/logger.h"
#include "toolbox/include/toolbox/lang/Class.h"
#include "toolbox/include/toolbox/task/Action.h"
#include "toolbox/include/toolbox/task/WorkLoop.h"
#include "toolbox/include/BSem.h"

#include <vector>


namespace evf {
  
  class FUResourceTable : public toolbox::lang::Class
  {
  public:
    //
    // construction/destruction
    //
    FUResourceTable(UInt_t nbResources,UInt_t eventBufferSize,
		    BUProxy* bu,log4cplus::Logger logger);
    virtual ~FUResourceTable();

    
    //
    // member functions
    //
    
    // allocate new events from builder unit
    void   sendAllocate();
    
    // discard event to builder unit
    void   sendDiscard(UInt_t buResourceId);
    
    // drop next available event
    void   dropEvent();
    
    // send empty events to notify clients to shutdown
    void   shutDownClients();
    
    // initialization of the resource queue
    void   initialize(UInt_t nbResources,UInt_t eventBufferSize);
    
    // emtpy all containers (resources & ids)
    void   clear();

    // reset the resource table to start over (in its current configuration)
    void   reset();

    // reset event & error counters
    void   resetCounters();

    // work loop to discard/allocate events
    void   startDiscardWorkLoop() throw (evf::Exception);
    bool   discard(toolbox::task::WorkLoop* workLoop);
    
    // tell resources wether to check the crc
    void   setDoCrcCheck(UInt_t doCrcCheck) { doCrcCheck_=doCrcCheck; }

    // returns the fuResourceId of the allocated resource
    UInt_t allocateResource();
    
    // process buffer received via I2O_FU_TAKE message
    bool   buildResource(MemRef_t* bufRef);
    
    // check if this is the last message of the events without processing the msg
    bool   isLastMessageOfEvent(MemRef_t* bufRef);
    
    // check if resource table can be savely destroyed
    bool   isReadyToShutDown() const { return isReadyToShutDown_; }
    
    
    // various counters
    UInt_t nbResources()  const { return resources_.size(); }
    UInt_t nbFreeSlots()  const { return shmBuffer_->writerSemValue(); }
    UInt_t nbShmClients() const;
    UInt_t nbAllocated()  const { return nbAllocated_; }
    UInt_t nbPending()    const { return nbPending_; }
    UInt_t nbCompleted()  const { return nbCompleted_; }
    UInt_t nbProcessed()  const { return nbProcessed_; }
    UInt_t nbDiscarded()  const { return nbDiscarded_; }
    UInt_t nbLost()       const { return nbLost_; }
    
    UInt_t nbErrors()     const { return nbErrors_; }
    UInt_t nbCrcErrors()  const { return nbCrcErrors_; }
    UInt_t nbAllocSent()  const { return nbAllocSent_; }
    
    UInt_t nbBytes(bool reset=true);
    
  private:
    //
    // private member functions
    //
    void   lock()          { lock_.take(); }
    void   unlock()        { lock_.give(); }
    void   waitWriterSem() { shmBuffer_->waitWriterSem(); }
    void   postWriterSem() { shmBuffer_->postWriterSem(); }
    void   waitReaderSem() { shmBuffer_->waitReaderSem(); }
    void   postReaderSem() { shmBuffer_->postReaderSem(); }
    

  private:
    //
    // member data
    //
    typedef toolbox::task::WorkLoop        WorkLoop_t;
    typedef toolbox::task::ActionSignature ActionSignature_t;

    BUProxy*          bu_;
    log4cplus::Logger log_;
    
    WorkLoop_t*       workLoopDiscard_;
    ActionSignature_t*asDiscard_;

    FUShmBuffer*      shmBuffer_;
    FUResourceVec_t   resources_;
    
    UInt_t            doCrcCheck_;

    UInt_t            nbAllocated_;
    UInt_t            nbPending_;
    UInt_t            nbCompleted_;
    UInt_t            nbDiscarded_;
    UInt_t            nbProcessed_;
    UInt_t            nbLost_;
    
    UInt_t            nbClientsToShutDown_;
    bool              isReadyToShutDown_;
    
    UInt_t            nbErrors_;
    UInt_t            nbCrcErrors_;
    UInt_t            nbAllocSent_;
    
    UInt_t            nbBytes_;

    BSem              lock_;
    
  };
  
} // namespace evf


//
// implementation of inline member functions
//

//______________________________________________________________________________
inline
evf::UInt_t evf::FUResourceTable::nbBytes(bool reset)
{
  lock();
  UInt_t result=nbBytes_;
  if (reset) nbBytes_=0;
  unlock();
  return result;
}


#endif
