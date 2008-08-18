#ifndef _FUShmOutputModule_h
#define _FUShmOutputModule_h 

/*
   Description:
     Header file shared memory to be used with FUShmOutputModule.
     See CMS EvF Storage Manager wiki page for further notes.

   $Id: FUShmOutputModule.h,v 1.2 2007/05/01 22:37:31 hcheung Exp $
*/

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "IOPool/Streamer/interface/InitMsgBuilder.h"
#include "IOPool/Streamer/interface/EventMsgBuilder.h"

#include "EventFilter/ShmBuffer/interface/FUShmBuffer.h"

// Data structure to be shared by all output modules for event serialization
struct SM_SharedMemoryHandle
{
  SM_SharedMemoryHandle():
    shmBuffer_(0)
  { }

  evf::FUShmBuffer* getShmBuffer() {
   if(!shmBuffer_) {
     shmBuffer_ = evf::FUShmBuffer::getShmBuffer();
     return shmBuffer_;
   } else {
     return shmBuffer_;
   }
  }
  void detachShmBuffer() {
   if(!shmBuffer_) {
     // no shared memory was attached to!
   } else {
     shmdt(shmBuffer_);
     shmBuffer_ = 0;
   }
  }

  evf::FUShmBuffer* shmBuffer_;
};

namespace edm
{
  class FUShmOutputModule
  {
  public:

    FUShmOutputModule(edm::ParameterSet const& ps);
    ~FUShmOutputModule();

    void doOutputHeader(InitMsgBuilder const& initMessage);
    void doOutputEvent(EventMsgBuilder const& eventMessage);
    void start();
    void stop();

  private:

    evf::FUShmBuffer* shmBuffer_;

  };
}

#endif
