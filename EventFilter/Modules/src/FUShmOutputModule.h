#ifndef _FUShmOutputModule_h
#define _FUShmOutputModule_h 

/*
   Description:
     Header file shared memory to be used with FUShmOutputModule.
     See CMS EvF Storage Manager wiki page for further notes.

   $Id: FUShmOutputModule.h,v 1.8.2.2 2012/04/17 13:50:52 smorovic Exp $
*/

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "IOPool/Streamer/interface/InitMsgBuilder.h"
#include "IOPool/Streamer/interface/EventMsgBuilder.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "EventFilter/ShmBuffer/interface/FUShmBuffer.h"
#include "EventFilter/Utilities/interface/ShmOutputModuleRegistry.h"

#include <iostream>

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
  evf::FUShmBuffer* getBufferRef() {
    return shmBuffer_;
  }

  evf::FUShmBuffer* shmBuffer_;
};

namespace edm
{
  class ParameterSetDescription;
  class FUShmOutputModule : public evf::OutputModule
  {
  public:

    FUShmOutputModule(edm::ParameterSet const& ps);
    ~FUShmOutputModule();

    void doOutputHeader(InitMsgBuilder const& initMessage);
    void doOutputEvent(EventMsgBuilder const& eventMessage);
    unsigned int getCounts(){
      return count_;
    }
    void start();
    void stop();
    // No parameters.
    static void fillDescription(ParameterSetDescription&) {}
  private:

    evf::FUShmBuffer* shmBuffer_;
    std::string name_;
    unsigned int count_;

    static bool fuIdsInitialized_;
    static uint32 fuGuidValue_;
    bool postponeInitMsg_;
    bool sentInitMsg_;
    unsigned char * initBuf_;
    unsigned int initBufSize_;
    bool postponeStart_;
    unsigned int nExpectedEPs_;

  public:
    //void writeLuminosityBlock(LuminosityBlockPrincipal const&);
    void setPostponeInitMsg();
    void sendPostponedStart();
    void sendPostponedInitMsg();
    void setNExpectedEPs(unsigned int EPs);
    void unregisterFromShm();

  };
}

#endif
