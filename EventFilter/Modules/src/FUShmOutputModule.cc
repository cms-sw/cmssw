/*
   Description:
     EDM output module that will write data to shared memory for 
     the resource broker to send to the Storage Manager.
     See the CMS EvF Storage Manager wiki page for further notes.

   $Id: FUShmOutputModule.cc,v 1.12 2011/01/19 13:19:02 meschi Exp $
*/

#include "EventFilter/Utilities/interface/i2oEvfMsgs.h"
#include "EventFilter/Utilities/interface/ShmOutputModuleRegistry.h"
#include "IOPool/Streamer/interface/EventMessage.h"
#include "EventFilter/Modules/src/FUShmOutputModule.h"
#include "DataFormats/Provenance/interface/EventID.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/src/Guid.h"

#include "xdaq/Application.h"
#include "xdaq/ApplicationContext.h"
#include "xdaq/ApplicationGroup.h"
#include "zlib.h"

#include <string>
#include <fstream>
#include <iostream>

using namespace edm;
using namespace std;

static SM_SharedMemoryHandle sm_sharedmemory;

namespace edm
{

  /**
   * Initialize the static variables for the filter unit identifiers.
   */
  bool FUShmOutputModule::fuIdsInitialized_ = false;
  uint32 FUShmOutputModule::fuGuidValue_ = 0;

  FUShmOutputModule::FUShmOutputModule(edm::ParameterSet const& ps):
    shmBuffer_(0)
    , name_(ps.getParameter<std::string>( "@module_label" ))
    , count_(0)
  {
    FDEBUG(9) << "FUShmOutputModule: constructor" << endl;
    if(edm::Service<evf::ShmOutputModuleRegistry>())
      edm::Service<evf::ShmOutputModuleRegistry>()->registerModule(name_, this);  
    if (! fuIdsInitialized_) {
      fuIdsInitialized_ = true;

      edm::Guid guidObj(true);
      std::string guidString = guidObj.toString();

      uLong crc = crc32(0L, Z_NULL, 0);
      Bytef* buf = (Bytef*)guidString.data();
      crc = crc32(crc, buf, guidString.length());
      fuGuidValue_ = crc;
    }
  }
  
  FUShmOutputModule::~FUShmOutputModule()
  {
    FDEBUG(9) << "FUShmOutputModule: FUShmOutputModule destructor" << endl;
    sm_sharedmemory.detachShmBuffer();
    //shmdt(shmBuffer_);
  }

  void FUShmOutputModule::doOutputHeader(InitMsgBuilder const& initMessage)
  {
    count_ = 0;
    if(!shmBuffer_) shmBuffer_ = sm_sharedmemory.getShmBuffer();
    if(!shmBuffer_) edm::LogError("FUShmOutputModule") 
      << " Error getting shared memory buffer for INIT. " 
      << " Make sure you configure the ResourceBroker before the FUEventProcessor! "
      << " No INIT is sent - this is probably fatal!";
    if(shmBuffer_)
    {
      unsigned char* buffer = (unsigned char*) initMessage.startAddress();
      unsigned int size = initMessage.size();
      FDEBUG(10) << "writing out INIT message with size = " << size << std::endl;
      // no method in InitMsgBuilder to get the output module id, recast
      InitMsgView dummymsg(buffer);
      uint32 dmoduleId = dummymsg.outputModuleId();

      //bool ret = shmBuffer_->writeRecoInitMsg(dmoduleId, buffer, size);
      bool ret = sm_sharedmemory.getShmBuffer()->writeRecoInitMsg(dmoduleId, getpid(), fuGuidValue_, buffer, size);
      if(!ret) edm::LogError("FUShmOutputModule") << " Error writing preamble to ShmBuffer";
    }
  }

  void FUShmOutputModule::doOutputEvent(EventMsgBuilder const& eventMessage)
  {
    if(!shmBuffer_) edm::LogError("FUShmOutputModule") 
      << " Invalid shared memory buffer at first event"
      << " Make sure you configure the ResourceBroker before the FUEventProcessor! "
      << " No event is sent - this is fatal! Should throw here";
    else
    {
      count_++;
      unsigned char* buffer = (unsigned char*) eventMessage.startAddress();
      unsigned int size = eventMessage.size();
      EventMsgView eventView(eventMessage.startAddress());
      unsigned int runid = eventView.run();
      unsigned int eventid = eventView.event();
      unsigned int outModId = eventView.outModId();
      FDEBUG(10) << "FUShmOutputModule: event size = " << size << std::endl;
      //bool ret = shmBuffer_->writeRecoEventData(runid, eventid, outModId, buffer, size);
      bool ret = sm_sharedmemory.getShmBuffer()->writeRecoEventData(runid, eventid, outModId, getpid(), fuGuidValue_, buffer, size);
      if(!ret) edm::LogError("FUShmOutputModule") << " Error with writing data to ShmBuffer";
    }
  }

  void FUShmOutputModule::start()
  {
    //shmBuffer_ = evf::FUShmBuffer::getShmBuffer();
    shmBuffer_ = sm_sharedmemory.getShmBuffer();
    if(0==shmBuffer_) 
      edm::LogError("FUShmOutputModule")<<"Failed to attach to shared memory";
  }

  void FUShmOutputModule::stop()
  {
    FDEBUG(9) << "FUShmOutputModule: sending terminate run" << std::endl;
    if(0!=shmBuffer_){
      sm_sharedmemory.detachShmBuffer();
      //shmdt(shmBuffer_);
      shmBuffer_ = 0;
    }
  }

}
