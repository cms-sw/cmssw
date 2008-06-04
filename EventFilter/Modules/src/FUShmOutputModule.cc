/*
   Description:
     EDM output module that will write data to shared memory for 
     the resource broker to send to the Storage Manager.
     See the CMS EvF Storage Manager wiki page for further notes.

   $Id: FUShmOutputModule.cc,v 1.6 2008/01/29 15:25:26 biery Exp $
*/

#include "EventFilter/Utilities/interface/i2oEvfMsgs.h"
#include "IOPool/Streamer/interface/EventMessage.h"
#include "EventFilter/Modules/src/FUShmOutputModule.h"
#include "DataFormats/Provenance/interface/EventID.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "xdaq/Application.h"
#include "xdaq/ApplicationContext.h"
#include "xdaq/ApplicationGroup.h"

#include <string>
#include <fstream>
#include <iostream>

using namespace edm;
using namespace std;

namespace edm
{

  FUShmOutputModule::FUShmOutputModule(edm::ParameterSet const& ps):
    shmBuffer_(0)
  {
    FDEBUG(9) << "FUShmOutputModule: constructor" << endl;
  }
  
  FUShmOutputModule::~FUShmOutputModule()
  {
    FDEBUG(9) << "FUShmOutputModule: FUShmOutputModule destructor" << endl;
    shmdt(shmBuffer_);
  }

  void FUShmOutputModule::doOutputHeader(InitMsgBuilder const& initMessage)
  {
    if(!shmBuffer_) shmBuffer_ = evf::FUShmBuffer::getShmBuffer();
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

      bool ret = shmBuffer_->writeRecoInitMsg(dmoduleId, buffer, size);
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
      unsigned char* buffer = (unsigned char*) eventMessage.startAddress();
      unsigned int size = eventMessage.size();
      EventMsgView eventView(eventMessage.startAddress());
      unsigned int runid = eventView.run();
      unsigned int eventid = eventView.event();
      unsigned int outModId = eventView.outModId();
      FDEBUG(10) << "FUShmOutputModule: event size = " << size << std::endl;
      bool ret = shmBuffer_->writeRecoEventData(runid, eventid, outModId, buffer, size);
      if(!ret) edm::LogError("FUShmOutputModule") << " Error with writing data to ShmBuffer";
    }
  }

  void FUShmOutputModule::start()
  {
    shmBuffer_ = evf::FUShmBuffer::getShmBuffer();
    if(0==shmBuffer_) 
      edm::LogError("FUShmOutputModule")<<"Failed to attach to shared memory";
  }

  void FUShmOutputModule::stop()
  {
    FDEBUG(9) << "FUShmOutputModule: sending terminate run" << std::endl;
    if(0!=shmBuffer_){
      shmdt(shmBuffer_);
      shmBuffer_ = 0;
    }
  }

}
