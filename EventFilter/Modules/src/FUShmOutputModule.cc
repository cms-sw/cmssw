/*
   Description:
     EDM output module that will write data to shared memory for 
     the resource broker to send to the Storage Manager.
     See the CMS EvF Storage Manager wiki page for further notes.

   $Id: FUShmOutputModule.cc,v 1.16 2012/09/02 15:04:25 smorovic Exp $
*/

#include "EventFilter/Utilities/interface/i2oEvfMsgs.h"

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
    , postponeInitMsg_(false)
    , sentInitMsg_(false)
    , initBuf_(nullptr)
    , initBufSize_(0)
    , postponeStart_(false)
    , nExpectedEPs_(0)
    , numDatasets_(0)
    , streamId_()
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

  void FUShmOutputModule::insertStreamAndDatasetInfo(edm::ParameterSet & streams, edm::ParameterSet datasets/*std:std::string & moduleList*/)
  {
    try {
      //compose dataset name string
      if (name_.size() > std::string("hltOutput").size() && name_.find("hltOutput")!=std::string::npos)
	streamId_=name_.substr(name_.find("hltOutput")+std::string("hltOutput").size());
      else return;

      //make local copy of dataset definitions

      if (streamId_.size()) {
	Strings streamDatasetList = streams.getParameter<Strings>(streamId_);
	for (size_t i=0;i<streamDatasetList.size();i++) {
	  selectedDatasetNames_.push_back(streamDatasetList[i]);
	  Strings thisDatasetPaths = datasets.getParameter<Strings>(streamDatasetList[i]); 
	  datasetPaths_.push_back(thisDatasetPaths);
	  numDatasets_++;
	}
      }
    }
    catch (...) {
      //not present:ignore
      selectedDatasetNames_.clear();
      datasetPaths_.clear();
      numDatasets_=0;
      streamId_=std::string();
    }
  }

  void FUShmOutputModule::fillDescription(ParameterSetDescription& description)
  {
  }


  void FUShmOutputModule::doOutputHeader(InitMsgBuilder const& initMessage)
  {
    unsigned char* buffer = (unsigned char*) initMessage.startAddress();
    unsigned int size = initMessage.size();
    InitMsgView dummymsg(buffer);
    parseDatasets(dummymsg);
    //saving message for later if postpone is on
    if (postponeInitMsg_) {
      sentInitMsg_=false;
      if (initBuf_) delete initBuf_;//clean up if there are leftovers from last run
      //copy message for later sending
      initBufSize_ = initMessage.size();
      initBuf_ = new unsigned char[initBufSize_];
      memcpy(initBuf_, (unsigned char*) initMessage.startAddress(),sizeof(unsigned char)*initBufSize_);
      return;
    }

    sentInitMsg_=true;
    count_ = 0;
    if(!shmBuffer_) shmBuffer_ = sm_sharedmemory.getShmBuffer();
    if(!shmBuffer_) edm::LogError("FUShmOutputModule") 
      << " Error getting shared memory buffer for INIT. " 
      << " Make sure you configure the ResourceBroker before the FUEventProcessor! "
      << " No INIT is sent - this is probably fatal!";
    if(shmBuffer_)
    {
      FDEBUG(10) << "writing out INIT message with size = " << size << std::endl;
      // no method in InitMsgBuilder to get the output module id, recast
      uint32 dmoduleId = dummymsg.outputModuleId();

      //bool ret = shmBuffer_->writeRecoInitMsg(dmoduleId, buffer, size);
      bool ret = sm_sharedmemory.getShmBuffer()->writeRecoInitMsg(dmoduleId, getpid(), fuGuidValue_, buffer, size,nExpectedEPs_);
      if(!ret) edm::LogError("FUShmOutputModule") << " Error writing preamble to ShmBuffer";
    }
  }

  void FUShmOutputModule::setPostponeInitMsg()
  {
    //postpone start and Init message for after beginRun
    postponeInitMsg_=true;
    postponeStart_=true;
    //reset this on each run
    if (initBuf_) delete initBuf_;
    initBufSize_=0;
    initBuf_=nullptr;
    sentInitMsg_=false;
  }

  void FUShmOutputModule::sendPostponedInitMsg() 
  {
    if (postponeStart_) {
      postponeStart_=false;
      start();
    }
    if (!sentInitMsg_ && postponeInitMsg_) {
      if(!shmBuffer_) shmBuffer_ = sm_sharedmemory.getShmBuffer();
      if(!shmBuffer_) edm::LogError("FUShmOutputModule")
	<< " Error getting shared memory buffer for INIT. "
	<< " Make sure you configure the ResourceBroker before the FUEventProcessor! "
	<< " No INIT is sent - this is probably fatal!";
      if(shmBuffer_)
      {
	FDEBUG(10) << "writing out (postponed) INIT message with size = " << initBufSize_ << std::endl;
	InitMsgView dummymsg(initBuf_);
	uint32 dmoduleId = dummymsg.outputModuleId();
	bool ret = sm_sharedmemory.getShmBuffer()->writeRecoInitMsg(dmoduleId, getpid(), fuGuidValue_, initBuf_, initBufSize_,nExpectedEPs_);
	if(!ret) edm::LogError("FUShmOutputModule") << " Error writing preamble to ShmBuffer";
      }
      sentInitMsg_=true;
      if (initBuf_) delete initBuf_;
      initBufSize_=0;
      initBuf_=nullptr;
    }
  }


  void FUShmOutputModule::doOutputEvent(EventMsgBuilder const& eventMessage)
  {
    if (!sentInitMsg_ && postponeInitMsg_) sendPostponedInitMsg();
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
      countEventForDatasets(eventView);
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
    if (postponeStart_) return;
    //shmBuffer_ = evf::FUShmBuffer::getShmBuffer();
    shmBuffer_ = sm_sharedmemory.getShmBuffer();
    if(0==shmBuffer_) 
      edm::LogError("FUShmOutputModule")<<"Failed to attach to shared memory";
  }

  void FUShmOutputModule::sendPostponedStart() {
      postponeStart_=false;
      start();
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

  void FUShmOutputModule::setNExpectedEPs(unsigned int EPs) {
    nExpectedEPs_ = EPs;
  }

  void FUShmOutputModule::unregisterFromShm() {
    shmBuffer_=sm_sharedmemory.getBufferRef();
    if (0!=shmBuffer_) {
      shmBuffer_->removeClientPrcId(getpid());
    }
  }

  void FUShmOutputModule::parseDatasets(InitMsgView const& initMessage)
  {
     //reset counter
     for (size_t i=0;i<datasetCounts_.size();i++) datasetCounts_[i]=0;
     if (!numDatasets_) return;
     if (dpEventSelectors_.size()) return;
     Strings allPaths;
     initMessage.hltTriggerNames(allPaths);
     totalPaths_ = allPaths.size();
     for (size_t i=0;i<numDatasets_;i++)
     {
       dpEventSelectors_.push_back(std::pair<std::string,edm::EventSelector*>(selectedDatasetNames_[i],new edm::EventSelector(datasetPaths_[i],allPaths))); 
       datasetCounts_.push_back(0);
     }
  }

  void FUShmOutputModule::countEventForDatasets(EventMsgView const& eventMessage)
  {
    if (!numDatasets_) return;
    uint8 hlt_out[totalPaths_];
    eventMessage.hltTriggerBits( hlt_out );
    for (size_t i=0;i<numDatasets_;i++) {
      if ( dpEventSelectors_[i].second->acceptEvent( hlt_out, totalPaths_)) {
	datasetCounts_[i]++;
      }
    }
  }
}
