#ifndef _FUShmOutputModule_h
#define _FUShmOutputModule_h 

/*
   Description:
     Header file shared memory to be used with FUShmOutputModule.
     See CMS EvF Storage Manager wiki page for further notes.

   $Id: FUShmOutputModule.h,v 1.12 2012/10/11 17:48:11 smorovic Exp $
*/

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/EventSelector.h"

#include "IOPool/Streamer/interface/EventMessage.h"
#include "IOPool/Streamer/interface/InitMsgBuilder.h"
#include "IOPool/Streamer/interface/EventMsgBuilder.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "EventFilter/ShmBuffer/interface/FUShmBuffer.h"
#include "EventFilter/Modules/interface/ShmOutputModuleRegistry.h"

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
  //class ParameterSetDescription;
  class FUShmOutputModule : public evf::OutputModule
  {
  public:

    FUShmOutputModule(edm::ParameterSet const& ps);
    ~FUShmOutputModule();

    void insertStreamAndDatasetInfo(edm::ParameterSet & streams, edm::ParameterSet datasets/*std:std::string & moduleList*/);
    void doOutputHeader(InitMsgBuilder const& initMessage);
    void doOutputEvent(EventMsgBuilder const& eventMessage);
    unsigned int getCounts(){
      return count_;
    }
    void start();
    void stop();
    static void fillDescription(ParameterSetDescription&);

    void parseDatasets(InitMsgView const& initMessage);
    void countEventForDatasets(EventMsgView const& eventMessage);
    std::vector<std::string> getDatasetNames() {return selectedDatasetNames_;}
    std::vector<unsigned int>& getDatasetCounts() {return datasetCounts_;}
    void clearDatasetCounts() {
	    for (unsigned int i=0;i<datasetCounts_.size();i++) datasetCounts_[i]=0;
    }
    std::string getStreamId() {return streamId_;}

    //void writeLuminosityBlock(LuminosityBlockPrincipal const&);
    void setPostponeInitMsg();
    void sendPostponedStart();
    void sendPostponedInitMsg();
    void setNExpectedEPs(unsigned int EPs);
    void unregisterFromShm();

  private:

    evf::FUShmBuffer* shmBuffer_;
    std::string name_;
    unsigned int count_;

    static bool fuIdsInitialized_;
    static uint32 fuGuidValue_;
    unsigned int nExpectedEPs_;

    //dataset parsing
    std::vector<unsigned int> datasetCounts_;

    unsigned int numDatasets_;
    std::vector<std::string> selectedDatasetNames_;
    std::vector<Strings> datasetPaths_; 
    std::vector<std::pair<std::string,edm::EventSelector*>> dpEventSelectors_;
    unsigned int totalPaths_;

    std::string streamId_;
  };
}

#endif
