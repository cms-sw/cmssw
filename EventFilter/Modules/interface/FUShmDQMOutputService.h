#ifndef FUShmDQMOutputService_H
#define FUShmDQMOutputService_H

/**
 * This class is responsible for collecting data quality monitoring (DQM)
 * information, packaging it in DQMEvent objects, and writing out the data
 * to shared memory for the Resource Broker to send to the Storage Manager
 *
 * 27-Dec-2006 - KAB  - Initial Implementation
 * 31-Mar-2007 - HWKC - modification for shared memory usage
 *
 * $Id: FUShmDQMOutputService.h,v 1.13 2012/05/02 15:02:19 smorovic Exp $
 */

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "IOPool/Streamer/interface/DQMEventMessage.h"
#include "IOPool/Streamer/interface/DQMEventMsgBuilder.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "IOPool/Streamer/interface/StreamDQMSerializer.h"
#include "IOPool/Streamer/interface/StreamDQMDeserializer.h"
#include "EventFilter/ShmBuffer/interface/FUShmBuffer.h"
#include "EventFilter/Utilities/interface/ServiceWeb.h"

#include "xdata/UnsignedInteger32.h"

class FUShmDQMOutputService : public evf::ServiceWeb
{
 public:
  FUShmDQMOutputService(const edm::ParameterSet &pset,
                        edm::ActivityRegistry &actReg);
  ~FUShmDQMOutputService(void);

  //serviceweb interface
  void defaultWebPage(xgi::Input *in, xgi::Output *out); 
  void publish(xdata::InfoSpace *);

  void postEventProcessing(const edm::Event &event,
                           const edm::EventSetup &eventSetup);

  // test routines to check on timing of various signals
  void postEndJobProcessing();
  void postSourceConstructionProcessing(const edm::ModuleDescription &modDesc);
  void preBeginRun(const edm::RunID &runID, const edm::Timestamp &timestamp);
  void postEndLumi(edm::LuminosityBlock const&, edm::EventSetup const&);
  bool attachToShm();
  bool detachFromShm();
  void reset(){
    nbUpdates_ = 0;
    initializationIsNeeded_ = true;
  }
 protected:
  DQMStore *bei;

  void findMonitorElements(DQMEvent::TObjectTable &toTable,
                           std::string folderPath);

 private:
  void writeShmDQMData(DQMEventMsgBuilder const& dqmMsgBuilder);

  std::vector<char> messageBuffer_;
  int lumiSectionInterval_;  
  double lumiSectionsPerUpdate_;
  uint32 updateNumber_;
  unsigned int lumiSectionOfPreviousUpdate_;
  unsigned int firstLumiSectionSeen_;
  double timeInSecSinceUTC_;
  bool initializationIsNeeded_;
  bool useCompression_;
  int compressionLevel_;
  edm::StreamDQMSerializer serializeWorker_;
  edm::StreamDQMDeserializer deserializeWorker_;

  evf::FUShmBuffer* shmBuffer_;

  xdata::UnsignedInteger32 nbUpdates_;
  char host_name_[255];

  const std::string input;
  const std::string dqm;

  static bool fuIdsInitialized_;
  static uint32 fuGuidValue_;

  bool attach_;
 public:
  void setAttachToShm();
};

#endif
