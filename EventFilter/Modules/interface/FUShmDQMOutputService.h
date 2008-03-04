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
 * $Id: FUShmDQMOutputService.h,v 1.3 2007/05/01 22:37:30 hcheung Exp $
 */

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "IOPool/Streamer/interface/DQMEventMessage.h"
#include "IOPool/Streamer/interface/DQMEventMsgBuilder.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "IOPool/Streamer/interface/StreamDQMSerializer.h"
#include "IOPool/Streamer/interface/StreamDQMDeserializer.h"
#include "EventFilter/ShmBuffer/interface/FUShmBuffer.h"

class FUShmDQMOutputService 
{
 public:
  FUShmDQMOutputService(const edm::ParameterSet &pset,
                   edm::ActivityRegistry &actReg);
  ~FUShmDQMOutputService(void);

  void postEventProcessing(const edm::Event &event,
                           const edm::EventSetup &eventSetup);

  // test routines to check on timing of various signals
  void postBeginJobProcessing();
  void postEndJobProcessing();
  void preSourceProcessing();
  void postSourceProcessing();
  void preModuleProcessing(const edm::ModuleDescription &modDesc);
  void postModuleProcessing(const edm::ModuleDescription &modDesc);
  void preSourceConstructionProcessing(const edm::ModuleDescription &modDesc);
  void postSourceConstructionProcessing(const edm::ModuleDescription &modDesc);
  void preModuleConstructionProcessing(const edm::ModuleDescription &modDesc);
  void postModuleConstructionProcessing(const edm::ModuleDescription &modDesc);

  bool attachToShm();
  bool detachFromShm();

 protected:
  DQMStore *bei;

  void findMonitorElements(DQMEvent::TObjectTable &toTable,
                           std::string folderPath);

 private:
  void writeShmDQMData(DQMEventMsgBuilder const& dqmMsgBuilder);

  std::vector<char> messageBuffer_;
  int lumiSectionInterval_;  
  double lumiSectionsPerUpdate_;
  //edm::LuminosityBlockID lumiSectionOfPreviousUpdate_;
  //edm::LuminosityBlockID firstLumiSectionSeen_;
  unsigned int lumiSectionOfPreviousUpdate_;
  unsigned int firstLumiSectionSeen_;
  double timeInSecSinceUTC_;
  bool initializationIsNeeded_;
  bool useCompression_;
  int compressionLevel_;
  edm::StreamDQMSerializer serializeWorker_;
  edm::StreamDQMDeserializer deserializeWorker_;

  evf::FUShmBuffer* shmBuffer_;

};

#endif
