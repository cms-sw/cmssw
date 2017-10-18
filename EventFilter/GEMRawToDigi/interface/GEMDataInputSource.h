#ifndef EventFilter_GEMDataInputSource_h
#define EventFilter_GEMDataInputSource_h
// -*- C++ -*-
//
// Package:    SiPixelInputSources
// Class:      GEMDataInputSource
// 
/**\class GEMDataInputSource GEMDataInputSource.h IORawData/SiPixelInputSources/interface/GEMDataInputSource.h

   Description: <one line class summary>

   Implementation:
   <Notes on implementation>
*/
//
// Original Author:  Freya Blekman
//         Created:  Fri Sep  7 15:46:34 CEST 2007
//
//

#include <iostream>
#include <iomanip>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Sources/interface/ProducerSourceFromFiles.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "Utilities/StorageFactory/interface/StorageAccount.h"
#include "Utilities/StorageFactory/interface/Storage.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"

class FEDRawDataCollection;

class GEMDataInputSource : public edm::ProducerSourceFromFiles {

 public:

  explicit GEMDataInputSource(const edm::ParameterSet& pset, 
			      const edm::InputSourceDescription& desc);

  virtual ~GEMDataInputSource();

 private:

  virtual bool setRunAndEventInfo(edm::EventID& id, edm::TimeValue_t& time, edm::EventAuxiliary::ExperimentType&);
  virtual void produce(edm::Event& event);
  uint32_t synchronizeEvents();

  int m_fedid;
  uint32_t m_fileindex;
  std::unique_ptr<Storage> storage;
  int m_runnumber;
  uint64_t m_data;
  uint32_t m_currenteventnumber;
  uint32_t m_currenttriggernumber;
  uint32_t m_globaleventnumber;
  int32_t m_eventnumber_shift;
  int getEventNumberFromFillWords(const std::vector<uint64_t>& data, uint32_t &totword);
  std::unique_ptr<FEDRawDataCollection> buffers;
};
#endif
