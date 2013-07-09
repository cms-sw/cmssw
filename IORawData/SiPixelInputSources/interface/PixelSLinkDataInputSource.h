#ifndef IORawDataSiPixelInputSources_PixelSLinkDataInputSource_h
#define IORawDataSiPixelInputSources_PixelSLinkDataInputSource_h
// -*- C++ -*-
//
// Package:    SiPixelInputSources
// Class:      PixelSLinkDataInputSource
// 
/**\class PixelSLinkDataInputSource PixelSLinkDataInputSource.h IORawData/SiPixelInputSources/interface/PixelSLinkDataInputSource.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Freya Blekman
//         Created:  Fri Sep  7 15:46:34 CEST 2007
// $Id: PixelSLinkDataInputSource.h,v 1.9 2008/02/25 21:12:09 fblekman Exp $
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

class PixelSLinkDataInputSource : public edm::ProducerSourceFromFiles {

public:

  explicit PixelSLinkDataInputSource(const edm::ParameterSet& pset, 
				     const edm::InputSourceDescription& desc);

  virtual ~PixelSLinkDataInputSource();

private:

  virtual bool setRunAndEventInfo(edm::EventID& id, edm::TimeValue_t& time);
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
  int getEventNumberFromFillWords(std::vector<uint64_t> data, uint32_t &totword);
  std::auto_ptr<FEDRawDataCollection> buffers;
};
#endif
