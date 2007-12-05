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
// $Id: PixelSLinkDataInputSource.h,v 1.7 2007/11/17 23:23:02 wmtan Exp $
//
//

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Sources/interface/ExternalInputSource.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <iostream>
#include <iomanip>
#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "Utilities/StorageFactory/interface/StorageAccount.h"
#include "Utilities/StorageFactory/interface/Storage.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"

class PixelSLinkDataInputSource : public edm::ExternalInputSource {

public:

  explicit PixelSLinkDataInputSource(const edm::ParameterSet& pset, 
				     const edm::InputSourceDescription& desc);

  virtual ~PixelSLinkDataInputSource();

  bool produce(edm::Event& event);


private:

  int m_fedid;
  uint32_t m_fileindex;
  std::auto_ptr<Storage> storage;
  int m_runnumber;
  uint64_t m_data;
  uint32_t m_currenteventnumber;
};
#endif
