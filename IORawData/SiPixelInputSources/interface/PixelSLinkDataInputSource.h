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
// $Id$
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
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "SealIOTools/StorageStreamBuf.h"
#include "SealBase/Storage.h"
#include "SealBase/DebugAids.h"
#include "SealBase/Signal.h"

class PixelSLinkDataInputSource : public edm::ExternalInputSource {

public:

  explicit PixelSLinkDataInputSource(const edm::ParameterSet& pset, 
				     const edm::InputSourceDescription& desc);

  virtual ~PixelSLinkDataInputSource();

  bool produce(edm::Event& event);


private:

  int m_fedid;
  uint32_t m_fileindex;
  std::auto_ptr<seal::Storage> storage;
};
#endif
