#ifndef SiPixelMonitorRawData_SiPixelHLTSource_h
#define SiPixelMonitorRawData_SiPixelHLTSource_h
// -*- C++ -*-
//
// Package:     SiPixelMonitorRawData
// Class  :     SiPixelHLTSource
// 
/**

 Description: 
 Produces histograms for error information generated at the raw2digi stage for the 
 pixel tracker.

 Implementation:
 Takes raw data and error data as input, and uses it to populate three histograms 
 indexed by FED id.

*/
//
// Original Author:  Andrew York
//

#include <memory>

// user include files
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelRawDataError.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include <boost/cstdint.hpp>

 class SiPixelHLTSource : public edm::EDAnalyzer {
    public:
       explicit SiPixelHLTSource(const edm::ParameterSet& conf);
       ~SiPixelHLTSource();

       virtual void analyze(const edm::Event&, const edm::EventSetup&);
       virtual void beginJob() ;
       virtual void endJob() ;
       virtual void beginRun(const edm::Run&, edm::EventSetup const&) ;
       virtual void bookMEs();

    private:
       edm::ParameterSet conf_;
       edm::EDGetTokenT<FEDRawDataCollection> rawin_;
       edm::EDGetTokenT<edm::DetSetVector<SiPixelRawDataError> > errin_;
       edm::ESHandle<TrackerGeometry> pDD;
       bool saveFile;
       bool slowDown;
       std::string dirName_;
       int eventNo;
       DQMStore* theDMBE;
       MonitorElement* meRawWords_;
       MonitorElement* meNCRCs_;
       MonitorElement* meNErrors_;
       bool firstRun;
 };

#endif
