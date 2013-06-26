#ifndef SiPixelMonitorCluster_SiPixelClusterSource_h
#define SiPixelMonitorCluster_SiPixelClusterSource_h
// -*- C++ -*-
//
// Package:     SiPixelMonitorCluster
// Class  :     SiPixelClusterSource
// 
/*

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Vincenzo Chiochia & Andrew York
//
// Updated by: Lukas Wehrli
// for pixel offline DQM 
//         Created:  
// $Id: SiPixelClusterSource.h,v 1.20 2011/05/20 17:17:24 wmtan Exp $

#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/SiPixelMonitorCluster/interface/SiPixelClusterModule.h"

#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <boost/cstdint.hpp>

#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h" 
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h" 
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"

 class SiPixelClusterSource : public edm::EDAnalyzer {
    public:
       explicit SiPixelClusterSource(const edm::ParameterSet& conf);
       ~SiPixelClusterSource();

       typedef edmNew::DetSet<SiPixelCluster>::const_iterator    ClusterIterator;
       
       virtual void analyze(const edm::Event&, const edm::EventSetup&);
       virtual void beginJob() ;
       virtual void endJob() ;
       virtual void beginRun(const edm::Run&, edm::EventSetup const&) ;

       virtual void buildStructure(edm::EventSetup const&);
       virtual void bookMEs();

    private:
       edm::ParameterSet conf_;
       edm::InputTag src_;
       bool saveFile;
       bool isPIB;
       bool slowDown;
       int eventNo;
       DQMStore* theDMBE;
       std::map<uint32_t,SiPixelClusterModule*> thePixelStructure;
       bool modOn; 
       bool twoDimOn;
       bool reducedSet;
       //barrel:
       bool ladOn, layOn, phiOn;
       //forward:
       bool ringOn, bladeOn, diskOn; 
       bool smileyOn; //cluster sizeY vs Cluster eta plot 
       bool firstRun;
       int lumSec;
       int nLumiSecs;
       int nBigEvents;
       MonitorElement* bigFpixClusterEventRate;
       int bigEventSize;

  MonitorElement* meClPosLayer1;
  MonitorElement* meClPosLayer2;
  MonitorElement* meClPosLayer3;
  MonitorElement* meClPosDisk1pz;
  MonitorElement* meClPosDisk2pz;
  MonitorElement* meClPosDisk1mz;
  MonitorElement* meClPosDisk2mz;


};

#endif
