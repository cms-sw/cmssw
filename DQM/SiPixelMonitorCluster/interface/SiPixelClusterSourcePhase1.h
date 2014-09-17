#ifndef SiPixelMonitorCluster_SiPixelClusterSourcePhase1_h
#define SiPixelMonitorCluster_SiPixelClusterSourcePhase1_h
// -*- C++ -*-
//
// Package:     SiPixelMonitorCluster
// Class  :     SiPixelClusterSourcePhase1
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

#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "DQM/SiPixelMonitorCluster/interface/SiPixelClusterModulePhase1.h"

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

 class SiPixelClusterSourcePhase1 : public DQMEDAnalyzer {
    public:
       explicit SiPixelClusterSourcePhase1(const edm::ParameterSet& conf);
       ~SiPixelClusterSourcePhase1();

       typedef edmNew::DetSet<SiPixelCluster>::const_iterator    ClusterIterator;
       
       virtual void analyze(const edm::Event&, const edm::EventSetup&);
       virtual void dqmBeginRun(const edm::Run&, edm::EventSetup const&) ;
       virtual void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

       virtual void buildStructure(edm::EventSetup const&);
       virtual void bookMEs(DQMStore::IBooker &);

    private:
       edm::ParameterSet conf_;
       edm::InputTag src_;
       bool saveFile;
       bool isPIB;
       bool slowDown;
       int eventNo;
       std::map<uint32_t,SiPixelClusterModulePhase1*> thePixelStructure;
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
  MonitorElement* meClPosLayer4;
  MonitorElement* meClPosDisk1pz;
  MonitorElement* meClPosDisk2pz;
  MonitorElement* meClPosDisk3pz;
  MonitorElement* meClPosDisk1mz;
  MonitorElement* meClPosDisk2mz;
  MonitorElement* meClPosDisk3mz;

  //define Token(-s)
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > srcToken_;
};

#endif
