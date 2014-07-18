#ifndef SiPixelMuonHLT_SiPixelMuonHLT_h
#define SiPixelMuonHLT_SiPixelMuonHLT_h
// -*- C++ -*-
//
// Package:     SiPixelMuonHLT
// Class  :     SiPixelMuonHLT
// 
/*
 Description: <one line class summary>

 Usage:
    <usage>

*/
//
//////////////////////////////////////////////////////////
//
// Original Author:  Dan Duggan
//         Created:
//
//////////////////////////////////////////////////////////

#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
//#include "DQM/SiPixelCommon/interface/SiPixelFolderOrganizer.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
//Pixel data formats
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
//DataFormats for MuonHLT
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeedCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeedCollection.h"
#include "DataFormats/Common/interface/TriggerResults.h"
// Geometry
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
//More
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"

#include <boost/cstdint.hpp>

 class SiPixelMuonHLT : public edm::EDAnalyzer {

   struct LayerMEs{
     MonitorElement* EtaPhiAllClustersMap;
   };

    public:
       explicit SiPixelMuonHLT(const edm::ParameterSet& conf);
       ~SiPixelMuonHLT();

       typedef edmNew::DetSet<SiPixelCluster>::const_iterator    ClusterIterator;
       
       virtual void analyze(const edm::Event&, const edm::EventSetup&);
       virtual void beginJob() ;
       virtual void endJob() ;

       virtual void Histo_init();

    private:
       edm::ParameterSet conf_;
       edm::InputTag src_;
       bool saveOUTput_;
       int eventNo;
       DQMStore* theDMBE;

       edm::ParameterSet parameters_;
       std::string monitorName_;
       std::string outputFile_;
  
       int counterEvt_;      ///counter
       int prescaleEvt_;     ///every n events
       bool verbose_;
       //int nTrigs;
       
       //std::vector<std::string> theTriggerBits;
       //std::vector<std::string> theDirectoryName;
       //std::vector<std::string> theHLTCollectionLevel;

       edm::InputTag clusterCollectionTag_;
       edm::InputTag rechitsCollectionTag_;
       edm::InputTag l3MuonCollectionTag_;

       SiStripFolderOrganizer folder_organizer;
       std::map<int, MonitorElement*> MEContainerAllBarrelEtaPhi;
       std::map<int, MonitorElement*> MEContainerAllBarrelZPhi;
       std::map<int, MonitorElement*> MEContainerAllBarrelEta;
       std::map<int, MonitorElement*> MEContainerAllBarrelZ;
       std::map<int, MonitorElement*> MEContainerAllBarrelPhi;
       std::map<int, MonitorElement*> MEContainerAllBarrelN;
       std::map<int, MonitorElement*> MEContainerAllEndcapXY;
       std::map<int, MonitorElement*> MEContainerAllEndcapPhi;
       std::map<int, MonitorElement*> MEContainerAllEndcapN;
       std::map<int, MonitorElement*> MEContainerOnTrackBarrelEtaPhi;
       std::map<int, MonitorElement*> MEContainerOnTrackBarrelZPhi;
       std::map<int, MonitorElement*> MEContainerOnTrackBarrelEta;
       std::map<int, MonitorElement*> MEContainerOnTrackBarrelZ;
       std::map<int, MonitorElement*> MEContainerOnTrackBarrelPhi;
       std::map<int, MonitorElement*> MEContainerOnTrackBarrelN;
       std::map<int, MonitorElement*> MEContainerOnTrackEndcapXY;
       std::map<int, MonitorElement*> MEContainerOnTrackEndcapPhi;
       std::map<int, MonitorElement*> MEContainerOnTrackEndcapN;

       //define Token(-s)
       edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > clustersToken_;
       edm::EDGetTokenT<edmNew::DetSetVector<SiPixelRecHit> > rechitsToken_;
       edm::EDGetTokenT<reco::RecoChargedCandidateCollection> l3MuonCollectionToken_;
 };

#endif
