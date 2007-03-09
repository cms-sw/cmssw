#ifndef ClusterAnalysis_h
#define ClusterAnalysis_h

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h" 
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"  
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
//needed for the geometry:
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"

//Data Formats
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "AnalysisDataFormats/SiStripClusterInfo/interface/SiStripClusterInfo.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/LTCDigi/interface/LTCDigi.h"

//Services
#include "CommonTools/SiStripZeroSuppression/interface/SiStripPedestalsService.h"
#include "CommonTools/SiStripZeroSuppression/interface/SiStripNoiseService.h"

#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include "FWCore/ParameterSet/interface/InputTag.h"

#include "TFile.h"
#include "TString.h"
#include "TROOT.h"
#include "TObjArray.h"
#include "TRandom.h"

#include "vector"
#include <memory>
#include <string>
#include <iostream>

//#include "ClusterTree.h"

namespace cms{
  class ClusterAnalysis : public edm::EDAnalyzer
    {
    public:
      
      ClusterAnalysis(const edm::ParameterSet& conf);
  
      ~ClusterAnalysis();
      
      void beginJob( const edm::EventSetup& es );
      
      void endJob();
      
      void analyze(const edm::Event& e, const edm::EventSetup& c);

    private:
      void book();
      void AllClusters();
      void trackStudy();
      bool clusterInfos(const SiStripClusterInfo* cluster, const uint32_t& detid,TString flag);	
      const SiStripClusterInfo* MatchClusterInfo(const SiStripCluster* cluster, const uint32_t& detid);	
      std::pair<std::string,uint32_t> GetSubDetAndLayer(const uint32_t& detid);

      void fillTH1(float,TString,bool,float=0);
      void fillTH2(float,float,TString,bool,float=0);

    private:
  
      edm::ParameterSet conf_;
      const StripTopology* topol;
      edm::ESHandle<TrackerGeometry> tkgeom;
      edm::ESHandle<SiStripDetCabling> SiStripDetCabling_;

      edm::Handle< edm::DetSetVector<SiStripClusterInfo> >  dsv_SiStripClusterInfo;
      edm::Handle< edm::DetSetVector<SiStripCluster> >  dsv_SiStripCluster;
      edm::Handle<reco::TrackCollection> trackCollection;
      edm::Handle<LTCDigiCollection> ltcdigis;
      edm::Handle<uint16_t> filterWord;

      std::vector<const SiStripCluster*> vPSiStripCluster;
      
      std::map<std::pair<std::string,uint32_t>,bool> DetectedLayers;

      std::string filename_;
      std::string psfilename_;
      int32_t psfiletype_;
      
      TFile* fFile;
      TObjArray* Hlist;
      
      int runNb;
      int eventNb;

      SiStripNoiseService SiStripNoiseService_;  
      SiStripPedestalsService SiStripPedestalsService_;  
      edm::InputTag Filter_src_;
      edm::InputTag Track_src_;
      edm::InputTag ClusterInfo_src_;
      edm::InputTag Cluster_src_;
      std::vector<uint32_t> ModulesToBeExcluded_;
      int EtaAlgo_;
      int NeighStrips_;
 
      bool not_the_first_event;

      bool tracksCollection_in_EventTree;
      bool ltcdigisCollection_in_EventTree;

      int countOn, countOff, countAll, NClus[4][3];

      TRandom rnd;      
    };
}
#endif
