#ifndef ClusterAnalysis_h
#define ClusterAnalysis_h

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h" 
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CommonTools/TrackerMap/interface/TrackerMap.h"
//needed for the geometry:
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"  
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

//Data Formats
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "AnalysisDataFormats/SiStripClusterInfo/interface/SiStripClusterInfo.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfoFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfo.h"
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfoTrackAssociation.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/LTCDigi/interface/LTCDigi.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

//Services
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"

#include "FWCore/ParameterSet/interface/InputTag.h"

// Function to evaluate the local angles
//#include "AnalysisExamples/SiStripDetectorPerformance/interface/TrackLocalAngleTIF.h"

#include "TFile.h"
#include "TROOT.h"
#include "TRandom.h"
#include "THashList.h"

#include "vector"
#include <memory>
#include <string>
#include <iostream>

namespace cms{
  class ClusterAnalysis : public edm::EDAnalyzer
    {
    
    private:
	const TrackingRecHit* _TrackingRecHit;
	LocalVector LV;

    public:
      
      ClusterAnalysis(const edm::ParameterSet& conf);
  
      ~ClusterAnalysis();
      
      void beginRun(const edm::Run& run,  const edm::EventSetup& es );
      
      void endJob();
      
      void analyze(const edm::Event& e, const edm::EventSetup& c);

    private:

      void bookHlist( char* HistoType, char* ParameterSetLabel, TFileDirectory subDir, TString & HistoName,
		      char* xTitle="", char* yTitle="", char* zTitle="");		      

      void book();
      void AllClusters();
      void trackStudy();

      void RecHitInfo(const SiStripRecHit2D* tkrecHit, LocalVector LV,reco::TrackRef track_ref);

      bool clusterInfos(const SiStripClusterInfo* cluster, const uint32_t& detid,TString flag, LocalVector LV );	
      const SiStripClusterInfo* MatchClusterInfo(const SiStripCluster* cluster, const uint32_t& detid);	
      std::pair<std::string,uint32_t> GetSubDetAndLayer(const uint32_t& detid);

      void fillTH1(float,TString,bool,float=0);
      void fillTH2(float,float,TString,bool,float=0);
      void fillTProfile(float,float,TString,bool,float=0);
      void fillPedNoiseFromDB();

    private:
  
      edm::ParameterSet conf_;
      const StripTopology* topol;
      edm::ESHandle<TrackerGeometry> tkgeom;
      edm::ESHandle<SiStripDetCabling> SiStripDetCabling_;
      edm::ESHandle<SiStripQuality> SiStripQuality_;

      edm::Handle< edm::DetSetVector<SiStripClusterInfo> >  dsv_SiStripClusterInfo;
      edm::Handle< edm::DetSetVector<SiStripCluster> >  dsv_SiStripCluster;
      edm::Handle<reco::TrackCollection> trackCollection;
      edm::Handle<uint16_t> filterWord;
      edm::Handle<reco::TrackInfoTrackAssociationCollection> tkiTkAssCollection;
      std::vector<const SiStripCluster*> vPSiStripCluster;

      edm::Service<TFileService> fFile;
      
      std::map<std::pair<std::string,uint32_t>,bool> DetectedLayers;

      TString name;
      edm::ParameterSet Parameters;

      //      std::string filename_;
      std::string psfilename_;
      int32_t psfiletype_;
      int32_t psfilemode_;    

      THashList* Hlist;
      TrackerMap* tkMap_ClusOcc[3];//0 for onTrack, 1 for offTrack, 2 for All
      TrackerMap* tkInvHit;      
      int runNb;
      int eventNb;

      edm::ESHandle<SiStripPedestals> pedestalHandle;
      edm::ESHandle<SiStripNoises> noiseHandle;

      edm::InputTag Filter_src_;
      edm::InputTag Track_src_;
      edm::InputTag ClusterInfo_src_;
      edm::InputTag Cluster_src_;
      std::vector<uint32_t> ModulesToBeExcluded_;
      int EtaAlgo_;
      int NeighStrips_;
 
      bool not_the_first_event;

      bool tracksCollection_in_EventTree;
      bool trackAssociatorCollection_in_EventTree;

      int countOn, countOff, countAll, NClus[4][3];
      uint32_t istart;

      TRandom rnd;
     
      const TrackerGeometry * _tracker;

    };
}
#endif
