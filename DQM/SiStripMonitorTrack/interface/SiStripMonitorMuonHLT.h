#ifndef SiStripMonitorMuonHLT_SiStripMonitorMuonHLT_h
#define SiStripMonitorMuonHLT_SiStripMonitorMuonHLT_h


// system include files
#include <memory>
#include <string>
#include <cmath>

// user include files
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/LazyGetter.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
//#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"

#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"

#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "RecoTracker/TransientTrackingRecHit/interface/TSiStripRecHit2DLocalPos.h"

#include "DQM/SiStripCommon/interface/TkHistoMap.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"

#include "CalibTracker/SiStripCommon/interface/TkDetMap.h"

	//needed for normalisation
//Id
#include <DataFormats/SiStripDetId/interface/SiStripDetId.h>
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
//BoundPlane
#include <DataFormats/GeometrySurface/interface/BoundPlane.h>
#include <DataFormats/GeometrySurface/interface/BoundSurface.h>
#include <DataFormats/GeometrySurface/interface/Bounds.h>
#include <DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h>
#include <DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h>
//Point
#include <DataFormats/GeometryVector/interface/Point2DBase.h>
//Root
#include "TGraph.h"
#include "TFile.h"
#include "TH1F.h"
#include "TMath.h"
#include "Math/GenVector/CylindricalEta3D.h"

#include <DQMServices/Core/interface/DQMEDAnalyzer.h>


//////////////////////////
//
// class decleration
//

class SiStripMonitorMuonHLT : public DQMEDAnalyzer {

  //structure which contains all MonitorElement for a Layer
  // there is 34 layers in the tracker
  struct LayerMEs{ // MEs for Layer Level
      MonitorElement* EtaPhiAllClustersMap;
      MonitorElement* EtaDistribAllClustersMap;
      MonitorElement* PhiDistribAllClustersMap;
      MonitorElement* EtaPhiOnTrackClustersMap;
      MonitorElement* EtaDistribOnTrackClustersMap;
      MonitorElement* PhiDistribOnTrackClustersMap;
      MonitorElement* EtaPhiL3MuTrackClustersMap;
      MonitorElement* EtaDistribL3MuTrackClustersMap;
      MonitorElement* PhiDistribL3MuTrackClustersMap;
  };
						    

  public:
      explicit SiStripMonitorMuonHLT(const edm::ParameterSet& ps);
      ~SiStripMonitorMuonHLT();

   private:
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      void analyzeOnTrackClusters( const reco::Track* l3tk, const TrackerGeometry & theTracker, bool isL3MuTrack = true );
      virtual void endJob() ;
      void createMEs(DQMStore::IBooker &, const edm::EventSetup& es);
      void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
      //methods needed for normalisation
      float GetEtaWeight(std::string label, GlobalPoint gp);
      float GetPhiWeight(std::string label, GlobalPoint gp);
      void GeometryFromTrackGeom (const std::vector<DetId>& Dets,const TrackerGeometry & theTracker, const edm::EventSetup& iSetup,
                                  std::map<std::string,std::vector<float> > & m_PhiStripMod_Eta,std::map<std::string,std::vector<float> > & m_PhiStripMod_Nb);
      void Normalizer (const std::vector<DetId>& Dets,const TrackerGeometry & theTracker);
      void PrintNormalization (const std::vector<std::string>& v_LabelHisto);

      // ----------member data ---------------------------

      edm::ParameterSet parameters_;

      DQMStore* dbe_;   
      std::string monitorName_;
      std::string outputFile_;
      int counterEvt_;      ///counter
      int nTrig_;           /// mutriggered events
      int prescaleEvt_;     ///every n events
      bool verbose_;
      bool normalize_;
      bool printNormalize_;

      //booleans to active part of the code
      bool runOnClusters_;   //all clusters collection 
      bool runOnMuonCandidates_;  //L3 muons candidates
      bool runOnTracks_;     //tracks available in HLT stream

      //tag for collection taken as input
      edm::InputTag clusterCollectionTag_;
      edm::InputTag l3collectionTag_;
      edm::InputTag TrackCollectionTag_;

      edm::EDGetTokenT<edm::LazyGetter < SiStripCluster > > clusterCollectionToken_;
      edm::EDGetTokenT<reco::RecoChargedCandidateCollection> l3collectionToken_;
      edm::EDGetTokenT<reco::TrackCollection> TrackCollectionToken_;



      int HistoNumber; //nof layers in Tracker = 34 
      TkDetMap* tkdetmap_;
      std::map<std::string, LayerMEs> LayerMEMap;
      //2D info from TkHistoMap 
      TkHistoMap* tkmapAllClusters;
      TkHistoMap* tkmapOnTrackClusters;
      TkHistoMap* tkmapL3MuTrackClusters;

    
      // FOR NORMALISATION     
      std::map<std::string,std::vector<float> > m_BinPhi ;
      std::map<std::string,std::vector<float> > m_BinEta ;
      std::map<std::string,std::vector<float> > m_ModNormPhi;
      std::map<std::string,std::vector<float> > m_ModNormEta;
      


};
#endif
