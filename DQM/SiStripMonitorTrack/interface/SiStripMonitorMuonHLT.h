#ifndef SiStripMonitorMuonHLT_SiStripMonitorMuonHLT_h
#define SiStripMonitorMuonHLT_SiStripMonitorMuonHLT_h


// system include files
#include <memory>
#include <string>
#include <cmath>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/TriggerNames.h"
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
//#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
//#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

//
// class decleration
//

class SiStripMonitorMuonHLT : public edm::EDAnalyzer {

  //structure which contains all MonitorElement for a Layer
  // there is 34 layers in the tracker
  struct LayerMEs{ // MEs for Layer Level
      MonitorElement* EtaPhiAllClustersMap;
      MonitorElement* EtaDistribAllClustersMap;
      MonitorElement* PhiDistribAllClustersMap;
      MonitorElement* EtaPhiOnTrackClustersMap;
      MonitorElement* EtaDistribOnTrackClustersMap;
      MonitorElement* PhiDistribOnTrackClustersMap;
  };
						    

  public:
      explicit SiStripMonitorMuonHLT(const edm::ParameterSet& ps);
      ~SiStripMonitorMuonHLT();

   private:
      virtual void beginJob(const edm::EventSetup& es) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      void createMEs(const edm::EventSetup& es);

      // ----------member data ---------------------------

      edm::ParameterSet parameters_;

      DQMStore* dbe_;   
      std::string monitorName_;
      std::string outputFile_;
      int counterEvt_;      ///counter
      int nTrig_;           /// mutriggered events
      int prescaleEvt_;     ///every n events
      bool verbose_;

      //tag for collection taken as input
      edm::InputTag clusterCollectionTag_;
      edm::InputTag l3collectionTag_;

      int HistoNumber; //nof layers in Tracker = 34 
      TkDetMap* tkdetmap_;
      std::map<std::string, LayerMEs> LayerMEMap;

};
#endif
