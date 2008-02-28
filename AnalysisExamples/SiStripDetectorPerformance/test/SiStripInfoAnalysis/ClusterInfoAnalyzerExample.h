#ifndef ClusterInfoAnalyzerExample_h
#define ClusterInfoAnalyzerExample_h


#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h" 

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"


#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"  
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

//needed for the geometry:
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

//Data Formats
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include "AnalysisDataFormats/SiStripClusterInfo/interface/SiStripClusterInfo.h"
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfo.h"
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfoTrackAssociation.h"
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfoFwd.h"


//Services
#include "CommonTools/SiStripZeroSuppression/interface/SiStripPedestalsService.h"
#include "CommonTools/SiStripZeroSuppression/interface/SiStripNoiseService.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

//DQM
#include "DQMServices/Core/interface/MonitorElement.h"


#include "TFile.h"
#include "TROOT.h"
#include "TObjArray.h"
#include "TRandom.h"
#include "THashList.h"

#include "vector"
#include <memory>
#include <string>
#include <iostream>


class DaqMonitorBEInterface;

class ClusterInfoAnalyzerExample : public edm::EDAnalyzer
{
  
 public:
  
  explicit ClusterInfoAnalyzerExample(const edm::ParameterSet& conf);
  ~ClusterInfoAnalyzerExample();
  
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginJob(edm::EventSetup const&) ;
  virtual void endJob() ;
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
  virtual void endRun(const edm::Run&, const edm::EventSetup&);
  
  struct ModMEs{ // MEs for one single detector module
    MonitorElement* ClusterPosition;
    MonitorElement* ClusterWidth;
    MonitorElement* ClusterCharge;
    MonitorElement* ClusterMaxPosition;
    MonitorElement* ClusterMaxCharge;
    MonitorElement* ClusterChargeLeft;
    MonitorElement* ClusterChargeRight;
    MonitorElement* ClusterNoise;
    MonitorElement* ClusterNoiseRescaledByGain;
    MonitorElement* ClusterSignalOverNoise;
    MonitorElement* ClusterSignalOverNoiseRescaledByGain;
    MonitorElement* ModuleNrOfClusters;
    MonitorElement* ModuleNrOfClusterizedStrips;
    MonitorElement* ModuleLocalOccupancy;//occupancy calculations
  };
  
 private: 
  
  void ResetME(MonitorElement* h1);
  void ResetModuleMEs(uint32_t idet);
  void createMEs(const edm::EventSetup& es);
  
  void getClusterInfoFromRecHit(const SiStripRecHit2D* trackerRecHit_, 
		                LocalVector       clusterLocalVector_,
		                reco::TrackRef              trackRef_,
		                const edm::EventSetup&            es);
  
  
  bool fillClusterHistos(SiStripClusterInfo*       clusterInfo_,
			 const uint32_t&                  detid_,
			 TString                          flag , 
			 const LocalVector clusterLocalVector_); 
  
 private :
  
  edm::ParameterSet conf_;       
 
  std::string theCMNSubtractionMode;
  edm::InputTag theTrackSourceLabel; 
  edm::InputTag theTrackTrackInfoAssocLabel; 
  edm::InputTag theClusterSourceLabel;     
  
  std::vector<uint32_t> theModulesToBeExcluded; 
 
  DaqMonitorBEInterface* daqMonInterface_; 
  bool show_mechanical_structure_view; 
  bool reset_each_run;  
  
  
  unsigned long long cacheID_;
  
  std::map<uint32_t, ModMEs> clusterMEs_;
         
  MonitorElement* position_of_each_cluster;     
  MonitorElement* width_of_each_cluster;            
  MonitorElement* charge_of_each_cluster;          
  MonitorElement* maxPosition_of_each_cluster;            
  MonitorElement* maxCharge_of_each_cluster;           
  MonitorElement* chargeLeft_of_each_cluster;         
  MonitorElement* chargeRight_of_each_cluster;       
  MonitorElement* noise_of_each_cluster;             
  MonitorElement* noiseRescaledByGain_of_each_cluster; 
  MonitorElement* signalOverNoise_of_each_cluster;             
  MonitorElement* signalOverNoiseRescaledByGain_of_each_cluster; 
  
  
  int runNb;
  int eventNb;
  
  edm::Handle<reco::TrackCollection>                     trackHandle_;
  edm::Handle<reco::TrackInfoTrackAssociationCollection> trackTrackInfoAssocHandle_;
  edm::Handle< edm::DetSetVector<SiStripCluster> >       clusterHandle_;

  edm::ESHandle<TrackerGeometry> trackerGeom_;   
  edm::ESHandle<SiStripDetCabling> stripDetCabling_;
  
  LocalVector clusterLocalVector_;

  int totalNumberOfClustersOnTrack;
  int countOn;
  
  
  
  
};

#endif
