#ifndef SiStripMonitorTrack_H
#define SiStripMonitorTrack_H

// system include files
#include <memory>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//---------------- from ClusterAnalysis

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"  
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"

//needed for the geometry:
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"

//Data Formats
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
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

//Services
#include "CommonTools/SiStripZeroSuppression/interface/SiStripPedestalsService.h"
#include "CommonTools/SiStripZeroSuppression/interface/SiStripNoiseService.h"

#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include "FWCore/ParameterSet/interface/InputTag.h"

#include "TString.h"

#include "vector"
#include <memory>
#include <string>
#include <iostream>
//
// class declaration
//

class SiStripMonitorTrack : public edm::EDAnalyzer {
 public:
  explicit SiStripMonitorTrack(const edm::ParameterSet&);
  ~SiStripMonitorTrack();
  virtual void beginJob(edm::EventSetup const& );
  virtual void endJob(void);
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

 private:
  //booking
  void book();
  void bookModMEs(TString);
  MonitorElement * bookME1D(const char*, const char*);
  MonitorElement * bookME2D(const char*, const char*);
  MonitorElement * bookME3D(const char*, const char*);
  MonitorElement * bookMEProfile(const char*, const char*);
  MonitorElement * bookMETrend(const char*, const char*);
  // internal evaluation of monitorables
  void AllClusters();
  void trackStudy();
  bool clusterInfos(const SiStripClusterInfo* cluster, const uint32_t& detid,TString flag, float angle = 0);	
  const SiStripClusterInfo* MatchClusterInfo(const SiStripCluster* cluster, const uint32_t& detid);	
  std::pair<std::string,uint32_t> GetSubDetAndLayer(const uint32_t& detid);
  float clusEta(const SiStripClusterInfo*);
  std::vector<std::pair<const TrackingRecHit*,float> > SeparateHits( reco::TrackInfoRef & trackinforef );
  // fill monitorables 
  void fillModMEs(const SiStripClusterInfo*,TString,bool);
  void fillTrend(MonitorElement* ME,float value1);
  inline void fillME(MonitorElement* ME,float value1){if (ME!=0)ME->Fill(value1);}
  inline void fillME(MonitorElement* ME,float value1,float value2){if (ME!=0)ME->Fill(value1,value2);}
  inline void fillME(MonitorElement* ME,float value1,float value2,float value3){if (ME!=0)ME->Fill(value1,value2,value3);}
  inline void fillME(MonitorElement* ME,float value1,float value2,float value3,float value4){if (ME!=0)ME->Fill(value1,value2,value3,value4);}

  // ----------member data ---------------------------
      
 private:
  DaqMonitorBEInterface * dbe;
  edm::ParameterSet conf_;
  std::string histname; 

  struct ModMEs{
	  ModMEs():    
          nClusters(0),
          nClustersTrend(0),
          ClusterStoN(0),
          ClusterStoNTrend(0),
          ClusterSignal(0),
          ClusterSignalTrend(0),
          ClusterNoise(0),
          ClusterNoiseTrend(0),
          ClusterWidth(0),
          ClusterWidthTrend(0),
          ClusterPos(0),
          ClusterEta(0),
          ClusterPGV(0){};
        MonitorElement* nClusters;
        MonitorElement* nClustersTrend;
        MonitorElement* ClusterStoN;
        MonitorElement* ClusterStoNTrend;
        MonitorElement* ClusterSignal;
        MonitorElement* ClusterSignalTrend;
        MonitorElement* ClusterNoise;
        MonitorElement* ClusterNoiseTrend;
        MonitorElement* ClusterWidth;
        MonitorElement* ClusterWidthTrend;
        MonitorElement* ClusterPos;
        MonitorElement* ClusterEta;
        MonitorElement* ClusterPGV;
      };

  MonitorElement * NumberOfTracks;
  MonitorElement * NumberOfTracksTrend;
  MonitorElement * NumberOfRecHitsPerTrack;
  MonitorElement * NumberOfRecHitsPerTrackTrend;

  std::map<uint32_t, ModMEs> ClusterMEforDet; //still unused
  std::map<uint32_t, ModMEs> ClusterMEforLayer; //still unused
  std::map<TString, ModMEs> ModMEsMap;
  std::map<TString, MonitorElement*> MEMap;

  const StripTopology* topol;
  edm::ESHandle<TrackerGeometry> tkgeom;
  edm::ESHandle<SiStripDetCabling> SiStripDetCabling_;
  edm::Handle< edm::DetSetVector<SiStripClusterInfo> >  dsv_SiStripClusterInfo;
  edm::Handle< edm::DetSetVector<SiStripCluster> >  dsv_SiStripCluster;
  edm::Handle<reco::TrackCollection> trackCollection;
  edm::Handle<reco::TrackInfoTrackAssociationCollection> TItkAssociatorCollection;
  std::vector<const SiStripCluster*> vPSiStripCluster;
  std::map<std::pair<std::string,uint32_t>,bool> DetectedLayers;
  TString name;
  edm::ParameterSet Parameters;
  int runNb, eventNb;
  int firstEvent;
  SiStripNoiseService SiStripNoiseService_;  
  SiStripPedestalsService SiStripPedestalsService_;  
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

  // For track angles
  typedef std::vector<std::pair<const TrackingRecHit *, float> > HitAngleAssociation;
  typedef std::vector<std::pair<const TrackingRecHit *, LocalVector > > HitLclDirAssociation;
  typedef std::vector<std::pair<const TrackingRecHit *, GlobalVector> > HitGlbDirAssociation;
  const TrackerGeometry * _tracker;
  reco::TrackInfo::TrajectoryInfo::const_iterator _tkinfoiter;
  HitAngleAssociation oXZHitAngle;
  HitAngleAssociation oYZHitAngle;
  HitLclDirAssociation oLocalDir;
  HitGlbDirAssociation oGlobalDir;
};
#endif
