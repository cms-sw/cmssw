
#ifndef SiStripMonitorTrack_H
#define SiStripMonitorTrack_H

// system include files
#include <memory>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_set>

// user include files
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"  
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetNew.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"

#include "AnalysisDataFormats/SiStripClusterInfo/interface/SiStripClusterInfo.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"

//******** Single include for the TkMap *************
#include "DQM/SiStripCommon/interface/TkHistoMap.h" 
//***************************************************

#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

class SiStripDCSStatus;
class GenericTriggerEventFlag;
class TrackerTopology;

//
// class declaration
//

class SiStripMonitorTrack : public DQMEDAnalyzer {
public:
  typedef TrackingRecHit::ConstRecHitPointer ConstRecHitPointer;
  enum RecHitType { Single=0, Matched=1, Projected=2, Null=3};
  explicit SiStripMonitorTrack(const edm::ParameterSet&);
  ~SiStripMonitorTrack();
  void dqmBeginRun(const edm::Run& run, const edm::EventSetup& es) ;
  virtual void endJob(void);
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

private:
  enum ClusterFlags {
    OffTrack,
    OnTrack
  };
  //booking
  void book(DQMStore::IBooker &, const TrackerTopology* tTopo);
  void bookModMEs(DQMStore::IBooker &, const uint32_t& );
  void bookLayerMEs(DQMStore::IBooker &, const uint32_t&, std::string&);
  void bookSubDetMEs(DQMStore::IBooker &, std::string& name);
  MonitorElement * bookME1D(DQMStore::IBooker & , const char*, const char*);
  MonitorElement * bookME2D(DQMStore::IBooker & , const char*, const char*);
  MonitorElement * bookME3D(DQMStore::IBooker & , const char*, const char*);
  MonitorElement * bookMEProfile(DQMStore::IBooker & , const char*, const char*);
  MonitorElement * bookMETrend(DQMStore::IBooker & , const char*, const char*);
  // internal evaluation of monitorables
  void AllClusters(const edm::Event& ev, const edm::EventSetup& es); 
  void trackStudyFromTrack(edm::Handle<reco::TrackCollection > trackCollectionHandle, const edm::EventSetup& es);
  void trackStudyFromTrajectory(edm::Handle<TrajTrackAssociationCollection> TItkAssociatorCollection, const edm::EventSetup& es);
  void trajectoryStudy(const edm::Ref<std::vector<Trajectory> > traj, const edm::EventSetup& es);
  //  void trajectoryStudy(const edm::Ref<std::vector<Trajectory> > traj, reco::TrackRef trackref, const edm::EventSetup& es);
  void trackStudy(const edm::Event& ev, const edm::EventSetup& es);
  //  LocalPoint project(const GeomDet *det,const GeomDet* projdet,LocalPoint position,LocalVector trackdirection)const;
  void hitStudy(const edm::EventSetup& es,
		const ProjectedSiStripRecHit2D* projhit,
		const SiStripMatchedRecHit2D*   matchedhit,
		const SiStripRecHit2D*          hit2D,
		const SiStripRecHit1D*          hit1D,
		LocalVector localMomentum);
  bool clusterInfos(SiStripClusterInfo* cluster, const uint32_t& detid, const TrackerTopology* tTopo, enum ClusterFlags flags, LocalVector LV);	
  template <class T> void RecHitInfo(const T* tkrecHit, LocalVector LV, const edm::EventSetup&);

  // fill monitorables 
  void fillModMEs(SiStripClusterInfo*,std::string,float);
  void fillMEs(SiStripClusterInfo*,uint32_t detid, const TrackerTopology* tTopo, float,enum ClusterFlags);
  inline void fillME(MonitorElement* ME,float value1){if (ME!=0)ME->Fill(value1);}
  inline void fillME(MonitorElement* ME,float value1,float value2){if (ME!=0)ME->Fill(value1,value2);}
  inline void fillME(MonitorElement* ME,float value1,float value2,float value3){if (ME!=0)ME->Fill(value1,value2,value3);}
  inline void fillME(MonitorElement* ME,float value1,float value2,float value3,float value4){if (ME!=0)ME->Fill(value1,value2,value3,value4);}
  void getSubDetTag(std::string& folder_name, std::string& tag);

  // ----------member data ---------------------------
  
private:
  DQMStore * dbe;
  edm::ParameterSet conf_;
  std::string histname; 
  LocalVector LV;
  float iOrbitSec;

  std::string topFolderName_;
  
  //******* TkHistoMaps
  TkHistoMap *tkhisto_StoNCorrOnTrack, *tkhisto_NumOnTrack, *tkhisto_NumOffTrack;  
  //******** TkHistoMaps
 
  struct ModMEs{  
    MonitorElement* ClusterStoNCorr;
    MonitorElement* ClusterCharge;
    MonitorElement* ClusterChargeCorr; 
    MonitorElement* ClusterWidth;
    MonitorElement* ClusterPos;
    MonitorElement* ClusterPGV;
  };

  struct LayerMEs{
    MonitorElement* ClusterStoNCorrOnTrack;
    MonitorElement* ClusterChargeCorrOnTrack;
    MonitorElement* ClusterChargeOnTrack;
    MonitorElement* ClusterChargeOffTrack;
    MonitorElement* ClusterNoiseOnTrack;
    MonitorElement* ClusterNoiseOffTrack;
    MonitorElement* ClusterWidthOnTrack;
    MonitorElement* ClusterWidthOffTrack;
    MonitorElement* ClusterPosOnTrack;
    MonitorElement* ClusterPosOffTrack;
  };
  struct SubDetMEs{
    int totNClustersOnTrack;
    int totNClustersOffTrack;
    MonitorElement* nClustersOnTrack;
    MonitorElement* nClustersTrendOnTrack;
    MonitorElement* nClustersOffTrack;
    MonitorElement* nClustersTrendOffTrack;
    MonitorElement* ClusterStoNCorrOnTrack;
    MonitorElement* ClusterChargeOnTrack;
    MonitorElement* ClusterChargeOffTrack;
    MonitorElement* ClusterStoNOffTrack;
 
  };  
  std::map<std::string, ModMEs> ModMEsMap;
  std::map<std::string, LayerMEs> LayerMEsMap;
  std::map<std::string, SubDetMEs> SubDetMEsMap;  
  
  edm::ESHandle<TrackerGeometry> tkgeom_;
  edm::ESHandle<SiStripDetCabling> SiStripDetCabling_;
  
  edm::ParameterSet Parameters;
  edm::InputTag Cluster_src_;
  
  edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> > clusterToken_;
  edm::EDGetTokenT<reco::TrackCollection> trackToken_;
  //  edm::EDGetTokenT<std::vector<Trajectory> > trajectoryToken_;
  edm::EDGetTokenT<TrajTrackAssociationCollection> trackTrajToken_;

  bool Mod_On_;
  bool Trend_On_;
  bool OffHisto_On_;
  bool HistoFlag_On_;
  bool ring_flag;
  bool TkHistoMap_On_;

  std::string TrackProducer_;
  std::string TrackLabel_;

  std::unordered_set<const SiStripCluster*> vPSiStripCluster;
  bool tracksCollection_in_EventTree;
  bool trackAssociatorCollection_in_EventTree;
  bool flag_ring;
  edm::EventNumber_t eventNb;
  int firstEvent;

  bool   applyClusterQuality_;
  double sToNLowerLimit_;  
  double sToNUpperLimit_;  
  double widthLowerLimit_;
  double widthUpperLimit_;

  SiStripDCSStatus* dcsStatus_;
  GenericTriggerEventFlag* genTriggerEventFlag_;
  SiStripFolderOrganizer folderOrganizer_;                                                                                                                                                                                                                                   
};
#endif
