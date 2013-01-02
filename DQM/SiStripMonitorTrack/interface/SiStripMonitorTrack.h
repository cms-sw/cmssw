
#ifndef SiStripMonitorTrack_H
#define SiStripMonitorTrack_H

// system include files
#include <memory>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>

// user include files
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

class SiStripDCSStatus;
class GenericTriggerEventFlag;
class TrackerTopology;
//
// class declaration
//

class SiStripMonitorTrack : public edm::EDAnalyzer {
public:
  typedef TransientTrackingRecHit::ConstRecHitPointer ConstRecHitPointer;
  enum RecHitType { Single=0, Matched=1, Projected=2, Null=3};
  explicit SiStripMonitorTrack(const edm::ParameterSet&);
  ~SiStripMonitorTrack();
  virtual void beginRun(const edm::Run& run, const edm::EventSetup& c);
  virtual void endJob(void);
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

private:
  enum ClusterFlags {
    OffTrack,
    OnTrack
  };
  //booking
  void book(edm::ESHandle<TrackerTopology>& tTopo);
  void bookModMEs(const uint32_t& );
  void bookLayerMEs(const uint32_t&, std::string&);
  void bookSubDetMEs(std::string& name);
  MonitorElement * bookME1D(const char*, const char*);
  MonitorElement * bookME2D(const char*, const char*);
  MonitorElement * bookME3D(const char*, const char*);
  MonitorElement * bookMEProfile(const char*, const char*);
  MonitorElement * bookMETrend(const char*, const char*);
  // internal evaluation of monitorables
  void AllClusters(const edm::Event& ev, const edm::EventSetup& es); 
  void trackStudy(const edm::Event& ev, const edm::EventSetup& es);
  //  LocalPoint project(const GeomDet *det,const GeomDet* projdet,LocalPoint position,LocalVector trackdirection)const;
  bool clusterInfos(SiStripClusterInfo* cluster, const uint32_t& detid, edm::ESHandle<TrackerTopology>& tTopo, enum ClusterFlags flags, LocalVector LV);	
  template <class T> void RecHitInfo(const T* tkrecHit, LocalVector LV,reco::TrackRef track_ref, const edm::EventSetup&);

  // fill monitorables 
  void fillModMEs(SiStripClusterInfo*,std::string,float);
  void fillMEs(SiStripClusterInfo*,uint32_t detid, edm::ESHandle<TrackerTopology>& tTopo, float,enum ClusterFlags);
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
    MonitorElement* ClusterChargeOffTrack;
    MonitorElement* ClusterStoNOffTrack;
 
  };  
  std::map<std::string, ModMEs> ModMEsMap;
  std::map<std::string, LayerMEs> LayerMEsMap;
  std::map<std::string, SubDetMEs> SubDetMEsMap;  
  
  edm::ESHandle<TrackerGeometry> tkgeom;
  edm::ESHandle<SiStripDetCabling> SiStripDetCabling_;
  
  edm::ParameterSet Parameters;
  edm::InputTag Cluster_src_;
  
  bool Mod_On_;
  bool Trend_On_;
  bool OffHisto_On_;
  bool HistoFlag_On_;
  bool ring_flag;
  bool TkHistoMap_On_;

  bool layerontrack;
  bool layerofftrack;
  bool layercharge;
  bool layerston;
  bool layerchargecorr;
  bool layerstoncorrontrack;
  bool layernoise;
  bool layerwidth;

  std::string TrackProducer_;
  std::string TrackLabel_;

  std::vector<uint32_t> ModulesToBeExcluded_;
  std::vector<const SiStripCluster*> vPSiStripCluster;
  bool tracksCollection_in_EventTree;
  bool trackAssociatorCollection_in_EventTree;
  bool flag_ring;
  int runNb, eventNb;
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
