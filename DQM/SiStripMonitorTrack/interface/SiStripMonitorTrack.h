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
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"  
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"

#include "AnalysisDataFormats/SiStripClusterInfo/interface/SiStripClusterInfo.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
//#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "TString.h"

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
  //booking
  void book();
  void bookModMEs(TString, uint32_t);
  void bookTrendMEs(TString, int32_t,uint32_t,std::string flag);
  void bookSubDetMEs(TString name,TString flag);
  MonitorElement * bookME1D(const char*, const char*);
  MonitorElement * bookME2D(const char*, const char*);
  MonitorElement * bookME3D(const char*, const char*);
  MonitorElement * bookMEProfile(const char*, const char*);
  MonitorElement * bookMETrend(const char*, const char*);
  // internal evaluation of monitorables
  void AllClusters(const edm::EventSetup& es);
  void trackStudy(const edm::EventSetup& es);
  //  LocalPoint project(const GeomDet *det,const GeomDet* projdet,LocalPoint position,LocalVector trackdirection)const;
  bool clusterInfos(SiStripClusterInfo* cluster, const uint32_t& detid,std::string flag, LocalVector LV);	
  void RecHitInfo(const SiStripRecHit2D* tkrecHit, LocalVector LV,reco::TrackRef track_ref, const edm::EventSetup&);
  // fill monitorables 
  void fillModMEs(SiStripClusterInfo*,TString,float);
  void fillTrendMEs(SiStripClusterInfo*,std::string,float,std::string);
  void fillTrend(MonitorElement* ME,float value1);
  inline void fillME(MonitorElement* ME,float value1){if (ME!=0)ME->Fill(value1);}
  inline void fillME(MonitorElement* ME,float value1,float value2){if (ME!=0)ME->Fill(value1,value2);}
  inline void fillME(MonitorElement* ME,float value1,float value2,float value3){if (ME!=0)ME->Fill(value1,value2,value3);}
  inline void fillME(MonitorElement* ME,float value1,float value2,float value3,float value4){if (ME!=0)ME->Fill(value1,value2,value3,value4);}

  // ----------member data ---------------------------
      
 private:
  DQMStore * dbe;
  edm::ParameterSet conf_;
  std::string histname; 
  TString name;
  LocalVector LV;

  struct ModMEs{
	  ModMEs():    
          nClusters(0),
          nClustersTrend(0),
          ClusterStoN(0),
          ClusterStoNCorr(0),
          ClusterStoNTrend(0),
          ClusterStoNCorrTrend(0),
          ClusterCharge(0),
          ClusterChargeCorr(0),
          ClusterChargeTrend(0),
          ClusterChargeCorrTrend(0),
          ClusterNoise(0),
          ClusterNoiseTrend(0),
          ClusterWidth(0),
          ClusterWidthTrend(0),
          ClusterPos(0),
          ClusterPGV(0){};
          MonitorElement* nClusters;
          MonitorElement* nClustersTrend;
          MonitorElement* ClusterStoN;
          MonitorElement* ClusterStoNCorr;
          MonitorElement* ClusterStoNTrend;
          MonitorElement* ClusterStoNCorrTrend;
          MonitorElement* ClusterCharge;
          MonitorElement* ClusterChargeCorr; 
          MonitorElement* ClusterChargeTrend;
          MonitorElement* ClusterChargeCorrTrend;
          MonitorElement* ClusterNoise;
          MonitorElement* ClusterNoiseTrend;
          MonitorElement* ClusterWidth;
          MonitorElement* ClusterWidthTrend;
          MonitorElement* ClusterPos;
          MonitorElement* ClusterPGV;
      };

  std::map<TString, ModMEs> ModMEsMap;
  std::map<TString, MonitorElement*> MEMap;

  edm::Handle< edm::DetSetVector<SiStripCluster> >  dsv_SiStripCluster;

  edm::Handle<std::vector<Trajectory> > TrajectoryCollection;
  edm::Handle<reco::TrackCollection > trackCollection;
  edm::Handle<TrajTrackAssociationCollection> TItkAssociatorCollection;
  
  edm::ESHandle<TrackerGeometry> tkgeom;
  edm::ESHandle<SiStripDetCabling> SiStripDetCabling_;

  edm::ParameterSet Parameters;
  edm::InputTag Cluster_src_;

  bool Mod_On_;
  bool OffHisto_On_;
  int off_Flag;
  std::vector<uint32_t> ModulesToBeExcluded_;
  std::vector<const SiStripCluster*> vPSiStripCluster;
  std::map<std::pair<std::string,int32_t>,bool> DetectedLayers;
  SiStripFolderOrganizer folder_organizer;
  bool tracksCollection_in_EventTree;
  bool trackAssociatorCollection_in_EventTree;
  int runNb, eventNb;
  int firstEvent;
  int countOn, countOff, countAll, NClus[4][3];
  uint32_t neighbourStripNumber;
};
#endif
