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

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "AnalysisDataFormats/SiStripClusterInfo/interface/SiStripClusterInfo.h"
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfo.h"
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfoTrackAssociation.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "TString.h"

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
  void bookModMEs(TString, uint32_t);
  void bookTrendMEs(TString);
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
  std::pair<std::string,int32_t> GetSubDetAndLayer(const uint32_t& detid);
  std::vector<std::pair<const TrackingRecHit*,float> > SeparateHits( reco::TrackInfoRef & trackinforef );
  // fill monitorables 
  void fillModMEs(const SiStripClusterInfo*,TString);
  void fillTrendMEs(const SiStripClusterInfo*,TString);
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
  TString name;

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
          MonitorElement* ClusterPGV;
      };

  MonitorElement * NumberOfTracks;
  MonitorElement * NumberOfTracksTrend;
  MonitorElement * NumberOfRecHitsPerTrack;
  MonitorElement * NumberOfRecHitsPerTrackTrend;
  //MonitorElement * LocalAngle;

  std::map<TString, ModMEs> ModMEsMap;
  std::map<TString, MonitorElement*> MEMap;

  edm::Handle< edm::DetSetVector<SiStripClusterInfo> >  dsv_SiStripClusterInfo;
  edm::Handle< edm::DetSetVector<SiStripCluster> >  dsv_SiStripCluster;
  edm::Handle<reco::TrackCollection> trackCollection;
  edm::Handle<reco::TrackInfoTrackAssociationCollection> TItkAssociatorCollection;
  edm::ESHandle<TrackerGeometry> tkgeom;
  edm::ESHandle<SiStripDetCabling> SiStripDetCabling_;

  edm::ParameterSet Parameters;
  edm::InputTag Track_src_;
  edm::InputTag ClusterInfo_src_;
  edm::InputTag Cluster_src_;
  std::vector<uint32_t> ModulesToBeExcluded_;
  std::vector<const SiStripCluster*> vPSiStripCluster;
  std::map<std::pair<std::string,int32_t>,bool> DetectedLayers;
  SiStripFolderOrganizer folder_organizer;
  bool tracksCollection_in_EventTree;
  int runNb, eventNb;
  int firstEvent;
  int countOn, countOff, countAll, NClus[4][3];
};
#endif
