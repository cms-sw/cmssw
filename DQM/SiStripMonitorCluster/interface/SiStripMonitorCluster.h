#ifndef SiStripMonitorCluster_SiStripMonitorCluster_h
#define SiStripMonitorCluster_SiStripMonitorCluster_h
// -*- C++ -*-
// Package:     SiStripMonitorCluster
// Class  :     SiStripMonitorCluster
/**\class SiStripMonitorCluster SiStripMonitorCluster.h DQM/SiStripMonitorCluster/interface/SiStripMonitorCluster.h
   Data Quality Monitoring source of the Silicon Strip Tracker. Produces histograms related to clusters.
*/
// Original Author:  dkcira
//         Created:  Wed Feb  1 16:47:14 CET 2006
#include <memory>
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQM/SiStripCommon/interface/TkHistoMap.h"

#include <vector>

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

class DQMStore;
class SiStripDetCabling;
class SiStripCluster;
class SiPixelCluster;
class EventWithHistory;
class APVCyclePhaseCollection;
class SiStripDCSStatus;
class GenericTriggerEventFlag;

class SiStripMonitorCluster : public DQMEDAnalyzer {
 public:
  explicit SiStripMonitorCluster(const edm::ParameterSet&);
  ~SiStripMonitorCluster();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  //virtual void beginJob() ;
  virtual void endJob() ;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) ;
  
  struct ModMEs{ // MEs for one single detector module

    MonitorElement* NumberOfClusters = 0;
    MonitorElement* ClusterPosition = 0;
    MonitorElement* ClusterDigiPosition = 0;
    MonitorElement* ClusterWidth = 0;
    MonitorElement* ClusterCharge = 0;
    MonitorElement* ClusterNoise = 0;
    MonitorElement* ClusterSignalOverNoise = 0;
    MonitorElement* ClusterSignalOverNoiseVsPos = 0;
    MonitorElement* ModuleLocalOccupancy = 0;
    MonitorElement* NrOfClusterizedStrips = 0; // can be used at client level for occupancy calculations
  };

  struct LayerMEs{ // MEs for Layer Level
    MonitorElement* LayerClusterStoN = 0;
    MonitorElement* LayerClusterStoNTrend = 0;
    MonitorElement* LayerClusterCharge = 0;
    MonitorElement* LayerClusterChargeTrend = 0;
    MonitorElement* LayerClusterNoise = 0;
    MonitorElement* LayerClusterNoiseTrend = 0;
    MonitorElement* LayerClusterWidth = 0;
    MonitorElement* LayerClusterWidthTrend = 0;
    MonitorElement* LayerLocalOccupancy = 0;
    MonitorElement* LayerLocalOccupancyTrend = 0;
    MonitorElement* LayerNumberOfClusterProfile = 0;
    MonitorElement* LayerClusterWidthProfile = 0;

  };

  struct SubDetMEs{ // MEs for Subdetector Level
    int totNClusters = 0; 
    MonitorElement* SubDetTotClusterTH1 = 0;
    MonitorElement* SubDetTotClusterProf = 0;
    MonitorElement* SubDetClusterApvProf = 0;
    MonitorElement* SubDetClusterApvTH2 = 0;
    MonitorElement* SubDetClusterDBxCycleProf = 0;
    MonitorElement* SubDetApvDBxProf2 = 0;
  };

  struct ClusterProperties { // Cluster Properties
    float charge;
    float position;
    short start;
    short width;
    float noise;
  };

  MonitorElement* GlobalApvCycleDBxTH2 = 0; 
  MonitorElement* GlobalCStripVsCpix = 0;
  MonitorElement* PixVsStripMultiplicityRegions = 0;
  MonitorElement* GlobalMainDiagonalPosition = 0;
  MonitorElement* StripNoise2Cycle = 0;
  MonitorElement* StripNoise3Cycle = 0;
  MonitorElement* NumberOfPixelClus = 0;
  MonitorElement* NumberOfStripClus = 0;

  MonitorElement* BPTXrateTrend = 0;

 private:

  void createMEs(const edm::EventSetup& es , DQMStore::IBooker & ibooker);
  void createLayerMEs(std::string label, int ndets , DQMStore::IBooker & ibooker );
  void createModuleMEs(ModMEs& mod_single, uint32_t detid , DQMStore::IBooker & ibooker);
  void createSubDetMEs(std::string label , DQMStore::IBooker & ibooker);
  int FindRegion(int nstrip,int npixel);
  void fillModuleMEs(ModMEs& mod_mes, ClusterProperties& cluster);
  void fillLayerMEs(LayerMEs&, ClusterProperties& cluster, float timeinorbit);

  void ResetModuleMEs(uint32_t idet);

  inline void fillME(MonitorElement* ME,float value1){if (ME!=0)ME->Fill(value1);}
  inline void fillME(MonitorElement* ME,float value1,float value2){if (ME!=0)ME->Fill(value1,value2);}
  inline void fillME(MonitorElement* ME,float value1,float value2,float value3){if (ME!=0)ME->Fill(value1,value2,value3);}
  inline void fillME(MonitorElement* ME,float value1,float value2,float value3,float value4){if (ME!=0)ME->Fill(value1,value2,value3,value4);}
  MonitorElement * bookMETrend(const char*, const char* , DQMStore::IBooker & ibooker);
  MonitorElement* bookME1D(const char* ParameterSetLabel, const char* HistoName , DQMStore::IBooker & ibooker);

 private:
  DQMStore* dqmStore_;
  edm::ParameterSet conf_;
  std::map<uint32_t, ModMEs> ModuleMEsMap;
  std::map<std::string, LayerMEs> LayerMEsMap;
  std::map<std::string, std::vector< uint32_t > > LayerDetMap;
  std::map<std::string, SubDetMEs> SubDetMEsMap;
  std::map<std::string, std::string> SubDetPhasePartMap;

  // flags
  bool show_mechanical_structure_view, show_readout_view, show_control_view, select_all_detectors, reset_each_run;
  unsigned long long m_cacheID_;

  edm::ESHandle<SiStripDetCabling> SiStripDetCabling_;
  std::vector<uint32_t> ModulesToBeExcluded_;

  edm::ParameterSet Parameters;

  // TkHistoMap added
  TkHistoMap* tkmapcluster; 

  int runNb, eventNb;
  int firstEvent;

  bool layerswitchncluson;
  bool layerswitchcluschargeon;
  bool layerswitchclusstonon;
  bool layerswitchclusstonVsposon;
  bool layerswitchclusposon;
  bool layerswitchclusdigiposon;
  bool layerswitchclusnoiseon;
  bool layerswitchcluswidthon;
  bool layerswitchlocaloccupancy;
  bool layerswitchnrclusterizedstrip;
  bool layerswitchnumclusterprofon;
  bool layerswitchclusterwidthprofon;
  
  bool globalswitchstripnoise2apvcycle;
  bool globalswitchstripnoise3apvcycle;
  bool globalswitchmaindiagonalposition;

  bool moduleswitchncluson;
  bool moduleswitchcluschargeon;
  bool moduleswitchclusstonon;
  bool moduleswitchclusstonVsposon;
  bool moduleswitchclusposon;
  bool moduleswitchclusdigiposon;
  bool moduleswitchclusnoiseon;
  bool moduleswitchcluswidthon;
  bool moduleswitchlocaloccupancy;
  bool moduleswitchnrclusterizedstrip;
  bool subdetswitchtotclusprofon;
  bool subdetswitchapvcycleprofon;
  bool subdetswitchapvcycleth2on;
  bool subdetswitchapvcycledbxprof2on;
  bool subdetswitchdbxcycleprofon;
  bool subdetswitchtotclusth1on;
  bool globalswitchapvcycledbxth2on;
  bool globalswitchcstripvscpix;
  bool globalswitchMultiRegions;
  bool clustertkhistomapon;
  bool createTrendMEs;

  bool Mod_On_;
  bool ClusterHisto_;

  std::string topFolderName_;
  std::string qualityLabel_;

  /*
  edm::InputTag clusterProducerStrip_;
  edm::InputTag clusterProducerPix_;
  edm::InputTag historyProducer_;  
  edm::InputTag apvPhaseProducer_;
  */

  edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> >         clusterProducerStripToken_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> >         clusterProducerPixToken_;
  edm::EDGetTokenT<EventWithHistory>        historyProducerToken_;
  edm::EDGetTokenT<APVCyclePhaseCollection> apvPhaseProducerToken_;

  bool   applyClusterQuality_;
  double sToNLowerLimit_;  
  double sToNUpperLimit_;  
  double widthLowerLimit_;
  double widthUpperLimit_;

  double k0;
  double q0;
  double dk0;
  double maxClus;
  double minPix;

  SiStripDCSStatus* dcsStatus_;

  // add for selecting on ZeroBias events in the MinimumBias PD
  GenericTriggerEventFlag* genTriggerEventFlagBPTXfilter_;
  GenericTriggerEventFlag* genTriggerEventFlagPixelDCSfilter_;
  GenericTriggerEventFlag* genTriggerEventFlagStripDCSfilter_;

  bool passBPTXfilter_;
  bool passPixelDCSfilter_;
  bool passStripDCSfilter_;
};
#endif
