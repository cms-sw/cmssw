#ifndef SiStripMonitorCluster_SiStripMonitorCluster_h
#define SiStripMonitorCluster_SiStripMonitorCluster_h
// -*- C++ -*-
// Package:     SiStripMonitorCluster
// Class  :     SiStripMonitorCluster
/**\class SiStripMonitorCluster SiStripMonitorCluster.h
   DQM/SiStripMonitorCluster/interface/SiStripMonitorCluster.h Data Quality
   Monitoring source of the Silicon Strip Tracker. Produces histograms related
   to clusters.
*/
// Original Author:  dkcira
//         Created:  Wed Feb  1 16:47:14 CET 2006
#include <memory>
#include "DQM/SiStripCommon/interface/TkHistoMap.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include <vector>

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "CalibTracker/SiStripCommon/interface/TkDetMap.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

class SiStripCluster;
class SiPixelCluster;
class EventWithHistory;
class APVCyclePhaseCollection;
class SiStripDCSStatus;
class GenericTriggerEventFlag;

class SiStripMonitorCluster : public DQMEDAnalyzer {
public:
  explicit SiStripMonitorCluster(const edm::ParameterSet&);
  ~SiStripMonitorCluster() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;

  struct ModMEs {  // MEs for one single detector module

    MonitorElement* NumberOfClusters = nullptr;
    MonitorElement* ClusterPosition = nullptr;
    MonitorElement* ClusterDigiPosition = nullptr;
    MonitorElement* ClusterWidth = nullptr;
    MonitorElement* ClusterCharge = nullptr;
    MonitorElement* ClusterNoise = nullptr;
    MonitorElement* ClusterSignalOverNoise = nullptr;
    MonitorElement* ClusterSignalOverNoiseVsPos = nullptr;
    MonitorElement* ModuleLocalOccupancy = nullptr;
    MonitorElement* NrOfClusterizedStrips = nullptr;  // can be used at client level for occupancy calculations
    MonitorElement* Module_ClusWidthVsAmpTH2 = nullptr;
  };

  struct LayerMEs {  // MEs for Layer Level
    MonitorElement* LayerClusterStoN = nullptr;
    MonitorElement* LayerClusterStoNTrend = nullptr;
    MonitorElement* LayerClusterCharge = nullptr;
    MonitorElement* LayerClusterChargeTrend = nullptr;
    MonitorElement* LayerClusterNoise = nullptr;
    MonitorElement* LayerClusterNoiseTrend = nullptr;
    MonitorElement* LayerClusterWidth = nullptr;
    MonitorElement* LayerClusterWidthTrend = nullptr;
    MonitorElement* LayerLocalOccupancy = nullptr;
    MonitorElement* LayerLocalOccupancyTrend = nullptr;
    MonitorElement* LayerNumberOfClusterProfile = nullptr;
    MonitorElement* LayerNumberOfClusterPerRingTrend = nullptr;
    MonitorElement* LayerNumberOfClusterTrend = nullptr;
    MonitorElement* LayerClusterWidthProfile = nullptr;
    MonitorElement* LayerClusWidthVsAmpTH2 = nullptr;
    MonitorElement* LayerClusterPosition = nullptr;
  };

  struct SubDetMEs {  // MEs for Subdetector Level
    int totNClusters = 0;
    MonitorElement* SubDetTotClusterTH1 = nullptr;
    MonitorElement* SubDetTotClusterProf = nullptr;
    MonitorElement* SubDetClusterApvProf = nullptr;
    MonitorElement* SubDetClusterApvTH2 = nullptr;
    MonitorElement* SubDetClusterDBxCycleProf = nullptr;
    MonitorElement* SubDetApvDBxProf2 = nullptr;
    MonitorElement* SubDetClusterChargeTH1 = nullptr;
    MonitorElement* SubDetClusterWidthTH1 = nullptr;
    MonitorElement* SubDetClusWidthVsAmpTH2 = nullptr;
    MonitorElement* SubDetNumberOfClusterPerLayerTrend = nullptr;
  };

  struct ClusterProperties {  // Cluster Properties
    float charge;
    float position;
    short start;
    short width;
    float noise;
  };

  MonitorElement* GlobalApvCycleDBxTH2 = nullptr;
  MonitorElement* GlobalDBxTH1 = nullptr;
  MonitorElement* GlobalDBxCycleTH1 = nullptr;
  MonitorElement* GlobalCStripVsCpix = nullptr;
  MonitorElement* GlobalABXTH1_CSCP = nullptr;
  MonitorElement* PixVsStripMultiplicityRegions = nullptr;
  MonitorElement* GlobalMainDiagonalPosition = nullptr;
  MonitorElement* GlobalMainDiagonalPosition_vs_BX = nullptr;
  MonitorElement* GlobalTH2MainDiagonalPosition_vs_BX;
  MonitorElement* StripNoise2Cycle = nullptr;
  MonitorElement* StripNoise3Cycle = nullptr;
  MonitorElement* NumberOfPixelClus = nullptr;
  MonitorElement* NumberOfStripClus = nullptr;
  MonitorElement* BPTXrateTrend = nullptr;
  MonitorElement* NclusVsCycleTimeProf2D = nullptr;
  MonitorElement* ClusWidthVsAmpTH2 = nullptr;
  MonitorElement* NumberOfStripClus_vs_BX = nullptr;  // plot n. 3
  MonitorElement* NumberOfPixelClus_vs_BX = nullptr;  // plot n. 4
  MonitorElement* NumberOfFEDClus = nullptr;

private:
  void createMEs(const edm::EventSetup& es, DQMStore::IBooker& ibooker);
  void createLayerMEs(std::string label, int ndets, DQMStore::IBooker& ibooker);
  void createModuleMEs(ModMEs& mod_single, uint32_t detid, DQMStore::IBooker& ibooker, const SiStripDetCabling&);
  void createSubDetMEs(std::string label, DQMStore::IBooker& ibooker);
  int FindRegion(int nstrip, int npixel);
  void fillModuleMEs(ModMEs& mod_mes, ClusterProperties& cluster);
  void fillLayerMEs(LayerMEs&, ClusterProperties& cluster);

  void ResetModuleMEs(uint32_t idet);

  inline void fillME(MonitorElement* ME, float value1) {
    if (ME != nullptr)
      ME->Fill(value1);
  }
  inline void fillME(MonitorElement* ME, float value1, float value2) {
    if (ME != nullptr)
      ME->Fill(value1, value2);
  }
  inline void fillME(MonitorElement* ME, float value1, float value2, float value3) {
    if (ME != nullptr)
      ME->Fill(value1, value2, value3);
  }
  inline void fillME(MonitorElement* ME, float value1, float value2, float value3, float value4) {
    if (ME != nullptr)
      ME->Fill(value1, value2, value3, value4);
  }
  MonitorElement* bookMETrend(const char*, DQMStore::IBooker& ibooker);
  MonitorElement* bookME1D(const char* ParameterSetLabel, const char* HistoName, DQMStore::IBooker& ibooker);
  MonitorElement* bookME2D(const char* ParameterSetLabel, const char* HistoName, DQMStore::IBooker& ibooker);

  edm::ParameterSet conf_;
  std::map<uint32_t, ModMEs> ModuleMEsMap;
  std::map<std::string, LayerMEs> LayerMEsMap;
  std::map<std::string, std::vector<uint32_t> > LayerDetMap;
  std::map<std::string, SubDetMEs> SubDetMEsMap;
  std::map<std::string, std::string> SubDetPhasePartMap;

  // flags
  bool show_mechanical_structure_view, show_readout_view, show_control_view, select_all_detectors, reset_each_run;
  unsigned long long m_cacheID_;

  std::vector<uint32_t> ModulesToBeExcluded_;

  edm::ParameterSet Parameters;

  // TkHistoMap added
  std::unique_ptr<TkHistoMap> tkmapcluster;
  std::unique_ptr<TkHistoMap> tkmapclusterch;

  int runNb, eventNb;
  int firstEvent;
  float trendVar;

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
  bool layer_clusterWidth_vs_amplitude_on;

  bool globalswitchstripnoise2apvcycle;
  bool globalswitchstripnoise3apvcycle;
  bool globalswitchmaindiagonalposition;
  bool globalswitchFEDCluster;

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
  bool module_clusterWidth_vs_amplitude_on;
  bool subdetswitchtotclusprofon;
  bool subdetswitchapvcycleprofon;
  bool subdetswitchapvcycleth2on;
  bool subdetswitchapvcycledbxprof2on;
  bool subdetswitchdbxcycleprofon;
  bool subdetswitchtotclusth1on;
  bool subdetswitchcluschargeon;
  bool subdetswitchcluswidthon;
  bool subdet_clusterWidth_vs_amplitude_on;
  bool globalswitchapvcycledbxth2on;
  bool globalswitchcstripvscpix;
  bool globalswitchMultiRegions;
  bool clustertkhistomapon;
  bool clusterchtkhistomapon;
  bool createTrendMEs;
  bool trendVs10Ls_;
  bool globalswitchnclusvscycletimeprof2don;
  bool clusterWidth_vs_amplitude_on;

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

  edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster> > clusterProducerStripToken_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > clusterProducerPixToken_;
  edm::EDGetTokenT<EventWithHistory> historyProducerToken_;
  edm::EDGetTokenT<APVCyclePhaseCollection> apvPhaseProducerToken_;

  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopologyRunToken_;
  edm::ESGetToken<TkDetMap, TrackerTopologyRcd> tkDetMapToken_;
  edm::ESGetToken<SiStripDetCabling, SiStripDetCablingRcd> siStripDetCablingRunToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopologyEventToken_;
  edm::ESGetToken<SiStripNoises, SiStripNoisesRcd> siStripNoisesToken_;
  edm::ESGetToken<SiStripGain, SiStripGainRcd> siStripGainToken_;
  edm::ESGetToken<SiStripQuality, SiStripQualityRcd> siStripQualityToken_;
  edm::ESGetToken<SiStripDetCabling, SiStripDetCablingRcd> siStripDetCablingEventToken_;

  bool applyClusterQuality_;
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
