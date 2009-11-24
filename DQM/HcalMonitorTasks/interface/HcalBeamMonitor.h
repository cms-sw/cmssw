#ifndef GUARD_DQM_HCALMONITORTASKS_HCALBEAMMONITOR_H
#define GUARD_DQM_HCALMONITORTASKS_HCALBEAMMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "FWCore/Framework/interface/ESHandle.h"

// Use for stringstream
#include <iostream>
#include <fstream>

#include <iomanip>
#include <cmath>

/** \class HcalBeamMonitor
  *
  * $Date: 2009/11/19 12:49:22 $
  * $Revision: 1.15 $
  * \author J. Temple - Univ. of Maryland
  */

class HcalBeamMonitor:  public HcalBaseMonitor {
 public:
  HcalBeamMonitor();
  ~HcalBeamMonitor();
  
  void setup(const edm::ParameterSet& ps, DQMStore* dbe);
  void beginRun(const edm::EventSetup& c, int run);
  void processEvent(const  HBHERecHitCollection& hbHits,
		    const  HORecHitCollection& hoHits, 
		    const  HFRecHitCollection& hfHits,
		    const  HFDigiCollection& hf,
		    int    CalibType,
		    int    bunchCrossing
		    );
  void reset();
  void clearME();
  void beginLuminosityBlock(int lb);
  void endLuminosityBlock();

 private:
  void SetEtaLabels(MonitorElement* h);
  float occThresh_;  
  int ievt_;
  MonitorElement* meEVT_;

  bool     beammon_makeDiagnostics_;
  int      beammon_checkNevents_;
  double   beammon_minErrorFlag_;
  int      beammon_minEvents_;
  int      beammon_BX;
  std::map<int,MonitorElement* > HB_CenterOfEnergyRadius;
  std::map<int,MonitorElement* > HE_CenterOfEnergyRadius;
  std::map<int,MonitorElement* > HF_CenterOfEnergyRadius;
  std::map<int,MonitorElement* > HO_CenterOfEnergyRadius;

  MonitorElement* CenterOfEnergyRadius;
  MonitorElement* CenterOfEnergy;
  MonitorElement* COEradiusVSeta;

  MonitorElement* HBCenterOfEnergyRadius;
  MonitorElement* HBCenterOfEnergy;
  MonitorElement* HECenterOfEnergyRadius;
  MonitorElement* HECenterOfEnergy;
  MonitorElement* HOCenterOfEnergyRadius;
  MonitorElement* HOCenterOfEnergy;
  MonitorElement* HFCenterOfEnergyRadius;
  MonitorElement* HFCenterOfEnergy;

  MonitorElement* Etsum_eta_L;
  MonitorElement* Etsum_eta_S;
  MonitorElement* Etsum_phi_L;
  MonitorElement* Etsum_phi_S;
  MonitorElement* Etsum_ratio_p;
  MonitorElement* Etsum_ratio_m;
  MonitorElement* Etsum_map_L;
  MonitorElement* Etsum_map_S;
  MonitorElement* Etsum_ratio_map;
  MonitorElement* Etsum_rphi_L;
  MonitorElement* Etsum_rphi_S;
  MonitorElement* Energy_Occ;

  MonitorElement* Occ_rphi_L;
  MonitorElement* Occ_rphi_S;
  MonitorElement* Occ_eta_L;
  MonitorElement* Occ_eta_S;
  MonitorElement* Occ_phi_L;
  MonitorElement* Occ_phi_S;
  MonitorElement* Occ_map_L;
  MonitorElement* Occ_map_S;
  
  MonitorElement* HFlumi_ETsum_perwedge;
  MonitorElement* HFlumi_Occupancy_above_thr_r1;
  MonitorElement* HFlumi_Occupancy_between_thrs_r1;
  MonitorElement* HFlumi_Occupancy_below_thr_r1;
  MonitorElement* HFlumi_Occupancy_above_thr_r2;
  MonitorElement* HFlumi_Occupancy_between_thrs_r2;
  MonitorElement* HFlumi_Occupancy_below_thr_r2;

  MonitorElement* HFlumi_Occupancy_per_channel_vs_lumiblock_RING1;
  MonitorElement* HFlumi_Occupancy_per_channel_vs_lumiblock_RING2;
  MonitorElement* HFlumi_Occupancy_per_channel_vs_BX_RING1;
  MonitorElement* HFlumi_Occupancy_per_channel_vs_BX_RING2;
  MonitorElement* HFlumi_ETsum_vs_BX;
  MonitorElement* HFlumi_Et_per_channel_vs_lumiblock;

  MonitorElement* HFlumi_occ_LS;
  MonitorElement* HFlumi_total_hotcells;
  MonitorElement* HFlumi_total_deadcells;

  MonitorElement* HFlumi_Ring1Status_vs_LS;
  MonitorElement* HFlumi_Ring2Status_vs_LS;
  std::map <HcalDetId, int> BadCells_;

  int ring1totalchannels_;
  int ring2totalchannels_;
  const int ETA_OFFSET_HB;

  const int ETA_OFFSET_HE;
  const int ETA_BOUND_HE;
  
  const int ETA_OFFSET_HO;
  
  const int ETA_OFFSET_HF;
  const int ETA_BOUND_HF;

  static const float etaBounds[];
  static const float area[];
  static const float radius[];

  std::string beammon_lumiqualitydir_;
  std::ostringstream outfile_;
  int irun_;
}; // class HcalBeamMonitor

#endif  
