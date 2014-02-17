#ifndef GUARD_DQM_HCALMONITORTASKS_HCALBEAMMONITOR_H
#define GUARD_DQM_HCALMONITORTASKS_HCALBEAMMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseDQMonitor.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"
#include "Geometry/HcalTowerAlgo/src/HcalHardcodeGeometryData.h"

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

// Use for stringstream
#include <iostream>
#include <fstream>

/** \class HcalBeamMonitor
  *
  * $Date: 2012/06/27 13:20:28 $
  * $Revision: 1.19 $
  * \author J. Temple - Univ. of Maryland
  */

class HcalBeamMonitor:  public HcalBaseDQMonitor {
 public:
  HcalBeamMonitor(const edm::ParameterSet& ps);
  ~HcalBeamMonitor();
  
  void setup();
  void beginRun(const edm::Run& run, const edm::EventSetup& c);
  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
			    const edm::EventSetup& c);
  void analyze(const edm::Event& e, const edm::EventSetup& c);
  void processEvent(const  HBHERecHitCollection& hbHits,
		    const  HORecHitCollection& hoHits, 
		    const  HFRecHitCollection& hfHits,
		    const  HFDigiCollection& hf,
		    int    bunchCrossing
		    );

  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
			  const edm::EventSetup& c);
  void reset();
  void cleanup();

 private:
  void SetEtaLabels(MonitorElement* h);
  double occThresh_;  
  double hotrate_;
  int minEvents_;
  std::string lumiqualitydir_;

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
  MonitorElement* HFlumi_diag_hotcells;
  MonitorElement* HFlumi_diag_deadcells;

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

  static const float area[];
  static const float radius[];

  std::ostringstream outfile_;
  unsigned int lastProcessedLS_;
  int runNumber_;
  bool Overwrite_;
  bool setupDone_;

  int minBadCells_;  // number of channels that must be bad to be included in problem summary
  edm::InputTag digiLabel_;
  edm::InputTag hbheRechitLabel_, hfRechitLabel_, hoRechitLabel_;
}; // class HcalBeamMonitor

#endif  
