#ifndef GUARD_DQM_HCALMONITORTASKS_HCALBEAMMONITOR_H
#define GUARD_DQM_HCALMONITORTASKS_HCALBEAMMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

// Use for stringstream
#include <iostream>
#include <iomanip>
#include <cmath>

/** \class HcalBeamMonitor
  *
  * $Date: 2009/06/28 21:01:17 $
  * $Revision: 1.4.2.2 $
  * \author J. Temple - Univ. of Maryland
  */

class HcalBeamMonitor:  public HcalBaseMonitor {
 public:
  HcalBeamMonitor();
  ~HcalBeamMonitor();
  
  void setup(const edm::ParameterSet& ps, DQMStore* dbe);
  void processEvent(const  HBHERecHitCollection& hbHits,
		    const HORecHitCollection& hoHits, 
		    const HFRecHitCollection& hfHits,
                     const HFDigiCollection& hf
		    //const ZDCRecHitCollection& zdcHits
		    );
  void reset();
  void clearME();

 private:
  float occThresh_;  
  int ievt_;
  MonitorElement* meEVT_;

  bool     beammon_makeDiagnostics_;
  int      beammon_checkNevents_;
  double   beammon_minErrorFlag_;
  std::map<int,MonitorElement* > HB_CenterOfEnergyRadius;
  std::map<int,MonitorElement* > HE_CenterOfEnergyRadius;
  std::map<int,MonitorElement* > HF_CenterOfEnergyRadius;
  std::map<int,MonitorElement* > HO_CenterOfEnergyRadius;


  MonitorElement* ProblemBeamCells;
  std::vector<MonitorElement*> ProblemBeamCellsByDepth;

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

  const int ETA_OFFSET_HB;

  const int ETA_OFFSET_HE;
  const int ETA_BOUND_HE;
  
  const int ETA_OFFSET_HO;
  
  const int ETA_OFFSET_HF;
  const int ETA_BOUND_HF;

  static const float etaBounds[];
  static const float area[];
  static const float radius[];

}; // class HcalBeamMonitor

#endif  
