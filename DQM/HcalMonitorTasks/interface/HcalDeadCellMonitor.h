#ifndef DQM_HCALMONITORTASKS_HCALDEADCELLMONITOR_H
#define DQM_HCALMONITORTASKS_HCALDEADCELLMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseDQMonitor.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"

// Channel status DB stuff

#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"

#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

#include <cmath>
#include <iostream>
#include <fstream>

/** \class HcalDeadCellMonitor
  *
  * $Date: 2012/06/27 13:20:28 $
  * $Revision: 1.61 $
  * \author J. Temple - Univ. of Maryland
  */

class HcalDeadCellMonitor: public HcalBaseDQMonitor {

 public:
  HcalDeadCellMonitor(const edm::ParameterSet& ps);

  ~HcalDeadCellMonitor();

  void setup();
  void beginRun(const edm::Run& run, const edm::EventSetup& c);
  void analyze(edm::Event const&e, edm::EventSetup const&s);
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
			  const edm::EventSetup& c);
  void endRun(const edm::Run& run, const edm::EventSetup& c);
  void endJob();
  void cleanup(); // overrides base class function
  void reset();

  void processEvent(const HBHERecHitCollection& hbHits,
                    const HORecHitCollection& hoHits,
                    const HFRecHitCollection& hfHits,
		    const HBHEDigiCollection& hbhedigi,
                    const HODigiCollection& hodigi,
                    const HFDigiCollection& hfdigi
		    );

 private:
  void zeroCounters(bool resetpresent=false);

  void processEvent_HBHEdigi(const HBHEDataFrame digi);
  template<class T> void process_Digi(T& digi);
  template<class T> void process_RecHit(T& rechit);

  bool deadmon_makeDiagnostics_;
  int minDeadEventCount_;

  // Booleans to control which of the dead cell checking routines are used
  bool deadmon_test_digis_;
  bool deadmon_test_rechits_;

  void fillNevents_problemCells(); // problemcells always checks for never-present digis, rechits
  void fillNevents_recentdigis();
  void fillNevents_recentrechits();

  // specify minimum energy threshold for energy test
  double energyThreshold_;
  double HBenergyThreshold_;
  double HEenergyThreshold_;
  double HOenergyThreshold_;
  double HFenergyThreshold_;

  EtaPhiHists  RecentMissingDigisByDepth;
  EtaPhiHists  DigiPresentByDepth;
  EtaPhiHists  RecentMissingRecHitsByDepth;
  EtaPhiHists  RecHitPresentByDepth;

  // Problems vs. lumi block
  MonitorElement *ProblemsVsLB, *ProblemsVsLB_HB, *ProblemsVsLB_HE, *ProblemsVsLB_HO, *ProblemsVsLB_HO2, *ProblemsVsLB_HF;
  MonitorElement *RBX_loss_VS_LB;
  MonitorElement *ProblemsInLastNLB_HBHEHF_alarm;
  MonitorElement *ProblemsInLastNLB_HO01_alarm;
  MonitorElement *NumberOfNeverPresentDigis, *NumberOfNeverPresentDigisHB, *NumberOfNeverPresentDigisHE, *NumberOfNeverPresentDigisHO, *NumberOfNeverPresentDigisHF;
  MonitorElement *NumberOfRecentMissingDigis, *NumberOfRecentMissingDigisHB, *NumberOfRecentMissingDigisHE, *NumberOfRecentMissingDigisHO, *NumberOfRecentMissingDigisHF;
  MonitorElement *NumberOfRecentMissingRecHits, *NumberOfRecentMissingRecHitsHB, *NumberOfRecentMissingRecHitsHE, *NumberOfRecentMissingRecHitsHO, *NumberOfRecentMissingRecHitsHF;
  MonitorElement *NumberOfNeverPresentRecHits, *NumberOfNeverPresentRecHitsHB, *NumberOfNeverPresentRecHitsHE, *NumberOfNeverPresentRecHitsHO, *NumberOfNeverPresentRecHitsHF;

  MonitorElement *Nevents;
  int beamMode_;
  bool doReset_;

  MonitorElement *HBDeadVsEvent, *HEDeadVsEvent, *HODeadVsEvent, *HFDeadVsEvent;
  bool present_digi[85][72][4]; // tests that a good digi was present at least once
  bool present_rechit[85][72][4]; // tests that rechit with energy > threshold at least once
  unsigned int recentoccupancy_digi[85][72][4]; // tests that cells haven't gone missing for long periods
  unsigned int recentoccupancy_rechit[85][72][4]; // tests that cells haven't dropped below threshold for long periods
  unsigned int occupancy_RBX[156];

  int deadevt_; // running count of events processed since last dead cell check
  int is_RBX_loss_;
  int rbxlost[156];
  int alarmer_counter_;
  int alarmer_counterHO01_;
  bool is_stable_beam;
  bool hbhedcsON, hfdcsON, hodcsON;
  unsigned int NumBadHB, NumBadHE, NumBadHO, NumBadHO01, NumBadHO2, NumBadHF, NumBadHFLUMI, NumBadHO0, NumBadHO12;
  edm::InputTag digiLabel_;
  edm::InputTag hbheRechitLabel_, hoRechitLabel_, hfRechitLabel_;

  bool endLumiProcessed_;

  bool excludeHORing2_;
  bool excludeHO1P02_;
  bool setupDone_;
  int NumBadHO1P02;
};

#endif
