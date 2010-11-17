#ifndef DQM_HCALMONITORTASKS_HCALDEADCELLMONITOR_H
#define DQM_HCALMONITORTASKS_HCALDEADCELLMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseDQMonitor.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

#include <cmath>
#include <iostream>
#include <fstream>

/** \class HcalDeadCellMonitor
  *
  * $Date: 2010/11/10 20:00:33 $
  * $Revision: 1.47 $
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
  MonitorElement *ProblemsVsLB, *ProblemsVsLB_HB, *ProblemsVsLB_HE, *ProblemsVsLB_HO, *ProblemsVsLB_HF;
  MonitorElement *NumberOfNeverPresentDigis, *NumberOfNeverPresentDigisHB, *NumberOfNeverPresentDigisHE, *NumberOfNeverPresentDigisHO, *NumberOfNeverPresentDigisHF;
  MonitorElement *NumberOfRecentMissingDigis, *NumberOfRecentMissingDigisHB, *NumberOfRecentMissingDigisHE, *NumberOfRecentMissingDigisHO, *NumberOfRecentMissingDigisHF;
  MonitorElement *NumberOfRecentMissingRecHits, *NumberOfRecentMissingRecHitsHB, *NumberOfRecentMissingRecHitsHE, *NumberOfRecentMissingRecHitsHO, *NumberOfRecentMissingRecHitsHF;
  MonitorElement *NumberOfNeverPresentRecHits, *NumberOfNeverPresentRecHitsHB, *NumberOfNeverPresentRecHitsHE, *NumberOfNeverPresentRecHitsHO, *NumberOfNeverPresentRecHitsHF;

  MonitorElement *Nevents;

  MonitorElement *HBDeadVsEvent, *HEDeadVsEvent, *HODeadVsEvent, *HFDeadVsEvent;
  bool present_digi[85][72][4]; // tests that a good digi was present at least once
  bool present_rechit[85][72][4]; // tests that rechit with energy > threshold at least once
  unsigned int recentoccupancy_digi[85][72][4]; // tests that cells haven't gone missing for long periods
  unsigned int recentoccupancy_rechit[85][72][4]; // tests that cells haven't dropped below threshold for long periods
  
  int deadevt_; // running count of events processed since last dead cell check
  int NumBadHB, NumBadHE, NumBadHO, NumBadHF, NumBadHFLUMI, NumBadHO0, NumBadHO12;
  edm::InputTag digiLabel_;
  edm::InputTag hbheRechitLabel_, hoRechitLabel_, hfRechitLabel_;

  bool endLumiProcessed_;

  bool excludeHORing2_;
};

#endif
