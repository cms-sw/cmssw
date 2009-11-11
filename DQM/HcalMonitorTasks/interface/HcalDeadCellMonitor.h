#ifndef DQM_HCALMONITORTASKS_HCALDEADCELLMONITOR_H
#define DQM_HCALMONITORTASKS_HCALDEADCELLMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
//#include "CalibFormats/HcalObjects/interface/HcalCalibrationWidths.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"

#include <cmath>
#include <iostream>
#include <fstream>

/** \class HcalDeadCellMonitor
  *
  * $Date: 2009/11/10 21:03:13 $
  * $Revision: 1.38 $
  * \author J. Temple - Univ. of Maryland
  */

class HcalDeadCellMonitor: public HcalBaseMonitor {

 public:
  HcalDeadCellMonitor();

  ~HcalDeadCellMonitor();

  void setup(const edm::ParameterSet& ps, DQMStore* dbe);
  void beginRun();
  void clearME(); // overrides base class function
  void reset();
  
  void processEvent(const HBHERecHitCollection& hbHits,
                    const HORecHitCollection& hoHits,
                    const HFRecHitCollection& hfHits,
		    //const ZDCRecHitCollection& zdcHits,
		    const HBHEDigiCollection& hbhedigi,
                    const HODigiCollection& hodigi,
                    const HFDigiCollection& hfdigi,
		    int calibType
		    );

  void periodicReset();
  void beginLuminosityBlock(int lb);
  void endLuminosityBlock();

 private:
  void zeroCounters(bool resetpresent=false);

  void processEvent_HBHEdigi(const HBHEDataFrame digi);
  template<class T> void process_Digi(T& digi);
  template<class T> void process_RecHit(T& rechit);

  int deadmon_checkNevents_;  // specify how often to check is cell is dead
  int deadmon_minEvents_; // minimum number of events needed to perform checks on recent digis/rechits
  bool deadmon_makeDiagnostics_;

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

  double deadmon_minErrorFlag_; // minimum error rate needed to dump out bad bin info 

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

  bool present_digi[85][72][4]; // tests that a good digi was present at least once
  bool present_rechit[85][72][4]; // tests that rechit with energy > threshold at least once
  unsigned int recentoccupancy_digi[85][72][4]; // tests that cells haven't gone missing for long periods
  unsigned int recentoccupancy_rechit[85][72][4]; // tests that cells haven't dropped below threshold for long periods
  

  bool HBpresent_, HEpresent_, HOpresent_, HFpresent_;
};

#endif
