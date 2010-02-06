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
  * $Date: 2009/08/21 20:45:29 $
  * $Revision: 1.36 $
  * \author J. Temple - Univ. of Maryland
  */

class HcalDeadCellMonitor: public HcalBaseMonitor {

 public:
  HcalDeadCellMonitor();

  ~HcalDeadCellMonitor();

  void setup(const edm::ParameterSet& ps, DQMStore* dbe);
  void clearME(); // overrides base class function
  void reset();
  
  void processEvent(const HBHERecHitCollection& hbHits,
                    const HORecHitCollection& hoHits,
                    const HFRecHitCollection& hfHits,
		    //const ZDCRecHitCollection& zdcHits,
		    const HBHEDigiCollection& hbhedigi,
                    const HODigiCollection& hodigi,
                    const HFDigiCollection& hfdigi
		    //const ZDCDigiCollection& zdcdigi, 
		    );


  void fillDeadHistosAtEndRun();
  void periodicReset();

 private:
  void zeroCounters(bool resetpresent=false);

  void processEvent_HBHEdigi(const HBHEDataFrame digi);
  template<class T> void process_Digi(T& digi);
  template<class T> void process_RecHit(T& rechit);

  void fillNevents_occupancy(int checkN);
  void fillNevents_energy(int checkN);

  void fillNevents_problemCells(int checkN);

  int deadmon_checkNevents_;  // specify how often to check is cell is dead
  int deadmon_prescale_;
  bool deadmon_makeDiagnostics_;

  // Booleans to control which of the three dead cell checking routines are used
  bool deadmon_test_occupancy_;
  bool deadmon_test_energy_;

  // specify minimum energy threshold for energy test
  double energyThreshold_;
  double HBenergyThreshold_;
  double HEenergyThreshold_;
  double HOenergyThreshold_;
  double HFenergyThreshold_;

  double deadmon_minErrorFlag_; // minimum error rate needed to dump out bad bin info 

  EtaPhiHists  UnoccupiedDeadCellsByDepth;
  EtaPhiHists  DigiPresentByDepth;
  EtaPhiHists  BelowEnergyThresholdCellsByDepth;
  EtaPhiHists  EnergyPresentByDepth;

  // Problems vs. lumi block
  MonitorElement *ProblemsVsLB, *ProblemsVsLB_HB, *ProblemsVsLB_HE, *ProblemsVsLB_HO, *ProblemsVsLB_HF;
  MonitorElement *NumberOfNeverPresentCells, *NumberOfNeverPresentCellsHB, *NumberOfNeverPresentCellsHE, *NumberOfNeverPresentCellsHO, *NumberOfNeverPresentCellsHF;
  MonitorElement *NumberOfUnoccupiedCells, *NumberOfUnoccupiedCellsHB, *NumberOfUnoccupiedCellsHE, *NumberOfUnoccupiedCellsHO, *NumberOfUnoccupiedCellsHF;
  MonitorElement *NumberOfBelowEnergyCells, *NumberOfBelowEnergyCellsHB, *NumberOfBelowEnergyCellsHE, *NumberOfBelowEnergyCellsHO, *NumberOfBelowEnergyCellsHF;
  MonitorElement *NumberOfEnergyNeverPresentCells, *NumberOfEnergyNeverPresentCellsHB, *NumberOfEnergyNeverPresentCellsHE, *NumberOfEnergyNeverPresentCellsHO, *NumberOfEnergyNeverPresentCellsHF;

  bool present[85][72][4]; // tests that a good digi was present at least once
  bool present_energy[85][72][4]; // tests that energy > threshold at least once
  unsigned int occupancy[85][72][4]; // tests that cells haven't gone missing for long periods
  unsigned int aboveenergy[85][72][4]; // tests that cells haven't dropped below threshold for long periods
  

  bool HBpresent_, HEpresent_, HOpresent_, HFpresent_;
};

#endif
