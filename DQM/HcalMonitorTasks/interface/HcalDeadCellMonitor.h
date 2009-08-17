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
  * $Date: 2009/08/09 12:46:35 $
  * $Revision: 1.33 $
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
  
 private:
  void zeroCounters(bool resetpresent=false);
  void periodicReset();

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

  MonitorElement* NumberOfDeadCells;
  MonitorElement* NumberOfDeadCellsHB;
  MonitorElement* NumberOfDeadCellsHE;
  MonitorElement* NumberOfDeadCellsHO;
  MonitorElement* NumberOfDeadCellsHF;

  MonitorElement* NumberOfNeverPresentCells;
  MonitorElement* NumberOfNeverPresentCellsHB;
  MonitorElement* NumberOfNeverPresentCellsHE;
  MonitorElement* NumberOfNeverPresentCellsHO;
  MonitorElement* NumberOfNeverPresentCellsHF;

  MonitorElement* NumberOfUnoccupiedCells;
  MonitorElement* NumberOfUnoccupiedCellsHB;
  MonitorElement* NumberOfUnoccupiedCellsHE;
  MonitorElement* NumberOfUnoccupiedCellsHO;
  MonitorElement* NumberOfUnoccupiedCellsHF;

  MonitorElement* NumberOfBelowEnergyCells;
  MonitorElement* NumberOfBelowEnergyCellsHB;
  MonitorElement* NumberOfBelowEnergyCellsHE;
  MonitorElement* NumberOfBelowEnergyCellsHO;
  MonitorElement* NumberOfBelowEnergyCellsHF;

  bool present[85][72][4];
  unsigned int occupancy[85][72][4];
  unsigned int aboveenergy[85][72][4];

  bool HBpresent_, HEpresent_, HOpresent_, HFpresent_;
};

#endif
