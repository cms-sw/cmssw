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
  * $Date: 2009/07/21 11:02:48 $
  * $Revision: 1.31 $
  * \author J. Temple - Univ. of Maryland
  */

// neighboring-cell test not currently used
struct neighborParams{
  int DeltaIphi;
  int DeltaIeta;
  int DeltaDepth;
  double maxCellEnergy; // cells above this threshold can never be considered "dead" by this algorithm
  double minNeighborEnergy; //neighbors must have some amount of energy to be counted
  double minGoodNeighborFrac; // fraction of neighbors with good energy must be above this value
  double maxEnergyFrac; // cell energy/(neighbors); must be less than maxEnergyFrac for cell to be dead
};

class HcalDeadCellMonitor: public HcalBaseMonitor {

 public:
  HcalDeadCellMonitor();

  ~HcalDeadCellMonitor();

  void setup(const edm::ParameterSet& ps, DQMStore* dbe);
  void setupNeighborParams(const edm::ParameterSet& ps, neighborParams& N, std::string type);
  void done(std::map<HcalDetId, unsigned int>& myqual); // overrides base class function
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

  void processEvent_HBHEdigi(HBHEDigiCollection::const_iterator j);
  void processEvent_HOdigi(HODigiCollection::const_iterator j);
  void processEvent_HFdigi(HFDigiCollection::const_iterator j);
  void processEvent_ZDCdigi(ZDCDigiCollection::const_iterator j);
  void processEvent_HBHERecHit(HBHERecHitCollection::const_iterator j);
  void processEvent_HORecHit(HORecHitCollection::const_iterator j);
  void processEvent_HFRecHit(HFRecHitCollection::const_iterator j);
  void processEvent_ZDCRecHit(ZDCRecHitCollection::const_iterator j);

  void fillNevents_neverpresent();
  void fillNevents_occupancy();
  void fillNevents_energy();

  void fillNevents_problemCells();

  int deadmon_checkNevents_;  // specify how often to check is cell is dead
  int deadmon_neverpresent_prescale_;
  bool deadmon_makeDiagnostics_;

  // Booleans to control which of the three dead cell checking routines are used
  bool deadmon_test_neverpresent_;
  bool deadmon_test_occupancy_;
  bool deadmon_test_energy_;

  // specify minimum energy threshold for energy test
  double energyThreshold_;
  double HBenergyThreshold_;
  double HEenergyThreshold_;
  double HOenergyThreshold_;
  double HFenergyThreshold_;
  double ZDCenergyThreshold_;

  double deadmon_minErrorFlag_; // minimum error rate needed to dump out bad bin info 

  EtaPhiHists  UnoccupiedDeadCellsByDepth;
  EtaPhiHists  DigisNeverPresentByDepth;
  EtaPhiHists  BelowEnergyThresholdCellsByDepth;

  MonitorElement* NumberOfDeadCells;
  MonitorElement* NumberOfDeadCellsHB;
  MonitorElement* NumberOfDeadCellsHE;
  MonitorElement* NumberOfDeadCellsHO;
  MonitorElement* NumberOfDeadCellsHF;
  MonitorElement* NumberOfDeadCellsZDC;

  MonitorElement* NumberOfNeverPresentCells;
  MonitorElement* NumberOfNeverPresentCellsHB;
  MonitorElement* NumberOfNeverPresentCellsHE;
  MonitorElement* NumberOfNeverPresentCellsHO;
  MonitorElement* NumberOfNeverPresentCellsHF;
  MonitorElement* NumberOfNeverPresentCellsZDC;

  MonitorElement* NumberOfUnoccupiedCells;
  MonitorElement* NumberOfUnoccupiedCellsHB;
  MonitorElement* NumberOfUnoccupiedCellsHE;
  MonitorElement* NumberOfUnoccupiedCellsHO;
  MonitorElement* NumberOfUnoccupiedCellsHF;
  MonitorElement* NumberOfUnoccupiedCellsZDC;

  MonitorElement* NumberOfBelowEnergyCells;
  MonitorElement* NumberOfBelowEnergyCellsHB;
  MonitorElement* NumberOfBelowEnergyCellsHE;
  MonitorElement* NumberOfBelowEnergyCellsHO;
  MonitorElement* NumberOfBelowEnergyCellsHF;
  MonitorElement* NumberOfBelowEnergyCellsZDC;

  bool present[85][72][4];
  unsigned int occupancy[85][72][4];
  unsigned int aboveenergy[85][72][4];

  bool HBpresent_, HEpresent_, HOpresent_, HFpresent_, ZDCpresent_;

  //neighborParams defaultNeighborParams_, HBNeighborParams_, HENeighborParams_, HONeighborParams_, HFNeighborParams_, ZDCNeighborParams_;
};

#endif
