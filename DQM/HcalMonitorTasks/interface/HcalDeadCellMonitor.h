#ifndef DQM_HCALMONITORTASKS_HCALDEADCELLMONITOR_H
#define DQM_HCALMONITORTASKS_HCALDEADCELLMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
//#include "CalibFormats/HcalObjects/interface/HcalCalibrationWidths.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include <cmath>
#include <ostream>

/** \class HcalDeadCellMonitor
  *
  * $Date: 2008/10/24 13:11:38 $
  * $Revision: 1.17 $
  * \author J. Temple - Univ. of Maryland
  */


class HcalDeadCellMonitor: public HcalBaseMonitor {

 public:
  HcalDeadCellMonitor();

  ~HcalDeadCellMonitor();

  void setup(const edm::ParameterSet& ps, DQMStore* dbe);
  //  const HcalDbService& cond);
  void done(); // overrides base class function
  void clearME(); // overrides base class function
  void reset();

  void createMaps(const HcalDbService& cond);
  
  void processEvent(const HBHERecHitCollection& hbHits,
                    const HORecHitCollection& hoHits,
                    const HFRecHitCollection& hfHits,
		    //const ZDCRecHitCollection& zdcHits,
		    const HBHEDigiCollection& hbhedigi,
                    const HODigiCollection& hodigi,
                    const HFDigiCollection& hfdigi,
		    //const ZDCDigiCollection& zdcdigi, 
		    const HcalDbService& cond
		    );

  void processEvent_digi(const HBHEDigiCollection& hbhedigi,
			 const HODigiCollection& hodigi,
			 const HFDigiCollection& hfdigi,
			 //const ZDCDigiCollection& zdcdigi, 
			 const HcalDbService& cond
			 );
 private:

  void fillNevents_occupancy();
  void fillNevents_pedestal();
  void fillNevents_neighbor();
  void fillNevents_energy();

  void fillNevents_problemCells();

  bool doFCpeds_; //specify whether pedestals are in fC (if not, assume ADC)
  bool deadmon_makeDiagnostics_;

  // Booleans to control which of the three dead cell checking routines are used
  bool deadmon_test_occupancy_;
  bool deadmon_test_pedestal_;
  bool deadmon_test_neighbor_;
  bool deadmon_test_energy_;

  int deadmon_checkNevents_;  // specify how often to check is cell is dead
  // Let each test have its own checkNevents value
  int deadmon_checkNevents_occupancy_;
  int deadmon_checkNevents_pedestal_;
  int deadmon_checkNevents_neighbor_;
  int deadmon_checkNevents_energy_;

  double HBenergyThreshold_;
  double HEenergyThreshold_;
  double HOenergyThreshold_;
  double HFenergyThreshold_;

  MonitorElement* meEVT_;
  int ievt_;

  double deadmon_minErrorFlag_; // minimum error rate needed to dump out bad bin info 
  // Problem Histograms
  MonitorElement* ProblemDeadCells;
  std::vector<MonitorElement*> ProblemDeadCellsByDepth;

  std::vector<MonitorElement*>UnoccupiedDeadCellsByDepth;
  std::vector<MonitorElement*>BelowPedestalDeadCellsByDepth;
  double nsigma_;
  double HBnsigma_, HEnsigma_, HOnsigma_, HFnsigma_;
  std::vector<MonitorElement*>BelowNeighborsDeadCellsByDepth;
  std::vector<MonitorElement*>BelowEnergyThresholdCellsByDepth;


  // map of pedestals from database (in ADC)
  std::map<HcalDetId, float> pedestals_;
  std::map<HcalDetId, float> widths_;
  std::map<HcalDetId, float> pedestal_thresholds_;
  
  unsigned int occupancy[ETABINS][PHIBINS][4]; // will get filled when an occupied digi is found
  unsigned int belowpedestal[ETABINS][PHIBINS][4]; // filled when digi is below pedestal+nsigma
  unsigned int belowneighbors[ETABINS][PHIBINS][4];
  unsigned int belowenergy[ETABINS][PHIBINS][4];
};

#endif
