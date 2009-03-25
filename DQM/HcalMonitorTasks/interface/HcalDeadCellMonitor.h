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
  * $Date: 2009/01/08 19:34:07 $
  * $Revision: 1.24 $
  * \author J. Temple - Univ. of Maryland
  */

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
  void setupNeighborParams(const edm::ParameterSet& ps, neighborParams& N, char* type);
  void done(std::map<HcalDetId, unsigned int>& myqual); // overrides base class function
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

  void processEvent_rechitenergy( const HBHERecHitCollection& hbheHits,
				  const HORecHitCollection& hoHits,
				  const HFRecHitCollection& hfHits);

  void processEvent_rechitneighbors( const HBHERecHitCollection& hbheHits,
				     const HORecHitCollection& hoHits,
				     const HFRecHitCollection& hfHits);
  void fillDeadHistosAtEndRun();
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
  bool deadmon_test_rechit_occupancy_;

  int deadmon_checkNevents_;  // specify how often to check is cell is dead
  // Let each test have its own checkNevents value
  int deadmon_checkNevents_occupancy_;
  int deadmon_checkNevents_pedestal_;
  int deadmon_checkNevents_neighbor_;
  int deadmon_checkNevents_energy_;
  int deadmon_checkNevents_rechit_occupancy_;

  double energyThreshold_;
  double HBenergyThreshold_;
  double HEenergyThreshold_;
  double HOenergyThreshold_;
  double HFenergyThreshold_;
  double ZDCenergyThreshold_;

  MonitorElement* meEVT_;
  int ievt_;

  double deadmon_minErrorFlag_; // minimum error rate needed to dump out bad bin info 
  // Problem Histograms
  MonitorElement* ProblemDeadCells;
  std::vector<MonitorElement*> ProblemDeadCellsByDepth;

  std::vector<MonitorElement*>UnoccupiedDeadCellsByDepth;
  std::vector<MonitorElement*>UnoccupiedRecHitsByDepth;
  std::vector<MonitorElement*>BelowPedestalDeadCellsByDepth;
  double nsigma_;
  double HBnsigma_, HEnsigma_, HOnsigma_, HFnsigma_, ZDCnsigma_;
  std::vector<MonitorElement*>BelowNeighborsDeadCellsByDepth;
  std::vector<MonitorElement*>BelowEnergyThresholdCellsByDepth;


  // map of pedestals from database (in ADC)
  std::map<HcalDetId, float> pedestals_;
  std::map<HcalDetId, float> widths_;
  std::map<HcalDetId, float> pedestal_thresholds_;
  std::map<HcalDetId, double> rechitEnergies_;
  

  unsigned int occupancy[ETABINS][PHIBINS][6]; // will get filled when an occupied digi is found
  unsigned int rechit_occupancy[ETABINS][PHIBINS][6]; // filled when rechit is present
  unsigned int abovepedestal[ETABINS][PHIBINS][6]; // filled when digi is below pedestal+nsigma
  unsigned int belowneighbors[ETABINS][PHIBINS][6];
  unsigned int aboveenergy[ETABINS][PHIBINS][6];

  // Diagnostic plots
  MonitorElement* d_HBnormped;
  MonitorElement* d_HEnormped;
  MonitorElement* d_HOnormped;
  MonitorElement* d_HFnormped;
  MonitorElement* d_ZDCnormped;

  MonitorElement* d_HBrechitenergy;
  MonitorElement* d_HErechitenergy;
  MonitorElement* d_HOrechitenergy;
  MonitorElement* d_HFrechitenergy;
  MonitorElement* d_ZDCrechitenergy;

  MonitorElement* d_HBenergyVsNeighbor;
  MonitorElement* d_HEenergyVsNeighbor;
  MonitorElement* d_HOenergyVsNeighbor;
  MonitorElement* d_HFenergyVsNeighbor;
  MonitorElement* d_ZDCenergyVsNeighbor;

  bool HBpresent_, HEpresent_, HOpresent_, HFpresent_;

  neighborParams defaultNeighborParams_, HBNeighborParams_, HENeighborParams_, HONeighborParams_, HFNeighborParams_, ZDCNeighborParams_;
};

#endif
