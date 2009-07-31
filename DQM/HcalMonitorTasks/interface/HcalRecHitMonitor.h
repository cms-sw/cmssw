#ifndef DQM_HCALMONITORTASKS_HCALRECHITMONITOR_H
#define DQM_HCALMONITORTASKS_HCALRECHITMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "CondFormats/HcalObjects/interface/HcalChannelStatus.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"

#include <cmath>
#include <iostream>
#include <fstream>

/** \class HcalRecHitMonitor
  *
  * $Date: 2009/07/27 19:21:49 $
  * $Revision: 1.28 $
  * \author J. Temple - Univ. of Maryland
  */


class HcalRecHitMonitor: public HcalBaseMonitor {

 public:
  HcalRecHitMonitor();

  ~HcalRecHitMonitor();

  void setup(const edm::ParameterSet& ps, DQMStore* dbe);
  void done();
  void clearME(); // overrides base class function
  void reset();
  void zeroCounters();
 
  void processEvent(const HBHERecHitCollection& hbHits,
                    const HORecHitCollection& hoHits,
                    const HFRecHitCollection& hfHits
		    //const ZDCRecHitCollection& zdcHits,
		    );

  void processEvent_rechit( const HBHERecHitCollection& hbheHits,
			    const HORecHitCollection& hoHits,
			    const HFRecHitCollection& hfHits);

  void fillRecHitHistosAtEndRun();
 private:


  void fillNevents();

  bool rechit_makeDiagnostics_;

  int rechit_checkNevents_;  // specify how often to fill histograms

  double energyThreshold_;
  double HBenergyThreshold_;
  double HEenergyThreshold_;
  double HOenergyThreshold_;
  double HFenergyThreshold_;
  double ZDCenergyThreshold_;

  double rechit_minErrorFlag_; // minimum error rate needed to dump out bad bin info 

  // Basic Histograms
  EtaPhiHists OccupancyByDepth;
  EtaPhiHists OccupancyThreshByDepth;
  EtaPhiHists EnergyByDepth;
  EtaPhiHists EnergyThreshByDepth;
  EtaPhiHists TimeByDepth;
  EtaPhiHists TimeThreshByDepth;
  
  //EtaPhiHists SumOccupancyByDepth;
  //EtaPhiHists SumOccupancyThreshByDepth;
  EtaPhiHists SumEnergyByDepth;
  EtaPhiHists SumEnergyThreshByDepth;
  EtaPhiHists SumTimeByDepth;
  EtaPhiHists SumTimeThreshByDepth;


  unsigned int occupancy_[85][72][4]; // will get filled when rechit found
  unsigned int occupancy_thresh_[85][72][4]; // filled when above given energy
  double energy_[85][72][4]; // will get filled when rechit found
  double energy2_[85][72][4]; // will get filled when rechit found
  double energy_thresh_[85][72][4]; // filled when above given  
  double time_[85][72][4]; // will get filled when rechit found
  double time_thresh_[85][72][4]; // filled when above given energy

  double HBenergy_[200];
  double HBenergy_thresh_[200];
  double HBtime_[300];
  double HBtime_thresh_[300];
  double HB_occupancy_[2593];
  double HB_occupancy_thresh_[2593];
  double HEenergy_[200];
  double HEenergy_thresh_[200];
  double HEtime_[300];
  double HEtime_thresh_[300];
  double HE_occupancy_[2593];
  double HE_occupancy_thresh_[2593];
  double HOenergy_[200];
  double HOenergy_thresh_[200];
  double HOtime_[300];
  double HOtime_thresh_[300];
  double HO_occupancy_[2161];
  double HO_occupancy_thresh_[2161];
  double HFenergy_[200];
  double HFenergy_thresh_[200];
  double HFtime_[300];
  double HFtime_thresh_[300];
  double HFenergyLong_[200];
  double HFenergyLong_thresh_[200];
  double HFtimeLong_[300];
  double HFtimeLong_thresh_[300];
  double HFenergyShort_[200];
  double HFenergyShort_thresh_[200];
  double HFtimeShort_[300];
  double HFtimeShort_thresh_[300];
  double HF_occupancy_[1729];
  double HF_occupancy_thresh_[1729];
  double HFlong_occupancy_[865];
  double HFlong_occupancy_thresh_[865];
  double HFshort_occupancy_[865];
  double HFshort_occupancy_thresh_[865];

  // Diagnostic plots
  MonitorElement* h_HBEnergy;
  MonitorElement* h_HBThreshEnergy;
  MonitorElement* h_HBTotalEnergy;
  MonitorElement* h_HBThreshTotalEnergy;
  MonitorElement* h_HBTime;
  MonitorElement* h_HBThreshTime;
  MonitorElement* h_HBOccupancy;
  MonitorElement* h_HBThreshOccupancy;

  MonitorElement* h_HEEnergy;
  MonitorElement* h_HEThreshEnergy;
  MonitorElement* h_HETotalEnergy;
  MonitorElement* h_HEThreshTotalEnergy;
  MonitorElement* h_HETime;
  MonitorElement* h_HEThreshTime;
  MonitorElement* h_HEOccupancy;
  MonitorElement* h_HEThreshOccupancy;

  MonitorElement* h_HOEnergy;
  MonitorElement* h_HOThreshEnergy;
  MonitorElement* h_HOTotalEnergy;
  MonitorElement* h_HOThreshTotalEnergy;
  MonitorElement* h_HOTime;
  MonitorElement* h_HOThreshTime;
  MonitorElement* h_HOOccupancy;
  MonitorElement* h_HOThreshOccupancy;

  MonitorElement* h_HFEnergy;
  MonitorElement* h_HFThreshEnergy;
  MonitorElement* h_HFTotalEnergy;
  MonitorElement* h_HFThreshTotalEnergy;
  MonitorElement* h_HFTime;
  MonitorElement* h_HFThreshTime;
  MonitorElement* h_HFOccupancy;
  MonitorElement* h_HFThreshOccupancy;

  MonitorElement* h_HBEnergy_1D;
  MonitorElement* h_HEEnergy_1D;
  MonitorElement* h_HOEnergy_1D;
  MonitorElement* h_HFEnergy_1D;

  MonitorElement* h_HBEnergyRMS_1D;
  MonitorElement* h_HEEnergyRMS_1D;
  MonitorElement* h_HOEnergyRMS_1D;
  MonitorElement* h_HFEnergyRMS_1D;

  bool HBpresent_, HEpresent_, HOpresent_, HFpresent_;
};

#endif
