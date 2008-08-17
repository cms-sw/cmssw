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
  * $Date:  2008/08/17 21:11:00 $
  * $Revision:  1.1 $
  * \author J. Temple - Univ. of Maryland
  */

class HcalBeamMonitor:  public HcalBaseMonitor {
 public:
  HcalBeamMonitor();
  ~HcalBeamMonitor();
  
  void setup(const edm::ParameterSet& ps, DQMStore* dbe);
  void processEvent(const  HBHERecHitCollection& hbHits,
		    const HORecHitCollection& hoHits, 
		    const HFRecHitCollection& hfHits
		    //const ZDCRecHitCollection& zdcHits
		    );
  void reset();
  void clearME();

 private:
  
  int ievt_;
  MonitorElement* meEVT_;


  std::map<int,MonitorElement* > HB_SecondMoments;
  std::map<int,MonitorElement* > HE_SecondMoments;
  std::map<int,MonitorElement* > HF_SecondMoments;
  std::map<int,MonitorElement* > HO_SecondMoments;
  MonitorElement* SecondMoment;

  const int ETA_OFFSET_HB;

  const int ETA_OFFSET_HE;
  const int ETA_BOUND_HE;
  const int ETA_OFFSET_HF;
  const int ETA_BOUND_HF;
}; // class HcalBeamMonitor

#endif  
