#ifndef DQM_HCALMONITORTASKS_HCALDEADCELLMONITOR_H
#define DQM_HCALMONITORTASKS_HCALDEADCELLMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrationWidths.h"

/** \class HcalDeadCellMonitor
  *  
  * $Date: 2007/11/03 22:58:23 $
  * $Revision: 1.1 $
  * \author J. Temple - Univ. of Maryland
  */

struct DeadCellHists{
  int type; // store subdetector type (0=hb, 1=he, 2=ho, 3=hf)
  MonitorElement* deadADC_map;
  MonitorElement* noADC_ID_map;
  MonitorElement* deadADC_eta;
  MonitorElement* noADC_ID_eta;
  MonitorElement* ADCdist;
  MonitorElement* NADA_cool_cell_map;
  MonitorElement* digiCheck;
  MonitorElement* cellCheck;

  MonitorElement* above_pedestal;
  MonitorElement* coolcell_below_pedestal;
  TH2F* above_pedestal_temp;

  std::vector<MonitorElement*> deadcapADC_map;
};


class HcalDeadCellMonitor: public HcalBaseMonitor {
 public:
  HcalDeadCellMonitor(); 
  ~HcalDeadCellMonitor(); 

  void setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe);

  void processEvent(const HBHERecHitCollection& hbHits, 
		    const HORecHitCollection& hoHits, 
		    const HFRecHitCollection& hfHits,
		    const HBHEDigiCollection& hbhedigi,
		    const HODigiCollection& hodigi,
		    const HFDigiCollection& hfdigi,
		    const HcalDbService& cond);
    
  void processEvent_digi(const HBHEDigiCollection& hbhedigi,
			 const HODigiCollection& hodigi,
			 const HFDigiCollection& hfdigi,
			 const HcalDbService& cond);

  void processEvent_hits(const HBHERecHitCollection& hbHits, 
			 const HORecHitCollection& hoHits, 
			 const HFRecHitCollection& hfHits);

  void reset_Nevents(DeadCellHists& h);
  void reset();


 private:  ///Methods
  
   bool debug_;
   int ievt_;
   double etaMax_, etaMin_, phiMax_, phiMin_;
   int etaBins_, phiBins_;
   int checkNevents_;
   HcalCalibrations calibs_;

   double coolcellfrac_;
   double Nsigma_;
   double minADCcount_;

   DeadCellHists hbHists, heHists, hoHists, hfHists;
   MonitorElement* meEVT_;


   // MonitorElement* HBDeadCell, HEDeadCell, HODeadCell, HFDeadCell;
   
}; 

#endif
