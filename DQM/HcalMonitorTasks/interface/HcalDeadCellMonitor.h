#ifndef DQM_HCALMONITORTASKS_HCALDEADCELLMONITOR_H
#define DQM_HCALMONITORTASKS_HCALDEADCELLMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrationWidths.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"


/** \class HcalDeadCellMonitor
  *  
  * $Date: 2008/05/22 14:04:19 $
  * $Revision: 1.7 $
  * \author J. Temple - Univ. of Maryland
  */

struct DeadCellHists{

  bool check; // determine whether to run DeadCell checks on this subdetector
  bool fVerbosity; // not yet implemented for subdetectors -- use later?

  int type; // store subdetector type (1=hb, 2=he, 3=ho, 4=hf, 10=hcal)
  std::string subdet; // store subdetector name (HB, HE,...)

  // Global histogram to keep track of all bad histograms in a given subdetector
  MonitorElement* problemDeadCells;

  // Dead cell routine #1:  low ADC counts for cell
  MonitorElement* deadADC_map;
  std::vector<MonitorElement*> deadADC_map_depth; // individual depth plots
  std::vector<TH2F*> deadADC_temp_depth;
  //MonitorElement* noADC_ID_map;
  MonitorElement* deadADC_eta;
  //MonitorElement* noADC_ID_eta;
  MonitorElement* ADCdist;
  std::vector<MonitorElement*> deadcapADC_map; // plots for individual CAPIDs

  // Dead cell routine #2:  cell cool compared to neighbors
  double floor, mindiff;
  MonitorElement* NADA_cool_cell_map;
  std::vector<MonitorElement*> NADA_cool_cell_map_depth; // individual depth plots

  // Dead cell routine #3:  cell consistently less than pedestal + N sigma
  MonitorElement* coolcell_below_pedestal;
  MonitorElement* above_pedestal;
  std::vector<MonitorElement*> coolcell_below_pedestal_depth;
  std::vector<MonitorElement*> above_pedestal_depth;
  std::vector<TH2F*> above_pedestal_temp_depth;
  
  // extra diagnostic plots - could be removed?  
  // Should already have these in DigiMonitor, RecHitMonitor
  MonitorElement* digiCheck;
  MonitorElement* cellCheck;
  std::vector<MonitorElement*> digiCheck_depth;
  std::vector<MonitorElement*> cellCheck_depth;
};


class HcalDeadCellMonitor: public HcalBaseMonitor {
 public:
  HcalDeadCellMonitor(); 
  ~HcalDeadCellMonitor(); 

  void setup(const edm::ParameterSet& ps, DQMStore* dbe);
  void done(); // overrides base class function 
  void clearME(); // overrides base class function

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
  void setupHists(DeadCellHists& hist,  DQMStore* dbe);
  void reset_Nevents(DeadCellHists& h);
  void reset();


 private:  ///Methods
  
   bool debug_;
   bool doFCpeds_; // true if ped values are in FC; otherwise, assume peds in ADC counts

   int ievt_;
   double etaMax_, etaMin_, phiMax_, phiMin_;
   int etaBins_, phiBins_;
   int checkNevents_;

   double coolcellfrac_;
   double Nsigma_;
   double minADCcount_;

   double floor_, mindiff_;

   DeadCellHists hbHists, heHists, hoHists, hfHists, hcalHists;
   MonitorElement* meEVT_;
   MonitorElement* meCheckN_; // copy of checkNevents saved to MonitorElement
   HcalCalibrations calibs_; // shouldn't be necessary any more


   //MonitorElement* problemDeadCells_;
}; 

#endif
