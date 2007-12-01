#ifndef DQM_HCALMONITORTASKS_HCALHOTCELLMONITOR_H
#define DQM_HCALMONITORTASKS_HCALHOTCELLMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include <map>

/** \class HcalHotCellMonitor
  *  
  * $Date: 2007/11/19 18:20:54 $
  * $Revision: 1.9 $
  * \author W. Fisher - FNAL
  */

struct HistList{
  MonitorElement* meOCC_MAP_GEO_Max;
  MonitorElement* meEN_MAP_GEO_Max;
  MonitorElement* meMAX_E;
  MonitorElement* meMAX_T;
  MonitorElement* meMAX_ID; 
  std::vector<MonitorElement*> OCCmap;
  std::vector<MonitorElement*> ENERGYmap;
};

struct NADAHistList{
  MonitorElement* NADA_OCC_MAP;
  MonitorElement* NADA_EN_MAP;
  MonitorElement* NADA_NumHotCells;
  MonitorElement* NADA_testcell;
  MonitorElement* NADA_Energy;
  MonitorElement* NADA_NumNegCells;
  MonitorElement* NADA_NEG_OCC_MAP;
  MonitorElement* NADA_NEG_EN_MAP;
};

class HcalHotCellMonitor: public HcalBaseMonitor {
public:
  HcalHotCellMonitor(); 
  ~HcalHotCellMonitor(); 

  void setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe);
  void processEvent(const HBHERecHitCollection& hbHits, const HORecHitCollection& hoHits, const HFRecHitCollection& hfHits);
  void reset();
  void FindHBHEHotCells(const HBHERecHitCollection& hbHits, HistList& hist, bool HB);
  void FindHOHotCells(const HORecHitCollection& hoHits, HistList& hist);
  void FindHFHotCells(const HFRecHitCollection& hfHits, HistList& hist);

  void HBHE_NADAFinder(const HBHERecHitCollection& c, NADAHistList& h, bool HB);
  void HO_NADAFinder(const HORecHitCollection& c, NADAHistList& h);
  void HF_NADAFinder(const HFRecHitCollection& c, NADAHistList& h);
  

private:  ///Monitoring elements

  bool debug_;
  int ievt_;

  bool checkHB_, checkHE_, checkHO_, checkHF_;

  // allow for individual threshold cuts in each subdetector
  std::vector<double> thresholds_;
  std::vector<double> HEthresholds_, HBthresholds_, HFthresholds_, HOthresholds_;



  double etaMax_, etaMin_, phiMax_, phiMin_;
  int etaBins_, phiBins_;
  double NADA_Ecand_cut0_,NADA_Ecand_cut1_, NADA_Ecand_cut2_;
  double NADA_Ecube_cut_,NADA_Ecell_cut_,NADA_NegCand_cut_;
  double NADA_Ecube_frac_, NADA_Ecell_frac_;
  int NADA_maxdepth_, NADA_maxeta_, NADA_maxphi_;

  double HB_NADA_Ecand_cut0_,HB_NADA_Ecand_cut1_, HB_NADA_Ecand_cut2_;
  double HB_NADA_Ecube_cut_,HB_NADA_Ecell_cut_,HB_NADA_NegCand_cut_;
  double HB_NADA_Ecube_frac_, HB_NADA_Ecell_frac_;
  double HE_NADA_Ecand_cut0_,HE_NADA_Ecand_cut1_, HE_NADA_Ecand_cut2_;
  double HE_NADA_Ecube_cut_,HE_NADA_Ecell_cut_,HE_NADA_NegCand_cut_;
  double HE_NADA_Ecube_frac_, HE_NADA_Ecell_frac_;
  double HO_NADA_Ecand_cut0_,HO_NADA_Ecand_cut1_, HO_NADA_Ecand_cut2_;
  double HO_NADA_Ecube_cut_,HO_NADA_Ecell_cut_,HO_NADA_NegCand_cut_;
  double HO_NADA_Ecube_frac_, HO_NADA_Ecell_frac_;
  double HF_NADA_Ecand_cut0_,HF_NADA_Ecand_cut1_, HF_NADA_Ecand_cut2_;
  double HF_NADA_Ecube_cut_,HF_NADA_Ecell_cut_,HF_NADA_NegCand_cut_;
  double HF_NADA_Ecube_frac_, HF_NADA_Ecell_frac_;
  int HB_NADA_maxdepth_, HB_NADA_maxeta_, HB_NADA_maxphi_;
  int HE_NADA_maxdepth_, HE_NADA_maxeta_, HE_NADA_maxphi_;
  int HO_NADA_maxdepth_, HO_NADA_maxeta_, HO_NADA_maxphi_;
  int HF_NADA_maxdepth_, HF_NADA_maxeta_, HF_NADA_maxphi_;



  float enS, tS, etaS, phiS, idS;
  float enA, tA, etaA, phiA;
  int depth;

  int hotcells;
  int negcells;
  float HF_offsets[13][36][2];


  MonitorElement* meOCC_MAP_L1;
  MonitorElement* meEN_MAP_L1;
  MonitorElement* meOCC_MAP_L2;
  MonitorElement* meEN_MAP_L2;
  MonitorElement* meOCC_MAP_L3;
  MonitorElement* meEN_MAP_L3;
  MonitorElement* meOCC_MAP_L4;
  MonitorElement* meEN_MAP_L4;

  MonitorElement* meOCC_MAP_all;
  MonitorElement* meEN_MAP_all;

  MonitorElement* meMAX_E_all;
  MonitorElement* meMAX_T_all;
  MonitorElement* meEVT_;

  MonitorElement* NADA_NumHotCells;
  MonitorElement* NADA_NumNegCells;

  HistList hbHists,heHists,hfHists,hoHists,hcalHists;
  NADAHistList NADA_hbHists, NADA_heHists, NADA_hfHists, NADA_hoHists, NADA_hcalHists;
  
  // To do:  Add in NADA histogram for hot NADA cells in each layer


};

#endif
