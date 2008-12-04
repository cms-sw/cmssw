#ifndef DQM_HCALMONITORTASKS_HCALHOTCELLMONITOR_H
#define DQM_HCALMONITORTASKS_HCALHOTCELLMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
//#include "CalibFormats/HcalObjects/interface/HcalCalibrationWidths.h"

#include <map>

/** \class HcalHotCellMonitor
  *  
  * $Date: 2008/07/21 22:55:06 $
  * $Revision: 1.20 $
  * \author W. Fisher - FNAL
  * \ updated by J. Temple - Univ. of Maryland
  */


// Structure holds all hot cell data for a subdetector
struct HotCellHists{
  bool origcheck;
  bool check;

  // Main problem cell histogram
  MonitorElement* problemHotCells;
  MonitorElement* problemHotCells_depth[4];

  double hotDigiSigma; // digi values - pedestal must be > hotDigiSigma * pedestal to be considered hot
  bool makeDiagnostics; // if disabled, don't make 

  //Miscellaneous hot cell plots -- could remove these?
  MonitorElement* maxCellOccMap;
  MonitorElement* maxCellEnergyMap;
  MonitorElement* maxCellEnergy;
  MonitorElement* maxCellTime;
  MonitorElement* maxCellID; 
  
  // Threshold plots
  std::vector<double> thresholds;
  std::vector<MonitorElement*> threshOccMap;
  std::vector<MonitorElement*> threshEnergyMap;
  
  // NADA hot cell info
  MonitorElement* nadaOccMap;
  MonitorElement* nadaEnergyMap;
  MonitorElement* nadaNumHotCells;
  MonitorElement* nadaEnergy;
  MonitorElement* nadaNumNegCells;
  MonitorElement* nadaNegOccMap;
  MonitorElement* nadaNegEnergyMap;
  

  // Digi Plots
  MonitorElement* abovePedSigma;
  std::vector<MonitorElement*> digiPedestalPlots;
  
  // diagnostic histograms (remove eventually?)
  std::vector<MonitorElement*> diagnostic;

  MonitorElement* hotcellsigma;
  MonitorElement* RecHitEnergyDist;
  MonitorElement* DigiEnergyDist;
  MonitorElement* EnergyVsNADAcube;
  MonitorElement* HOT_EnergyVsNADAcube;

  MonitorElement* pedestalValues_depth[4];
  MonitorElement* pedestalWidths_depth[4];
  MonitorElement* RecHitEnergyDist_depth[4];

  // Depth plots -- these are diagnostics
  std::vector <std::vector<MonitorElement*> > threshOccMap_depth;
  std::vector <std::vector<MonitorElement*> > threshEnergyMap_depth;
  MonitorElement* nadaOccMap_depth[4];
  MonitorElement* nadaEnergyMap_depth[4];
  MonitorElement* nadaNegOccMap_depth[4];
  MonitorElement* nadaNegEnergyMap_depth[4];
  
  // Skip over these plots now
  //std::vector <std::vector<MonitorElement*> > digiPedestalPlots_depth;

  // Parameters used in setting NADA cube sizes, thresholds
  double nadaEnergyCandCut0,nadaEnergyCandCut1, nadaEnergyCandCut2;
  double nadaEnergyCubeCut,nadaEnergyCellCut,nadaNegCandCut;
  double nadaEnergyCubeFrac, nadaEnergyCellFrac;
  int nadaMaxDeltaDepth, nadaMaxDeltaEta, nadaMaxDeltaPhi;

  // subdetector info
  int type;
  std::string name;
  bool subdetOn;
  
  /*
    // Not yet used -- eventually allow for turning on/off checks for individual subdetectors?
  bool checkThreshold_;
  bool checkNADA_;
  bool checkAbovePed_;
  */

  double etaMax, etaMin, phiMax, phiMin;
  int etaBins, phiBins;
  // store max cell
  int etaS, phiS, depthS;
  int idS;
  double enS, tS;
  std::vector<std::string> vetoCells;
  int numhotcells;
  int numnegcells;
  int fVerbosity; //0 = no message, 1 = basic messages, 2 = full verbosity
};


class HcalHotCellMonitor: public HcalBaseMonitor {
public:
  HcalHotCellMonitor(); 
  ~HcalHotCellMonitor(); 

  void setup(const edm::ParameterSet& ps, DQMStore* dbe);
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

  void setSubDetectors(bool hb, bool he, bool ho, bool hf);
  void reset();
  void setupVals(HotCellHists& h, int type, HotCellHists& base, const edm::ParameterSet& ps);
  void setupHists(HotCellHists& h, DQMStore* dbe);

private:  ///Monitoring elements

  bool debug_;

  int ievt_;

  float HF_offsets[13][36][2];
  bool checkThreshold_;
  bool checkNADA_;
  bool checkAbovePed_;


  HotCellHists hbHists,heHists,hfHists,hoHists;
  HotCellHists hcalHists;
 
  MonitorElement* meEVT_;

  bool doFCpeds_; // determins if pedestals are in fC or ADC counts
};

#endif
