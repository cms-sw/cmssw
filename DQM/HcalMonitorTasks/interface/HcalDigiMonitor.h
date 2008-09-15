#ifndef DQM_HCALMONITORTASKS_HCALDIGIMONITOR_H
#define DQM_HCALMONITORTASKS_HCALDIGIMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "EventFilter/HcalRawToDigi/interface/HcalUnpacker.h"
// The following are needed for using pedestals in fC:
#include "CondFormats/HcalObjects/interface/HcalPedestal.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidth.h"

/** \class HcalDigiMonitor
  *  
  * $Date: 2008/09/05 21:10:49 $
  * $Revision: 1.30 $
  * \author W. Fisher - FNAL
  */


struct DigiHists
{
  // check whether to create histograms for each subdetector
  bool origcheck;
  bool check;
  int checkNevents;
  
  bool makeDiagnostics; // determine whether or not to make diagnostic plots
  
  int type; // stores type  (1=hb, 2=he, 3=ho, 4=hf, 10=hcal)
  std::string subdet; // stores subdetector name (HB, HE, ...)

    // Problem cells will be those cells with an error or with low occupancy
  MonitorElement* PROBLEMDIGICELLS;
  std::vector<MonitorElement*> PROBLEMDIGICELLS_DEPTH;
  
  TH2F* PROBLEMDIGICELLS_TEMP;
  std::vector<TH2F*> PROBLEMDIGICELLS_TEMP_DEPTH;
  // Separate by types of problems
  std::vector<MonitorElement*> problemCell_noDigi;
  std::vector<MonitorElement*> problemCell_BadCapID;
  std::vector<MonitorElement*> problemCell_WrongDigiSize;
  std::vector<MonitorElement*> problemCell_ADCSumIsZero;

  MonitorElement* DIGI_NUM;
  MonitorElement* DIGI_SIZE;
  MonitorElement* DIGI_PRESAMPLE;
  MonitorElement* QIE_CAPID;
  MonitorElement* QIE_ADC;
  MonitorElement* QIE_DV;
  MonitorElement* ERR_MAP_GEO;
  MonitorElement* ERR_MAP_VME;
  
  MonitorElement* ERR_MAP_DCC;
  MonitorElement* OCC_MAP_GEO1;
  MonitorElement* OCC_MAP_GEO2;
  MonitorElement* OCC_MAP_GEO3;
  MonitorElement* OCC_MAP_GEO4;
  MonitorElement* OCC_ETA;
  MonitorElement* OCC_PHI;
  MonitorElement* OCC_MAP_VME;
  
  MonitorElement* OCC_MAP_DCC;
  MonitorElement* SHAPE_tot;
  MonitorElement* SHAPE_THR_tot;
  
  MonitorElement* CAPID_T0;
  MonitorElement* BQDIGI_NUM;
  MonitorElement* BQDIGI_FRAC;
  
  // Pedestal plots for each depth of each subdetector
  MonitorElement* RAW_PEDESTAL_MEAN[4];
  MonitorElement* RAW_PEDESTAL_RMS[4];
  MonitorElement* SUB_PEDESTAL_MEAN[4];
  MonitorElement* SUB_PEDESTAL_RMS[4];

  
  std::vector<MonitorElement*> TS_SUM_P, TS_SUM_M;
  
  MonitorElement* diagnostic_BeforeBadDigi;
  MonitorElement* diagnostic_AfterBadDigi;
  
  std::map<HcalDetId, MonitorElement*> SHAPE;

  // temp variables for storing values that are periodically Filled in histograms
  int temp_QIE_CAPID[5];
  int temp_QIE_ADC[200];
  float temp_SHAPE_tot[10];
  float temp_SHAPE_THR_tot[10];
  int temp_QIE_DV[4];
};

class HcalDigiMonitor: public HcalBaseMonitor {
public:
  HcalDigiMonitor(); 
  ~HcalDigiMonitor(); 

  void setup(const edm::ParameterSet& ps, DQMStore* dbe);
  void processEvent(const HBHEDigiCollection& hbhe,
		    const HODigiCollection& ho,
		    const HFDigiCollection& hf,
		    const HcalDbService& cond,
		    const HcalUnpackerReport& report);
  void setSubDetectors(bool hb, bool he, bool ho, bool hf);
  void reset();

private:  ///Methods

  void fillErrors(const HBHEDataFrame& hb);
  void fillErrors(const HODataFrame& ho);
  void fillErrors(const HFDataFrame& hf);
  void fillPedestalHistos(void);
  void reset_Nevents(DigiHists& h);
  void fill_Nevents(DigiHists& h);
  void setupHists(DigiHists& hist,  DQMStore* dbe); // enable this feature at some point


  void HBHEDigiCheck(const HBHEDigiCollection& hbhe, DigiHists& hbHists, DigiHists& heHists, 
		   DigiHists& hcalHists,  const HcalDbService& cond, 
		   int& ndigi, int& nbqdigi);
  void HODigiCheck(const HODigiCollection& ho, DigiHists& hoHists, 
		   DigiHists& hcalHists,  const HcalDbService& cond, 
		   int& ndigi, int& nbqdigi);
  void HFDigiCheck(const HFDigiCollection& hf, DigiHists& hfHists, 
		   DigiHists& hcalHists,  const HcalDbService& cond, 
		   int& ndigi, int& nbqdigi);


  int ievt_;
  double etaMax_, etaMin_, phiMax_, phiMin_;
  int etaBins_, phiBins_;
  bool doPerChannel_;
  bool doFCpeds_;
  int occThresh_;
  HcalCalibrations calibs_;
  int checkNevents_;

private:  
  const HcalQIEShape* shape_;
  const HcalQIECoder* channelCoder_;

  // Monitoring elements
  MonitorElement* meEVT_;
  MonitorElement* OCC_ETA;
  MonitorElement* OCC_PHI;
  MonitorElement* OCC_L1;
  MonitorElement* OCC_L2;
  MonitorElement* OCC_L3;
  MonitorElement* OCC_L4;
  MonitorElement* OCC_ELEC_VME;
  
  MonitorElement* OCC_ELEC_DCC;
  MonitorElement* ERR_MAP_GEO;
  MonitorElement* ERR_MAP_VME;

  MonitorElement* ERR_MAP_DCC;

  MonitorElement* CAPID_T0;
  MonitorElement* DIGI_NUM;
  MonitorElement* BQDIGI_NUM;
  MonitorElement* BQDIGI_FRAC;
  
  //Quick pedestal code
  int pedcounts[83][72][4];
  float rawpedsum[83][72][4];
  float rawpedsum2[83][72][4];
  float subpedsum[83][72][4]; 
  float subpedsum2[83][72][4]; 

  MonitorElement* RAW_PEDESTAL_MEAN[4];
  MonitorElement* RAW_PEDESTAL_RMS[4];
  MonitorElement* SUB_PEDESTAL_MEAN[4]; 
  MonitorElement* SUB_PEDESTAL_RMS[4]; 
 
  DigiHists hbHists, heHists, hfHists, hoHists, hcalHists;

};

#endif
