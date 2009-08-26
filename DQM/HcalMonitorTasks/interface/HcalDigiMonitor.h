#ifndef DQM_HCALMONITORTASKS_HCALDIGIMONITOR_H
#define DQM_HCALMONITORTASKS_HCALDIGIMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "EventFilter/HcalRawToDigi/interface/HcalUnpacker.h"
#include <cmath>

#define DIGI_BQ_FRAC_NBINS 101
#define DIGI_NUM 9072
#define DIGI_SUBDET_NUM 2593 

/** \class HcalDigiMonitor
  *  
  * $Date: 2009/08/26 13:56:42 $
  * $Revision: 1.47 $
  * \author W. Fisher - FNAL
  * \author J. Temple - Univ. of Maryland
  */

struct DigiHists
{
  // structure of MonitorElements for each subdetector, along with their associated counters

  bool check; // decide whether or not to use these histograms

  MonitorElement* shape;
  MonitorElement* shapeThresh;
  MonitorElement* presample;
  MonitorElement* BQ;
  MonitorElement* BQFrac;
  MonitorElement* DigiFirstCapID;
  MonitorElement* DVerr;
  MonitorElement* CapID;
  MonitorElement* ADC;
  MonitorElement* ADCsum;
  MonitorElement* fibBCNOff;
  std::vector<MonitorElement*> TS_sum_plus, TS_sum_minus;

  int count_shape[10];
  int count_shapeThresh[10];
  int count_presample[50];
  int count_BQ[DIGI_SUBDET_NUM];
  int count_BQFrac[DIGI_BQ_FRAC_NBINS];
  int count_bad;
  int count_good;

  int capIDdiff[8]; // only bins 0-7 used for expected real values of cap ID difference (since cap IDs run from 0-3); bin 8 is overflow
  int dverr[4];
  int capid[4];
  int adc[200];
  int adcsum[200];
  int fibbcnoff[15];
  int tssumplus[50][10];
  int tssumminus[50][10];
};

class HcalDigiMonitor: public HcalBaseMonitor {
public:
  HcalDigiMonitor(); 
  ~HcalDigiMonitor(); 

  void setup(const edm::ParameterSet& ps, DQMStore* dbe);
  void processEvent(const HBHEDigiCollection& hbhe,
		    const HODigiCollection& ho,
		    const HFDigiCollection& hf,
		    //const ZDCDigiCollection& zdc,
		    const HcalDbService& cond,
		    const HcalUnpackerReport& report);		
  void reset();
  void setSubDetectors(bool hb, bool he, bool ho, bool hf);
  void fill_Nevents();

private:  ///Methods, variables accessible only within class code

  void zeroCounters();
  void setupSubdetHists(DigiHists& hist,  std::string subdet); // enable this feature at some point

  template<class T> int process_Digi(T& digi, DigiHists& hist, int& firstcap);
  void UpdateHists(DigiHists& h);

  bool doPerChannel_;
  bool doFCpeds_;
  int shapeThresh_;
  int occThresh_;
  int digi_checkNevents_;
  int mindigisize_, maxdigisize_;
  bool digi_checkoccupancy_;
  bool digi_checkcapid_;
  bool digi_checkdigisize_;
  bool digi_checkadcsum_;
  bool digi_checkdverr_;

  int hbcount_, hecount_, hocount_, hfcount_;  // Counter # of good digis each event

  const HcalQIEShape* shape_;
  const HcalQIECoder* channelCoder_;

  // Monitoring elements

  EtaPhiHists DigiErrorsByDepth;
  EtaPhiHists DigiErrorsBadCapID;
  EtaPhiHists DigiErrorsBadDigiSize;
  EtaPhiHists DigiErrorsBadADCSum;
  EtaPhiHists DigiErrorsDVErr;
  EtaPhiHists DigiErrorsBadFibBCNOff;


  MonitorElement* DigiSize;
  int baddigis[85][72][4]; // sum of individual digi problems
  int badcapID[85][72][4];
  int baddigisize[85][72][4];
  int badADCsum[85][72][4];
  int badFibBCNOff[85][72][4];
  int digisize[20][4];
  int digierrorsdverr[85][72][4];


  // Digi Occupancy Plots
  EtaPhiHists DigiOccupancyByDepth;
  MonitorElement* DigiOccupancyEta;
  MonitorElement* DigiOccupancyPhi;
  MonitorElement* DigiOccupancyVME;
  MonitorElement* DigiOccupancySpigot;
  
  // Counters for good and bad digis
  int occupancyEtaPhi[85][72][4];
  int occupancyEta[85];
  int occupancyPhi[72];
  int occupancyVME[40][18];
  int occupancySpigot[40][36];

  // Plots for Digis that are present, but with errors
  EtaPhiHists DigiErrorOccupancyByDepth;
  MonitorElement* DigiErrorOccupancyEta;
  MonitorElement* DigiErrorOccupancyPhi;
  MonitorElement* DigiErrorVME;
  MonitorElement* DigiErrorSpigot;

  MonitorElement* DigiBQ;
  MonitorElement* DigiBQFrac;
  
  int occupancyErrorEtaPhi[85][72][4];
  int occupancyErrorEta[85];
  int occupancyErrorPhi[72];
  int errorVME[40][18];
  int errorSpigot[15][36];// 15 is the value of SPIGOT_COUNT; may need to change this in the future?
  int digiBQ[DIGI_NUM]; 
  int digiBQfrac[DIGI_BQ_FRAC_NBINS]; 

  MonitorElement* HBocc_vs_LB;
  MonitorElement* HEocc_vs_LB;
  MonitorElement* HOocc_vs_LB;
  MonitorElement* HFocc_vs_LB;



  MonitorElement* DigiFirstCapID;
  MonitorElement* DigiNum;
  int diginum[DIGI_NUM];
 

  // ZDC stuff


  DigiHists hbHists, heHists, hfHists, hoHists;

 
};

#endif
