#ifndef DQM_HCALMONITORTASKS_HCALDIGIMONITOR_H
#define DQM_HCALMONITORTASKS_HCALDIGIMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "EventFilter/HcalRawToDigi/interface/HcalUnpacker.h"
#include <cmath>

#define DIGI_BQ_FRAC_NBINS 101
#define DIGI_NUM 9072
#define DIGI_SUBDET_NUM 3000

/** \class HcalDigiMonitor
  *  
  * $Date: 2009/07/21 11:02:48 $
  * $Revision: 1.42 $
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
  std::vector<MonitorElement*> TS_sum_plus, TS_sum_minus;

  int count_shape[10];
  int count_shapeThresh[10];
  int count_presample[50];
  int count_BQ[DIGI_SUBDET_NUM];
  int count_BQFrac[DIGI_BQ_FRAC_NBINS];
  int count_bad;
  int count_all;
  int capIDdiff[8]; // only bins 0-7 used for expected real values of cap ID difference (since cap IDs run from 0-3); bin 8 is overflow
  int dverr[4];
  int capid[4];
  int adc[200];
  int adcsum[200];
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
		    const ZDCDigiCollection& zdc,
		    const HcalDbService& cond,
		    const HcalUnpackerReport& report);
  void reset();
  void setSubDetectors(bool hb, bool he, bool ho, bool hf,bool zdc);
private:  ///Methods

  void zeroCounters();
  void fill_Nevents();
  void setupHists(DigiHists& hist,  DQMStore* dbe); // enable this feature at some point

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

private:  
  const HcalQIEShape* shape_;
  const HcalQIECoder* channelCoder_;

  // Monitoring elements

  EtaPhiHists DigiErrorsBadCapID;
  EtaPhiHists DigiErrorsBadDigiSize;
  EtaPhiHists DigiErrorsBadADCSum;
  EtaPhiHists DigiErrorsDVErr;
  MonitorElement* DigiSize;
  int problemdigis[85][72][4];
  int badcapID[85][72][4];
  int baddigisize[85][72][4];
  int badADCsum[85][72][4];
  int digisize[20][4];
  int digierrorsdverr[85][72][4];

  EtaPhiHists DigiOccupancyByDepth;
  MonitorElement* DigiOccupancyEta;
  MonitorElement* DigiOccupancyPhi;
  MonitorElement* DigiOccupancyVME;
  MonitorElement* DigiOccupancySpigot;
  
  
  int occupancyEtaPhi[85][72][4];
  int occupancyEta[85];
  int occupancyPhi[72];
  int occupancyVME[40][18];
  int occupancySpigot[40][36];

  //MonitorElement* DigiErrorEtaPhi; //redundant; sample as ProblemDigis
  MonitorElement* DigiErrorVME;
  MonitorElement* DigiErrorSpigot;
  MonitorElement* DigiBQ;
  MonitorElement* DigiBQFrac;

  int digiBQ[DIGI_NUM]; 
  int digiBQfrac[DIGI_BQ_FRAC_NBINS]; 
  int errorVME[40][18];
  int errorSpigot[15][36];// 15 is the value of SPIGOT_COUNT; may need to change this in the future?


  MonitorElement* DigiFirstCapID;
  MonitorElement* DigiNum;
  int diginum[DIGI_NUM];
 

  // ZDC stuff


  DigiHists hbHists, heHists, hfHists, hoHists, zdcHists;

 
};

#endif
