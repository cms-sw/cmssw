#ifndef DQM_HCALMONITORTASKS_HCALTEMPLATEANALYSIS_H
#define DQM_HCALMONITORTASKS_HCALTEMPLATEANALYSYS_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "DataFormats/LTCDigi/interface/LTCDigi.h"
#include "TFile.h"
#include <iostream>

/** \class HcalTemplateAnalysis
  *  
  * $Date: 2007/04/02 13:23:14 $
  * $Revision: 1.1 $
  * \author W. Fisher - FNAL
  */

using namespace std;

class HcalTemplateAnalysis {
public:
  HcalTemplateAnalysis(); 
  ~HcalTemplateAnalysis(); 

  void setup(const edm::ParameterSet& ps);
  void processEvent(const HBHEDigiCollection& hbhe,
		    const HODigiCollection& ho,
		    const HFDigiCollection& hf,
		    const HBHERecHitCollection& hbHits, 
		    const HORecHitCollection& hoHits,
		    const HFRecHitCollection& hfHits,
		    const LTCDigiCollection& ltc,
		    const HcalDbService& cond);
  void done();
  void reset();

 private:  
  
  string outputFile_;
  double etaMax_, etaMin_, phiMax_, phiMin_;
  int etaBins_, phiBins_;

  TH1F* rechitEnergy_HB;
  TH1F* rechitTime_HB;
  TH1F* rechitEnergy_HF;
  TH1F* rechitTime_HF;
  TH1F* digiShape;
  TH2F* digiOccupancy;  

};

#endif
