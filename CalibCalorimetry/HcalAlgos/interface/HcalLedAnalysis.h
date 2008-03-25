#ifndef HcalLedAnalysis_H
#define HcalLedAnalysis_H


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"


#include "TH1F.h"
#include "TF1.h"

#include <memory>
//#include <iostream>
#include <fstream>
#include <vector>
#include <string>

class HcalPedestals;
class HcalDbService;
class HcalQIEShape;
class HcalQIECoder;

class HcalLedAnalysis{
  
public:
  
  /// Constructor
  HcalLedAnalysis(const edm::ParameterSet& ps);  
  /// Destructor
  ~HcalLedAnalysis();
  void LedSetup(const std::string& m_outputFileROOT);
  void doPeds(const HcalPedestals* fInputPedestals);
  void LedSampleAnalysis();
  void LedDone();
  void processLedEvent(const HBHEDigiCollection& hbhe,
		    const HODigiCollection& ho,
		    const HFDigiCollection& hf,
		    const HcalDbService& cond);

protected:
  
  
private:
  //###
  //#  LEDBUNCH is used in map<HcalDetId,map<int, LEDBUNCH > > LEDTRENDS;
  //#  For each HcalDetId (channel) a map<int, LEDBUNCH> is associated;
  //#  int was originally cap-id and now is just dummy;
  //#  LEDBUNCH is a pair - first element is the main 
  //#  histo with the ADC values and second one is another pair;
  //#  this pair contains map<int, std::vector<double> > as a first element;
  //#  vector contains some useful variables;
  //#  the second element is a vector of histos (pointers);
  //#  for the "trend" analysis the main histo (with ADC values) is reset every 
  //#  m_nevtsample events and info is put in the other part of the LEDBUNCH;
  //#  so at the end we have the trends for the variables in concern
  //#  which are written in THE vector<TH1F*>; 
  //###  
  typedef std::pair<TH1F*,std::pair<std::map<int, std::vector<double> >,std::vector<TH1F*> > > LEDBUNCH;
  TFile* m_file;
  void LedTSHists(int id, const HcalDetId detid, int TS, const HcalQIESample& qie1, std::map<HcalDetId, std::map<int,LEDBUNCH> > &toolT, float pedestal);
  void GetLedConst(std::map<HcalDetId,std::map<int, LEDBUNCH > > &toolT);
  void LedTrendings(std::map<HcalDetId,std::map<int, LEDBUNCH > > &toolT);
  float BinsizeCorr(float time);
  std::string m_outputFileROOT;
  std::string m_outputFileText;
  std::ofstream m_outFile;
  std::ofstream m_logFile;
  
  int m_startTS;
  int m_endTS;
  int m_nevtsample;
  int m_hiSaveflag;
// analysis flag:
//  m_fitflag = 0  - take mean TS value of averaged pulse shape
//              1  - take peak from landau fit to averaged pulse shape
//              2  - take average of mean TS values per event
//                     (preferred for laser)
//              3  - take average of peaks from landau fits per event
//                     (preferred for LED)
//              4  - 0+1+2+3
  int m_fitflag;
  
  const HcalQIEShape* m_shape;
  const HcalQIECoder* m_coder;
  struct{
    std::map<HcalDetId,std::map<int, LEDBUNCH > > LEDTRENDS;
    TH1F* ALLLEDS;
    TH1F* LEDRMS;
    TH1F* LEDMEAN;
    TH1F* CHI2;
  } hbHists, hfHists, hoHists;
  std::map<HcalDetId,std::map<int, LEDBUNCH > >::iterator _meol;
  std::map<HcalDetId,std::map<int,float> > m_AllPedVals;
  std::map<HcalDetId,std::map<int,float> >::iterator _meee;

  const HcalPedestals* pedCan;
  int evt;
  int sample;
  int evt_curr;
  std::vector<bool> state;

};

#endif
