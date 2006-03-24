#ifndef HcalPedestalAnalysis_H
#define HcalPedestalAnalysis_H


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
class HcalPedestalWidths;
class HcalDbService;
class HcalQIEShape;
class HcalQIECoder;

class HcalPedestalAnalysis{
  
public:
  
  /// Constructor
  HcalPedestalAnalysis(const edm::ParameterSet& ps);  
  /// Destructor
  ~HcalPedestalAnalysis();
  void setup(const std::string& m_outputFileROOT);
  void SampleAnalysis();
  void done(const HcalPedestals* fInputPedestals, 
	    const HcalPedestalWidths* fInputWidths,
	    HcalPedestals* fOutputPedestals, 
	    HcalPedestalWidths* fOutputWidths);

  void processEvent(const HBHEDigiCollection& hbhe,
		    const HODigiCollection& ho,
		    const HFDigiCollection& hf,
		    const HcalDbService& cond);

protected:
  
  
private:
  //###
  //#  PEDBUNCH is used in map<HcalDetId,map<int, PEDBUNCH > > PEDTRENDS;
  //#  For each HcalDetId (channel) a map<int, PEDBUNCH> is associated;
  //#  int is cap-id (1-4);
  //#  PEDBUNCH is a pair - first element is the main 
  //#  histo with the ADC values and second one is another pair;
  //#  this pair contains map<int, std::vector<double> > as a first element;
  //#  int is cap-id, and vector contains some useful variables;
  //#  the second element is a vector of histos (pointers);
  //#  for the "trend" analysis the main histo (with ADC values) is reset every 
  //#  nevt_ped events and info is put in the other part of the PEDBUNCH;
  //#  so at the end we have the trends for the variables in concern
  //#  which are written in THE vector<TH1F*>; 
  //###  
  typedef std::pair<TH1F*,std::pair<std::map<int, std::vector<double> >,std::vector<TH1F*> > > PEDBUNCH;
  TFile* m_file; // Histogram file  
  void per2CapsHists(int flag, int id, const HcalDetId detid, const HcalQIESample& qie1, const HcalQIESample& qie2, std::map<HcalDetId, std::map<int,PEDBUNCH> > &toolT);
  void GetPedConst(std::map<HcalDetId,std::map<int, PEDBUNCH > > &toolT);
  void Trendings(std::map<HcalDetId,std::map<int, PEDBUNCH > > &toolT, TH1F* Chi2, TH1F* CapidAverage, TH1F* CapidChi2);
  int PedValidtn(std::map<HcalDetId,std::map<int, PEDBUNCH > > &toolT);
  std::string m_outputFileROOT;
  std::string m_outputFileMean;
  std::string m_outputFileWidth;
  std::ofstream m_logFile;
  
  int m_startSample;
  int m_endSample;
  
  const HcalQIEShape* m_shape;
  const HcalQIECoder* m_coder;
  struct{
    std::map<HcalDetId,std::map<int, PEDBUNCH > > PEDTRENDS;
    TH1F* ALLPEDS;
    TH1F* PEDRMS; // sigma
    TH1F* PEDMEAN;
    TH1F* CHI2;
    TH1F* CAPID_AVERAGE;
    TH1F* CAPID_CHI2;//
  } hbHists, hfHists, hoHists;
  std::map<HcalDetId, std::map<int,TH1F*> >::iterator _meo;
  std::map<HcalDetId,std::map<int, PEDBUNCH > >::iterator _meot;
  HcalPedestals* pedCan;
  HcalPedestalWidths* widthCan;
  const HcalPedestals* pedCan_nominal;
  const HcalPedestalWidths* widthCan_nominal;
  HcalPedestals* meansper2caps;
  HcalPedestals* widthsper2caps;
  int evt;
  int sample;
  static const int fitflag=1;
  static const int nevt_ped=1000;
  static const int pedValflag=0;
  std::vector<bool> state;
};

#endif
