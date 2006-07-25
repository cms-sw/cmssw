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

// User switches for HcalPedestalAnalysis to set in cfg are:
//   nevtsample - number of events per sample in which data will be divided
//                for stability checks (default: 9999999),
//   hiSaveflag - flag to save histos of charge per cap-id (default: 0),
//   pedValflag - pedestal validation flag:
//                1 - write new constants and compare with nominal,
//                0 - do not compare, just calculate new constants,
//                2 - compare with nominal, update only if changed.

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
  int done(const HcalPedestals* fInputPedestals, 
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
  TFile* m_file;
  void per2CapsHists(int flag, int id, const HcalDetId detid, const HcalQIESample& qie1, const HcalQIESample& qie2, std::map<HcalDetId, std::map<int,PEDBUNCH> > &toolT);
  void GetPedConst(std::map<HcalDetId,std::map<int, PEDBUNCH > > &toolT, TH1F* PedMeans, TH1F* PedWidths);
  void Trendings(std::map<HcalDetId,std::map<int, PEDBUNCH > > &toolT, TH1F* Chi2, TH1F* CapidAverage, TH1F* CapidChi2);
  int PedValidtn(std::map<HcalDetId,std::map<int, PEDBUNCH > > &toolT, int nTS);
  void AllChanHists(const HcalDetId detid, const HcalQIESample& qie0, const HcalQIESample& qie1, const HcalQIESample& qie2, const HcalQIESample& qie3, const HcalQIESample& qie4, const HcalQIESample& qie5, std::map<HcalDetId, std::map<int,PEDBUNCH> > &toolT);
  std::string m_outputFileROOT;
  std::string m_outputFileMean;
  std::string m_outputFileWidth;
  std::ofstream m_logFile;
  
  int m_startTS;
  int m_endTS;
  int m_nevtsample;
  int m_hiSaveflag;
  int m_pedValflag;

// m_AllPedsOK says whether all new pedestals are consistent with nominal
// values (e.g. from DB): m_AllPedsOK = 1 everything consistent,
//                                      0 some inconsistencies found,
//                                     -1 validation not requested,
//                                     -2 no data to validate.
  int m_AllPedsOK;
  
  const HcalQIEShape* m_shape;
  const HcalQIECoder* m_coder;
  struct{
    std::map<HcalDetId,std::map<int, PEDBUNCH > > PEDTRENDS;
    TH1F* ALLPEDS;
    TH1F* PEDRMS;
    TH1F* PEDMEAN;
    TH1F* CHI2;
    TH1F* CAPID_AVERAGE;
    TH1F* CAPID_CHI2;
  } hbHists, hfHists, hoHists;
  std::map<HcalDetId,std::map<int, PEDBUNCH > >::iterator _meot;
  HcalPedestals* pedCan;
  HcalPedestalWidths* widthCan;
  const HcalPedestals* pedCan_nominal;
  const HcalPedestalWidths* widthCan_nominal;
  HcalPedestals* meansper2caps;
  HcalPedestals* widthsper2caps;
  int evt;
  int sample;
  int evt_curr;
  std::vector<bool> state;

// flag to make gaussian fits to charge dists
  static const int fitflag=0;
};

#endif
