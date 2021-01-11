#ifndef CastorLedAnalysis_H
#define CastorLedAnalysis_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

#include "TH1F.h"
#include "TF1.h"
#include "TProfile.h"

#include <memory>
//#include <iostream>
#include <fstream>
#include <vector>
#include <string>

class CastorPedestal;
class CastorDbService;
class CastorQIEShape;
class CastorQIECoder;
class TFile;

class CastorLedAnalysis {
public:
  /// Constructor
  CastorLedAnalysis(const edm::ParameterSet& ps);
  /// Destructor
  ~CastorLedAnalysis();
  void LedSetup(const std::string& m_outputFileROOT);
  //void doPeds(const CastorPedestals* fInputPedestals);
  void LedSampleAnalysis();
  void LedDone();
  void processLedEvent(const CastorDigiCollection& castor, const CastorDbService& cond);

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
  typedef std::pair<TH1F*, std::pair<std::map<int, std::vector<double> >, std::vector<TH1F*> > > LEDBUNCH;
  typedef struct {
    TProfile* avePulse[3];
    TH1F* thisPulse[3];
    TH1F* integPulse[3];
  } CALIBBUNCH;
  TFile* m_file;
  void LedCastorHists(const HcalDetId& detid,
                      const CastorDataFrame& ledDigi,
                      std::map<HcalDetId, std::map<int, LEDBUNCH> >& toolT,
                      const CastorDbService& cond);
  void SetupLEDHists(int id, const HcalDetId detid, std::map<HcalDetId, std::map<int, LEDBUNCH> >& toolT);
  void GetLedConst(std::map<HcalDetId, std::map<int, LEDBUNCH> >& toolT);
  void LedTrendings(std::map<HcalDetId, std::map<int, LEDBUNCH> >& toolT);
  float BinsizeCorr(float time);

  std::string m_outputFileROOT;
  std::string m_outputFileText;
  std::string m_outputFileX;
  std::ofstream m_outFile;
  std::ofstream m_logFile;
  std::ofstream m_outputFileXML;

  int m_startTS;
  int m_endTS;
  int m_nevtsample;
  int m_hiSaveflag;
  bool m_usecalib;
  // analysis flag:
  //  m_fitflag = 0  - take mean TS value of averaged pulse shape
  //              1  - take peak from landau fit to averaged pulse shape
  //              2  - take average of mean TS values per event
  //                     (preferred for laser & HF LED)
  //              3  - take average of peaks from landau fits per event
  //                     (preferred for LED)
  //              4  - 0+1+2+3 REMOVED in 1_6
  int m_fitflag;

  const CastorQIEShape* m_shape;
  const CastorQIECoder* m_coder;
  const CastorPedestal* m_ped;
  struct {
    std::map<HcalDetId, std::map<int, LEDBUNCH> > LEDTRENDS;
    TH1F* ALLLEDS;
    TH1F* LEDRMS;
    TH1F* LEDMEAN;
    TH1F* CHI2;
  } castorHists;
  std::map<HcalDetId, std::map<int, LEDBUNCH> >::iterator _meol;
  std::map<HcalDetId, std::map<int, float> > m_AllPedVals;
  std::map<HcalDetId, std::map<int, float> >::iterator _meee;

  std::map<HcalCalibDetId, CALIBBUNCH>::iterator _meca;

  //const CastorPedestal* pedCan;
  int evt;
  int sample;
  int evt_curr;
  std::vector<bool> state;
};

#endif
