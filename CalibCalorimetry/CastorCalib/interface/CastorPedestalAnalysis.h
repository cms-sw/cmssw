#ifndef CastorPedestalAnalysis_H
#define CastorPedestalAnalysis_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

#include "TH1F.h"
#include "TF1.h"

#include <memory>
//#include <iostream>
#include <fstream>
#include <vector>
#include <string>

// User switches for CastorPedestalAnalysis to set in cfg are:
//   nevtsample - number of events per sample in which data will be divided
//                for stability checks (default: 0 = do not use),
//   hiSaveflag - flag to save histos of charge per cap-id (default: 0),
//   pedValflag - pedestal validation flag:
//                0 - write out current raw constants (default)
//                1 - write out validated constants

class CastorPedestals;
class CastorPedestalWidths;
class CastorDbService;
class CastorQIEShape;
class CastorQIECoder;
class TFile;
class CastorPedestalAnalysis {
public:
  /// Constructor
  CastorPedestalAnalysis(const edm::ParameterSet& ps);
  /// Destructor
  ~CastorPedestalAnalysis();

  void setup(const std::string& m_outputFileROOT);

  void SampleAnalysis();

  int done(const CastorPedestals* fInputPedestals,
           const CastorPedestalWidths* fInputWidths,
           CastorPedestals* fOutputPedestals,
           CastorPedestalWidths* fOutputWidths);

  void processEvent(const CastorDigiCollection& castor, const CastorDbService& cond);

  // pedestal validation: CastorPedVal=-1 means not validated,
  //                                  0 everything OK,
  //                                  N>0 : mod(N,100000) drifts + width changes
  //                                        int(N/100000) missing channels
  static int CastorPedVal(int nstat[4],
                          const CastorPedestals* fRefPedestals,
                          const CastorPedestalWidths* fRefPedestalWidths,
                          CastorPedestals* fRawPedestals,
                          CastorPedestalWidths* fRawPedestalWidths,
                          CastorPedestals* fValPedestals,
                          CastorPedestalWidths* fValPedestalWidths);

protected:
private:
  //###
  //#  PEDBUNCH is used in map<HcalDetId,map<int, PEDBUNCH > > PEDTRENDS;
  //#  For each HcalDetId (channel) a map<int, PEDBUNCH> is associated;
  //#  int is cap-id (1-4);
  //#  PEDBUNCH is a pair - first element is the main
  //#  histo with the pedestal distribution and second one is another pair;
  //#  this pair contains map<int, std::vector<double> > as a first element;
  //#  int is cap-id, and vector contains some useful variables;
  //#  the second element is a vector of histos (pointers);
  //#  for the "trend" analysis the main histo (with pedestals) is reset every
  //#  nevt_ped events and info is put in the other part of the PEDBUNCH;
  //#  so at the end we have the trends for the variables in concern
  //#  which are written in THE vector<TH1F*>;
  //###
  typedef std::pair<TH1F*, std::pair<std::map<int, std::vector<double> >, std::vector<TH1F*> > > PEDBUNCH;

  void per2CapsHists(int flag,
                     int id,
                     const HcalDetId detid,
                     const HcalQIESample& qie1,
                     const HcalQIESample& qie2,
                     std::map<HcalDetId, std::map<int, PEDBUNCH> >& toolT,
                     const CastorDbService& cond);

  void GetPedConst(std::map<HcalDetId, std::map<int, PEDBUNCH> >& toolT, TH1F* PedMeans, TH1F* PedWidths);

  void Trendings(std::map<HcalDetId, std::map<int, PEDBUNCH> >& toolT, TH1F* Chi2, TH1F* CapidAverage, TH1F* CapidChi2);

  void AllChanHists(const HcalDetId detid,
                    const HcalQIESample& qie0,
                    const HcalQIESample& qie1,
                    const HcalQIESample& qie2,
                    const HcalQIESample& qie3,
                    const HcalQIESample& qie4,
                    const HcalQIESample& qie5,
                    std::map<HcalDetId, std::map<int, PEDBUNCH> >& toolT);

  TFile* m_file;

  std::string m_outputFileROOT;
  std::string m_outputFileMean;
  std::string m_outputFileWidth;
  std::ofstream m_logFile;
  int m_startTS;
  int m_endTS;
  int m_nevtsample;
  int m_pedsinADC;
  int m_hiSaveflag;
  int m_pedValflag;
  int m_AllPedsOK;

  const CastorQIEShape* m_shape;
  const CastorQIECoder* m_coder;
  struct {
    std::map<HcalDetId, std::map<int, PEDBUNCH> > PEDTRENDS;
    TH1F* ALLPEDS;
    TH1F* PEDRMS;
    TH1F* PEDMEAN;
    TH1F* CHI2;
    TH1F* CAPID_AVERAGE;
    TH1F* CAPID_CHI2;
  } castorHists;
  std::map<HcalDetId, std::map<int, PEDBUNCH> >::iterator _meot;
  const CastorPedestals* fRefPedestals;
  const CastorPedestalWidths* fRefPedestalWidths;
  CastorPedestals* fRawPedestals;
  CastorPedestalWidths* fRawPedestalWidths;
  CastorPedestals* fValPedestals;
  CastorPedestalWidths* fValPedestalWidths;
  int evt;
  int sample;
  int evt_curr;
  float m_stat[4];
  std::vector<bool> state;

  // flag to make gaussian fits to charge dists
  static const int fitflag = 0;
};

#endif
