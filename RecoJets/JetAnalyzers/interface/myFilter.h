#ifndef RecoJets_myFilter_h
#define RecoJets_myFilter_h

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <string>

class myFilter : public edm::EDFilter {

public:
  myFilter(const edm::ParameterSet& );
  virtual ~myFilter();
  virtual bool filter(edm::Event& e, edm::EventSetup const& c);
  virtual void beginJob();
  virtual void endJob();

private:

  int _nEvent;
  int _nTotal;

  int _acceptedEvt;

  int _passPt;
  int _passNTrks;
  int _passEMF;
  int _passNJets;
  int _passDiJet;
  int _passNTowers;
  int _passMET;
  int _passMETSig;
  int _passHighPtTower;
  int _passNRBX;
  int _passNHPDNoise;
  int _passHLT;
  int _passNPMTHits;
  int _passNMultiPMTHits;
  int _passPKAM;
  int _passHFMET;
  int _passNoiseSummary;
  int _passNoiseSummaryEMF;
  int _passNoiseSummaryE2E10;
  int _passNoiseSummaryNHITS;
  int _passNoiseSummaryADC0;
  int _passNoiseSummaryNoOther;
  int _passOERatio;
  int _passTime;
  int _passHFTime;
  int _passHBHETime;
  int _passHFFlagged;
  int _passHFHighEnergy;

  int _NoiseResult[10];

  std::string CaloJetAlgorithm;
  edm::InputTag theTriggerResultsLabel;
  edm::InputTag hcalNoiseSummaryTag_;
};

#endif
