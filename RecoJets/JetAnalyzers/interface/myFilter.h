#ifndef RecoJets_myFilter_h
#define RecoJets_myFilter_h

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
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

  int _acceptedEvt;

  int _passPt;
  int _passNTrks;
  int _passEMF;
  int _passNJets;
  int _passNTowers;
  int _passMET;
  int _passMETSig;
  int _passHighPtTower;
  int _passNRBX;
  int _passHLT;

  std::string CaloJetAlgorithm;
  edm::InputTag theTriggerResultsLabel;

};

#endif
