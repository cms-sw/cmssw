#ifndef TopObjects_TtGenEvent_h
#define TopObjects_TtGenEvent_h
#include "DataFormats/Candidate/interface/Candidate.h"

class TtGenEvent
{
 public:
  TtGenEvent();
  TtGenEvent(reco::CandidateRefProd&);
  virtual ~TtGenEvent();
  const reco::CandidateCollection& particles() const {return *parts_;};
  int numberOfLeptons() const;
  bool isFullHadronic() const { return (numberOfLeptons()==0);};
  bool isSemiLeptonic() const { return (numberOfLeptons()==1);};
  bool isFullLeptonic() const { return (numberOfLeptons()==2);};
  const reco::Candidate* candidate(int) const;

  //semi-leptonic getters
  const reco::Candidate* singleLepton() const;
  const reco::Candidate* singleNeutrino() const;
  const reco::Candidate* leptonicW() const;
  const reco::Candidate* leptonicB() const;
  const reco::Candidate* leptonicTop() const;
  const reco::Candidate* hadronicW() const;
  const reco::Candidate* hadronicB() const;
  const reco::Candidate* hadronicTop() const;
  const reco::Candidate* hadronicQuark() const;
  const reco::Candidate* hadronicQuarkBar() const;

  //full-leptonic getters
  const reco::Candidate* lepton() const;
  const reco::Candidate* leptonBar() const;
  const reco::Candidate* neutrino() const;
  const reco::Candidate* neutrinoBar() const;

  //full-hadronic getters
  const reco::Candidate* quarkFromTop() const;
  const reco::Candidate* quarkFromTopBar() const;
  const reco::Candidate* quarkFromAntiTop() const;
  const reco::Candidate* quarkFromAntiTopBar() const;

  //common getters
  const reco::Candidate* electron() const { return candidate( 11 );};
  const reco::Candidate* positron() const { return candidate(-11 );};
  const reco::Candidate* muon() const { return candidate( 13 );};
  const reco::Candidate* muonBar() const { return candidate(-13 );};
  const reco::Candidate* tau() const { return candidate( 15 );};
  const reco::Candidate* tauBar() const { return candidate(-15 );};
  const reco::Candidate* top() const { return candidate( 6 );};
  const reco::Candidate* topBar() const { return candidate(-6 );};
  const reco::Candidate* w() const { return candidate( 24 );};
  const reco::Candidate* wBar() const { return candidate(-24 );};
  const reco::Candidate* b() const { return candidate( 5 );};
  const reco::Candidate* bBar() const { return candidate(-5 );};

 private:
  bool isLepton(const reco::Candidate& part) const 
  {return (abs(part.pdgId())==11 || abs(part.pdgId())==13 || abs(part.pdgId())==15);};
  bool isNeutrino(const reco::Candidate& part) const 
  {return (abs(part.pdgId())==12 || abs(part.pdgId())==14 || abs(part.pdgId())==16);};
  double flavour(const reco::Candidate& part) const 
  {return (double)(part.pdgId() / abs(part.pdgId()) );};
  
  reco::CandidateRefProd parts_;
};

#endif
