#ifndef TopObjects_TtGenEvent_h
#define TopObjects_TtGenEvent_h
#include "AnalysisDataFormats/TopObjects/interface/TopGenEvent.h"

class TtGenEvent: public TopGenEvent {

 public:
  
  TtGenEvent();
  TtGenEvent(reco::CandidateRefProd&, std::vector<const reco::Candidate*>);
  virtual ~TtGenEvent();

  int numberOfLeptons() const;
  int numberOfBQuarks() const;
  bool isFullHadronic() const { return (numberOfLeptons()==0);}
  bool isSemiLeptonic() const { return (numberOfLeptons()==1);}
  bool isFullLeptonic() const { return (numberOfLeptons()==2);}
  
  //semi-leptonic getters
  const reco::Candidate* singleLepton() const;
  const reco::Candidate* singleNeutrino() const;
  const reco::Candidate* leptonicDecayW() const;
  const reco::Candidate* leptonicDecayB() const;
  const reco::Candidate* leptonicDecayTop() const;
  const reco::Candidate* hadronicDecayW() const;
  const reco::Candidate* hadronicDecayB() const;
  const reco::Candidate* hadronicDecayTop() const;
  const reco::Candidate* hadronicDecayQuark() const;
  const reco::Candidate* hadronicDecayQuarkBar() const;
  
  //full-leptonic getters
  const reco::Candidate* lepton() const;
  const reco::Candidate* leptonBar() const;
  const reco::Candidate* neutrino() const;
  const reco::Candidate* neutrinoBar() const;
  
  //full-hadronic getters
  std::vector<const reco::Candidate*> lightQuarks(bool plusB=false) const;
  const reco::Candidate* quarkFromTop() const;
  const reco::Candidate* quarkFromTopBar() const;
  const reco::Candidate* quarkFromAntiTop() const;
  const reco::Candidate* quarkFromAntiTopBar() const;
  
 private:
  
  bool isLepton(const reco::Candidate& part) const 
  {return (abs(part.pdgId())==11 || abs(part.pdgId())==13 || abs(part.pdgId())==15);}
  bool isNeutrino(const reco::Candidate& part) const 
  {return (abs(part.pdgId())==12 || abs(part.pdgId())==14 || abs(part.pdgId())==16);}
  double flavour(const reco::Candidate& part) const 
  {return (double)(part.pdgId() / abs(part.pdgId()) );}
};

#endif
