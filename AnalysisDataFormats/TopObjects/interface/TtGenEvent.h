#ifndef TopObjects_TtGenEvent_h
#define TopObjects_TtGenEvent_h
#include "AnalysisDataFormats/TopObjects/interface/TopGenEvent.h"

class TtGenEvent: public TopGenEvent {

 public:
  
  TtGenEvent();
  TtGenEvent(reco::CandidateRefProd&, std::vector<const reco::Candidate*>);
  virtual ~TtGenEvent();

  bool isFullHadronic() const { return (numberOfLeptons()==0);}
  bool isSemiLeptonic() const { return (numberOfLeptons()==1);}
  bool isFullLeptonic() const { return (numberOfLeptons()==2);}
  
  //semi-leptonic getters
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
  const reco::Candidate* quarkFromTop() const;
  const reco::Candidate* quarkFromTopBar() const;
  const reco::Candidate* quarkFromAntiTop() const;
  const reco::Candidate* quarkFromAntiTopBar() const;
  
 private:
  
};

#endif
