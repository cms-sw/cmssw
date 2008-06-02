#ifndef TopObjects_TtGenEvent_h
#define TopObjects_TtGenEvent_h
#include "AnalysisDataFormats/TopObjects/interface/TopGenEvent.h"

class TtGenEvent: public TopGenEvent {

 public:
  
  TtGenEvent();
  TtGenEvent(reco::GenParticleRefProd&, reco::GenParticleRefProd&);
  virtual ~TtGenEvent();

  bool isTtBar() const {return (top() && topBar());}
  bool isFullHadronic() const { return (isTtBar() && numberOfLeptons()==0);}
  bool isSemiLeptonic() const { return (isTtBar() && numberOfLeptons()==1);}
  bool isFullLeptonic() const { return (isTtBar() && numberOfLeptons()==2);}
  
  //semi-leptonic getters
  const reco::GenParticle* leptonicDecayW() const;
  const reco::GenParticle* leptonicDecayB() const;
  const reco::GenParticle* leptonicDecayTop() const;
  const reco::GenParticle* hadronicDecayW() const;
  const reco::GenParticle* hadronicDecayB() const;
  const reco::GenParticle* hadronicDecayTop() const;
  const reco::GenParticle* hadronicDecayQuark() const;
  const reco::GenParticle* hadronicDecayQuarkBar() const;
  
  //full-leptonic getters
  const reco::GenParticle* lepton() const;
  const reco::GenParticle* leptonBar() const;
  const reco::GenParticle* neutrino() const;
  const reco::GenParticle* neutrinoBar() const;
  
  //full-hadronic getters
  const reco::GenParticle* quarkFromTop() const;
  const reco::GenParticle* quarkFromTopBar() const;
  const reco::GenParticle* quarkFromAntiTop() const;
  const reco::GenParticle* quarkFromAntiTopBar() const;
  
 private:
  
};

#endif
