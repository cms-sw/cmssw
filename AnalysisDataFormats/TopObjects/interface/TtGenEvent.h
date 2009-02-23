#ifndef TopObjects_TtGenEvent_h
#define TopObjects_TtGenEvent_h
#include "AnalysisDataFormats/TopObjects/interface/TopGenEvent.h"

class TtGenEvent: public TopGenEvent {

 public:

  // semiletponic decay channel
  enum LeptonType {kNone, kElec, kMuon, kTau};
  
 public:
  
  TtGenEvent();
  TtGenEvent(reco::GenParticleRefProd&, reco::GenParticleRefProd&, int defaultStatus_ = 4);
  virtual ~TtGenEvent();


  bool isTtBar() const {return (top() && topBar());}
  bool isFullHadronic() const { return isTtBar() ? numberOfLeptons()==0 : false;}
  bool isSemiLeptonic() const { return isTtBar() ? numberOfLeptons()==1 : false;}
  bool isFullLeptonic() const { return isTtBar() ? numberOfLeptons()==2 : false;}
  
  //semi-leptonic getters
  LeptonType semiLeptonicChannel() const;
  bool isSemiLeptonic(LeptonType typeA) const 
  { return (semiLeptonicChannel()==typeA ? true : false); };
  bool isSemiLeptonic(LeptonType typeA, LeptonType typeB) const 
  { return ( (semiLeptonicChannel()==typeA || semiLeptonicChannel()==typeB)? true : false); };

  const reco::GenParticle* leptonicDecayW() const;
  const reco::GenParticle* leptonicDecayB() const;
  const reco::GenParticle* leptonicDecayTop() const;
  const reco::GenParticle* hadronicDecayW() const;
  const reco::GenParticle* hadronicDecayB() const;
  const reco::GenParticle* hadronicDecayTop() const;
  const reco::GenParticle* hadronicDecayQuark(bool invert=false ) const;
  const reco::GenParticle* hadronicDecayQuarkBar() const {return hadronicDecayQuark(true); };
  const std::vector<const reco::GenParticle*> leptonicDecayTopRadiation() const;
  const std::vector<const reco::GenParticle*> hadronicDecayTopRadiation() const;
  
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
