#ifndef TopObjects_TtGenEvent_h
#define TopObjects_TtGenEvent_h

#include "AnalysisDataFormats/TopObjects/interface/TopGenEvent.h"


namespace TtFullLepEvtPartons{
  /// full leptonic parton enum used to define the order 
  /// in the vector for lepton jet combinatorics; this 
  /// order has to be obeyed strictly then!
  enum { B, BBar, Lepton, LeptonBar };
}

namespace TtSemiLepEvtPartons{
  /// semi-leptonic parton enum used to define the order 
  /// in the vector for lepton jet combinatorics; this 
  /// order has to be obeyed strictly then!
  enum { LightQ, LightQBar, HadB, LepB, Lepton };
}

namespace TtFullHadEvtPartons{
  /// full hadronic parton enum used to define the order 
  /// in the vector for lepton jet combinatorics; this 
  /// order has to be obeyed strictly then!
  enum { LightQTop, LightQBarTop, B, LightQTopBar, LightQBarTopBar, BBar};
}

namespace WDecay{
  /// classification of leptons in the decay channel 
  /// of the W boson used in several places throughout 
  /// the package
  enum LeptonType {kNone, kElec, kMuon, kTau};
}

// ----------------------------------------------------------------------
// derived class for: 
//
//  * TtGenEvent
//
//  the structure holds reference information to the generator particles 
//  of the decay chains for each top quark and of the initial partons 
//  and provides access and administration;  the derived class contains 
//  a few additional getters with respect to its base class
// ----------------------------------------------------------------------

class TtGenEvent: public TopGenEvent {

 public:

  /// empty constructor  
  TtGenEvent() {};
  /// default constructor
  TtGenEvent(reco::GenParticleRefProd&, reco::GenParticleRefProd&);
  /// default destructor
  virtual ~TtGenEvent() {};

  /// check if the event can be classified as ttbar
  bool isTtBar() const {return (top() && topBar());}
  /// check if the event can be classified as full hadronic
  bool isFullHadronic() const { return isTtBar() ? numberOfLeptons()==0 : false;}
  /// check if the event can be classified as semi-laptonic
  bool isSemiLeptonic() const { return isTtBar() ? numberOfLeptons()==1 : false;}
  /// check if the event can be classified as full leptonic
  bool isFullLeptonic() const { return isTtBar() ? numberOfLeptons()==2 : false;}
  /// return decay channel
  WDecay::LeptonType semiLeptonicChannel() const;
  /// check if the event is semi-leptonic with the lepton being of typeA
  bool isSemiLeptonic(WDecay::LeptonType typeA) const { return (semiLeptonicChannel()==typeA ? true : false); };
  /// check if the event is semi-leptonic with the lepton being of typeA or typeB
  bool isSemiLeptonic(WDecay::LeptonType typeA, WDecay::LeptonType typeB) const { return ( (semiLeptonicChannel()==typeA || semiLeptonicChannel()==typeB)? true : false); };

  /// get W of leptonic decay branch
  const reco::GenParticle* leptonicDecayW() const;
  /// get b of leptonic decay branch
  const reco::GenParticle* leptonicDecayB() const;
  /// get top of leptonic decay branch
  const reco::GenParticle* leptonicDecayTop() const;
  /// get W of hadronic decay branch
  const reco::GenParticle* hadronicDecayW() const;
  /// get b of hadronic decay branch
  const reco::GenParticle* hadronicDecayB() const;
  /// get top of hadronic decay branch
  const reco::GenParticle* hadronicDecayTop() const;
  /// get light quark of hadronic decay branch
  const reco::GenParticle* hadronicDecayQuark(bool invert=false) const;
  /// get light anti-quark of hadronic decay branch
  const reco::GenParticle* hadronicDecayQuarkBar() const {return hadronicDecayQuark(true); };
  /// gluons as radiated from the leptonicly decaying top quark
  std::vector<const reco::GenParticle*> leptonicDecayTopRadiation() const;
  /// gluons as radiated from the hadronicly decaying top quark
  std::vector<const reco::GenParticle*> hadronicDecayTopRadiation() const;
  /// get lepton for semi-leptonic or full leptonic decays
  const reco::GenParticle* lepton() const;
  /// get anti-lepton for semi-leptonic or full leptonic decays
  const reco::GenParticle* leptonBar() const;
  /// get neutrino for semi-leptonic or full leptonic decays
  const reco::GenParticle* neutrino() const;
  /// get anti-neutrino for semi-leptonic or full leptonic decays
  const reco::GenParticle* neutrinoBar() const;
  /// get light quark from top for full hadronic decays
  const reco::GenParticle* lightQFromTop() const;
  /// get light anti-quark from top for full hadronic decays
  const reco::GenParticle* lightQBarFromTop() const;
  /// get light quark from anti-top for full hadronic decays
  const reco::GenParticle* lightQFromTopBar() const;
  /// get light anti-quark from anti-top for full hadronic decays
  const reco::GenParticle* lightQBarFromTopBar() const;
};

#endif
