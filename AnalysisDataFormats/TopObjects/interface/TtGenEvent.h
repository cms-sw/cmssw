#ifndef TopObjects_TtGenEvent_h
#define TopObjects_TtGenEvent_h

#include "CommonTools/CandUtils/interface/pdgIdUtils.h"
#include "AnalysisDataFormats/TopObjects/interface/TopGenEvent.h"

/**
   \class   TtGenEvent TtGenEvent.h "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"

   \brief   Class derived from the TopGenEvent for ttbar events

   The structure holds reference information to the generator particles 
   of the decay chains for each top quark and of the initial partons 
   and provides access and administration. The derived class contains 
   a few additional getters with respect to its base class.
*/

class TtGenEvent: public TopGenEvent {

 public:

  /// empty constructor  
  TtGenEvent() {};
  /// default constructor from decaySubset and initSubset
  TtGenEvent(reco::GenParticleRefProd& decaySubset, reco::GenParticleRefProd& initSubset);
  /// default destructor
  virtual ~TtGenEvent() {};

  /// check if the event can be classified as ttbar
  bool isTtBar() const {return (top() && topBar());}
  /// check if the tops were produced from a pair of gluons (and not from qqbar)
  bool fromGluonFusion() const;
  /// check if the tops were produced from a qqbar pair (and not from gg fusion)
  bool fromQuarkAnnihilation() const { return !fromGluonFusion(); };
  /// check if the event can be classified as full hadronic
  bool isFullHadronic(bool excludeTauLeptons=false) const { return isTtBar() ? isNumberOfLeptons(excludeTauLeptons, 0) : false;}
  /// check if the event can be classified as semi-laptonic
  bool isSemiLeptonic(bool excludeTauLeptons=false) const { return isTtBar() ? isNumberOfLeptons(excludeTauLeptons, 1) : false;}
  /// check if the event can be classified as full leptonic
  bool isFullLeptonic(bool excludeTauLeptons=false) const { return isTtBar() ? isNumberOfLeptons(excludeTauLeptons, 2) : false;}

  /// return decay channel; all leptons including taus are allowed 
  WDecay::LeptonType semiLeptonicChannel() const;
  /// check if the event is semi-leptonic with the lepton being of typeA; all leptons including taus are allowed
  bool isSemiLeptonic(WDecay::LeptonType typeA) const { return semiLeptonicChannel()==typeA ? true : false; };
  /// check if the event is semi-leptonic with the lepton being of typeA or typeB; all leptons including taus are allowed
  bool isSemiLeptonic(WDecay::LeptonType typeA, WDecay::LeptonType typeB) const { return (semiLeptonicChannel()==typeA || semiLeptonicChannel()==typeB)? true : false; };
  // return decay channel (as a std::pair of LeptonType's); all leptons including taus are allowed
  std::pair<WDecay::LeptonType, WDecay::LeptonType> fullLeptonicChannel() const;
  /// check if the event is full leptonic with the lepton being of typeA or typeB irrelevant of order; all leptons including taus are allowed
  bool isFullLeptonic(WDecay::LeptonType typeA, WDecay::LeptonType typeB) const;

  /// return single lepton if available; 0 else
  const reco::GenParticle* singleLepton(bool excludeTauLeptons=false) const;
  /// return single neutrino if available; 0 else
  const reco::GenParticle* singleNeutrino(bool excludeTauLeptons=false) const;
  /// get W of leptonic decay branch
  const reco::GenParticle* leptonicDecayW(bool excludeTauLeptons=false) const;
  /// get b of leptonic decay branch
  const reco::GenParticle* leptonicDecayB(bool excludeTauLeptons=false) const;
  /// get top of leptonic decay branch
  const reco::GenParticle* leptonicDecayTop(bool excludeTauLeptons=false) const;
  /// get W of hadronic decay branch
  const reco::GenParticle* hadronicDecayW(bool excludeTauLeptons=false) const;
  /// get b of hadronic decay branch
  const reco::GenParticle* hadronicDecayB(bool excludeTauLeptons=false) const;
  /// get top of hadronic decay branch
  const reco::GenParticle* hadronicDecayTop(bool excludeTauLeptons=false) const;
  /// get light quark of hadronic decay branch
  const reco::GenParticle* hadronicDecayQuark(bool invertFlavor=false) const;
  /// get light anti-quark of hadronic decay branch
  const reco::GenParticle* hadronicDecayQuarkBar() const {return hadronicDecayQuark(true); };
  /// gluons as radiated from the leptonicly decaying top quark
  std::vector<const reco::GenParticle*> leptonicDecayTopRadiation(bool excludeTauLeptons=false) const;
  /// gluons as radiated from the hadronicly decaying top quark
  std::vector<const reco::GenParticle*> hadronicDecayTopRadiation(bool excludeTauLeptons=false) const;
  /// get lepton for semi-leptonic or full leptonic decays
  const reco::GenParticle* lepton(bool excludeTauLeptons=false) const;
  /// get anti-lepton for semi-leptonic or full leptonic decays
  const reco::GenParticle* leptonBar(bool excludeTauLeptons=false) const;
  /// get neutrino for semi-leptonic or full leptonic decays
  const reco::GenParticle* neutrino(bool excludeTauLeptons=false) const;
  /// get anti-neutrino for semi-leptonic or full leptonic decays
  const reco::GenParticle* neutrinoBar(bool excludeTauLeptons=false) const;

  /// return combined 4-vector of top and topBar
  const math::XYZTLorentzVector* topPair() const { return isTtBar() ? &topPair_ : 0; };

 protected:

  /// combined 4-vector of top and topBar
  math::XYZTLorentzVector topPair_;

 private:

  /// check whether the number of leptons among the daughters of the W boson is nlep
  /// or not; there is an option to exclude taus from the list of leptons to consider
  bool isNumberOfLeptons(bool excludeTauLeptons, int nlep) const {return excludeTauLeptons ? (numberOfLeptons()-numberOfLeptons(WDecay::kTau))==nlep : numberOfLeptons()==nlep;}
};

inline bool
TtGenEvent::isFullLeptonic(WDecay::LeptonType typeA, WDecay::LeptonType typeB) const
{
  return ( (fullLeptonicChannel().first==typeA && fullLeptonicChannel().second==typeB)||
	   (fullLeptonicChannel().first==typeB && fullLeptonicChannel().second==typeA));
}

#endif
