#ifndef TopObjects_TtFullHadronicEvent_h
#define TopObjects_TtFullHadronicEvent_h

#include "AnalysisDataFormats/TopObjects/interface/TtEvent.h"

namespace TtFullHadDaughter{
  /// full hadronic daughter names for common
  /// use and use with the hypotheses
  static const std::string LightQ   ="LightQ"   , LightP   ="LightP",    WPlus ="WPlus" , B   ="B"   , Top   ="Top";
  static const std::string LightQBar="LightQBar", LightPBar="LightPBar", WMinus="WMinus", BBar="BBar", TopBar="TopBar"; 
}

/**
   \class   TtFullHadronicEvent TtFullHadronicEvent.h "AnalysisDataFormats/TopObjects/interface/TtFullHadronicEvent.h"

   \brief   Class derived from the TtEvent for the full hadronic decay channel

   The structure holds information on the hadronic decay channels, 
   all event hypotheses of different classes (user defined during
   production) and a reference to the TtGenEvent (if available). It 
   provides access and administration.
*/

class TtFullHadronicEvent: public TtEvent {
  
 public:
  /// empty constructor
  TtFullHadronicEvent(){};
  /// default destructor
  ~TtFullHadronicEvent() override{};

  /// get top of the given hypothesis
  const reco::Candidate* top(const std::string& key, const unsigned& cmb=0) const { return top(hypoClassKeyFromString(key), cmb); };
  /// get top of the given hypothesis
  const reco::Candidate* top(const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? nullptr : eventHypo(key,cmb). daughter(TtFullHadDaughter::Top); };
  /// get b of the given hypothesis
  const reco::Candidate* b(const std::string& key, const unsigned& cmb=0) const { return b(hypoClassKeyFromString(key), cmb); };
  /// get b of the given hypothesis
  const reco::Candidate* b(const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? nullptr : top(key,cmb)->daughter(TtFullHadDaughter::B); };

  /// get light Q of the given hypothesis
  const reco::Candidate* lightQ(const std::string& key, const unsigned& cmb=0) const { return lightQ(hypoClassKeyFromString(key), cmb); };
  /// get light Q of the given hypothesis
  const reco::Candidate* lightQ(const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? nullptr : wPlus(key,cmb)->daughter(TtFullHadDaughter::LightQ); };

  /// get light P of the given hypothesis
  const reco::Candidate* lightP(const std::string& key, const unsigned& cmb=0) const { return lightP(hypoClassKeyFromString(key), cmb); };
  /// get light P of the given hypothesis
  const reco::Candidate* lightP(const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? nullptr : wMinus(key,cmb)->daughter(TtFullHadDaughter::LightP); };

  /// get Wplus of the given hypothesis
  const reco::Candidate* wPlus(const std::string& key, const unsigned& cmb=0) const { return wPlus(hypoClassKeyFromString(key), cmb); };
  /// get Wplus of the given hypothesis
  const reco::Candidate* wPlus(const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? nullptr : top(key,cmb)->daughter(TtFullHadDaughter::WPlus); };

  /// get anti-top of the given hypothesis
  const reco::Candidate* topBar(const std::string& key, const unsigned& cmb=0) const { return topBar(hypoClassKeyFromString(key), cmb); };
  /// get anti-top of the given hypothesis
  const reco::Candidate* topBar(const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? nullptr : eventHypo(key,cmb). daughter(TtFullHadDaughter::TopBar); };
  /// get anti-b of the given hypothesis
  const reco::Candidate* bBar(const std::string& key, const unsigned& cmb=0) const { return bBar(hypoClassKeyFromString(key), cmb); };
  /// get anti-b of the given hypothesis
  const reco::Candidate* bBar(const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? nullptr : topBar(key,cmb)->daughter(TtFullHadDaughter::BBar  ); };

  /// get light Q bar of the given hypothesis
  const reco::Candidate* lightQBar(const std::string& key, const unsigned& cmb=0) const { return lightQBar(hypoClassKeyFromString(key), cmb); };
  /// get light Q bar of the given hypothesis
  const reco::Candidate* lightQBar(const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? nullptr : wPlus(key,cmb)->daughter(TtFullHadDaughter::LightQBar); };

  /// get light P bar of the given hypothesis
  const reco::Candidate* lightPBar(const std::string& key, const unsigned& cmb=0) const { return lightPBar(hypoClassKeyFromString(key), cmb); };
  /// get light P bar of the given hypothesis
  const reco::Candidate* lightPBar(const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? nullptr : wMinus(key,cmb)->daughter(TtFullHadDaughter::LightPBar); };

  /// get Wminus of the given hypothesis
  const reco::Candidate* wMinus(const std::string& key, const unsigned& cmb=0) const { return wMinus(hypoClassKeyFromString(key), cmb); };
  /// get Wminus of the given hypothesis
  const reco::Candidate* wMinus(const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? nullptr : topBar(key,cmb)->daughter(TtFullHadDaughter::WMinus); };

  /// get top of the TtGenEvent
  const reco::GenParticle* top        () const { return (!genEvt_ ? nullptr : this->genEvent()->top()  ); };
  /// get b of the TtGenEvent
  const reco::GenParticle* b          () const { return (!genEvt_ ? nullptr : this->genEvent()->b()    ); };

  /// get light Q of the TtGenEvent
  const reco::GenParticle* lightQ     () const { return (!genEvt_ ? nullptr : this->genEvent()->daughterQuarkOfWPlus()   ); };
  /// get light P of the TtGenEvent
  const reco::GenParticle* lightP     () const { return (!genEvt_ ? nullptr : this->genEvent()->daughterQuarkOfWMinus()  ); };

  /// get Wplus of the TtGenEvent
  const reco::GenParticle* wPlus      () const { return (!genEvt_ ? nullptr : this->genEvent()->wPlus()   ); };

  /// get anti-top of the TtGenEvent
  const reco::GenParticle* topBar     () const { return (!genEvt_ ? nullptr : this->genEvent()->topBar()  ); };
  /// get anti-b of the TtGenEvent
  const reco::GenParticle* bBar       () const { return (!genEvt_ ? nullptr : this->genEvent()->bBar()    ); };

  /// get light Q bar of the TtGenEvent
  const reco::GenParticle* lightQBar  () const { return (!genEvt_ ? nullptr : this->genEvent()->daughterQuarkBarOfWPlus()   ); };
  /// get light P bar of the TtGenEvent
  const reco::GenParticle* lightPBar  () const { return (!genEvt_ ? nullptr : this->genEvent()->daughterQuarkBarOfWMinus()  ); };

  /// get Wminus of the TtGenEvent
  const reco::GenParticle* wMinus     () const { return (!genEvt_ ? nullptr : this->genEvent()->wMinus()  ); };

  /// print full content of the structure as formated 
  /// LogInfo to the MessageLogger output for debugging  
  void print(const int verbosity=1) const;
};

#endif
