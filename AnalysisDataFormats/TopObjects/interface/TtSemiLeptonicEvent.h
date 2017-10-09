#ifndef TopObjects_TtSemiLeptonicEvent_h
#define TopObjects_TtSemiLeptonicEvent_h

#include "AnalysisDataFormats/TopObjects/interface/TtEvent.h"

namespace TtSemiLepDaughter{
  /// semi-leptonic daughter names for common
  /// use and use with the hypotheses
  static const std::string Nu  ="Nu"  , Lep ="Lep" , LepW="LepW", LepB="LepB", LepTop="LepTop";
  static const std::string HadQ="HadQ", HadP="HadP", HadW="HadW", HadB="HadB", HadTop="HadTop"; 
}

/**
   \class   TtSemiLeptonicEvent TtSemiLeptonicEvent.h "AnalysisDataFormats/TopObjects/interface/TtSemiLeptonicEvent.h"

   \brief   Class derived from the TtEvent for the semileptonic decay channel

   The structure holds information on the leptonic decay channels, 
   all event hypotheses of different classes (user defined during
   production) and a reference to the TtGenEvent (if available). It 
   provides access and administration.
*/

class TtSemiLeptonicEvent: public TtEvent {
  
 public:

  /// empty constructor
  TtSemiLeptonicEvent(){};
  /// default destructor
  virtual ~TtSemiLeptonicEvent(){};

  /// get hadronic top of the given hypothesis
  const reco::Candidate* hadronicDecayTop(const std::string& key, const unsigned& cmb=0) const { return hadronicDecayTop(hypoClassKeyFromString(key), cmb); };
  /// get hadronic top of the given hypothesis
  const reco::Candidate* hadronicDecayTop(const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : eventHypo(key,cmb). daughter(TtSemiLepDaughter::HadTop); };
  /// get hadronic b of the given hypothesis
  const reco::Candidate* hadronicDecayB(const std::string& key, const unsigned& cmb=0) const { return hadronicDecayB(hypoClassKeyFromString(key), cmb); };
  /// get hadronic b of the given hypothesis
  const reco::Candidate* hadronicDecayB(const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : hadronicDecayTop(key,cmb)->daughter(TtSemiLepDaughter::HadB); };
  /// get hadronic W of the given hypothesis
  const reco::Candidate* hadronicDecayW(const std::string& key, const unsigned& cmb=0) const { return hadronicDecayW(hypoClassKeyFromString(key), cmb); };
  /// get hadronic W of the given hypothesis
  const reco::Candidate* hadronicDecayW(const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : hadronicDecayTop(key,cmb)->daughter(TtSemiLepDaughter::HadW); };
  /// get hadronic light quark of the given hypothesis
  const reco::Candidate* hadronicDecayQuark(const std::string& key, const unsigned& cmb=0) const { return hadronicDecayQuark(hypoClassKeyFromString(key), cmb); };
  /// get hadronic light quark of the given hypothesis
  const reco::Candidate* hadronicDecayQuark(const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : hadronicDecayW(key,cmb)->daughter(TtSemiLepDaughter::HadP); };
  /// get hadronic light quark of the given hypothesis
  const reco::Candidate* hadronicDecayQuarkBar(const std::string& key, const unsigned& cmb=0) const { return hadronicDecayQuarkBar(hypoClassKeyFromString(key), cmb); };
  /// get hadronic light quark of the given hypothesis
  const reco::Candidate* hadronicDecayQuarkBar(const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : hadronicDecayW(key,cmb)->daughter(TtSemiLepDaughter::HadQ); };
  /// get leptonic top of the given hypothesis
  const reco::Candidate* leptonicDecayTop(const std::string& key, const unsigned& cmb=0) const { return leptonicDecayTop(hypoClassKeyFromString(key), cmb); };
  /// get leptonic top of the given hypothesis
  const reco::Candidate* leptonicDecayTop(const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : eventHypo(key,cmb). daughter(TtSemiLepDaughter::LepTop); };
  /// get leptonic b of the given hypothesis
  const reco::Candidate* leptonicDecayB(const std::string& key, const unsigned& cmb=0) const { return leptonicDecayB(hypoClassKeyFromString(key), cmb); };
  /// get leptonic b of the given hypothesis
  const reco::Candidate* leptonicDecayB(const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : leptonicDecayTop(key,cmb)->daughter(TtSemiLepDaughter::LepB); };
  /// get leptonic W of the given hypothesis
  const reco::Candidate* leptonicDecayW(const std::string& key, const unsigned& cmb=0) const { return leptonicDecayW(hypoClassKeyFromString(key), cmb); };
  /// get leptonic W of the given hypothesis
  const reco::Candidate* leptonicDecayW(const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : leptonicDecayTop(key,cmb)->daughter(TtSemiLepDaughter::LepW); };
  /// get leptonic light quark of the given hypothesis
  const reco::Candidate* singleNeutrino(const std::string& key, const unsigned& cmb=0) const { return singleNeutrino(hypoClassKeyFromString(key), cmb); };
  /// get leptonic light quark of the given hypothesis
  const reco::Candidate* singleNeutrino(const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : leptonicDecayW(key,cmb)->daughter(TtSemiLepDaughter::Nu); };
  /// get leptonic light quark of the given hypothesis
  const reco::Candidate* singleLepton(const std::string& key, const unsigned& cmb=0) const { return singleLepton(hypoClassKeyFromString(key), cmb); };
  /// get leptonic light quark of the given hypothesis
  const reco::Candidate* singleLepton(const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : leptonicDecayW(key,cmb)->daughter(TtSemiLepDaughter::Lep); };

  /// get hadronic top of the TtGenEvent
  const reco::GenParticle* hadronicDecayTop() const { return (!genEvt_ ? 0 : this->genEvent()->hadronicDecayTop()); };
  /// get hadronic b of the TtGenEvent
  const reco::GenParticle* hadronicDecayB() const { return (!genEvt_ ? 0 : this->genEvent()->hadronicDecayB()); };
  /// get hadronic W of the TtGenEvent
  const reco::GenParticle* hadronicDecayW() const { return (!genEvt_ ? 0 : this->genEvent()->hadronicDecayW()); };
  /// get hadronic light quark of the TtGenEvent
  const reco::GenParticle* hadronicDecayQuark() const { return (!genEvt_ ? 0 : this->genEvent()->hadronicDecayQuark()); };
  /// get hadronic light quark of the TtGenEvent
  const reco::GenParticle* hadronicDecayQuarkBar() const { return (!genEvt_ ? 0 : this->genEvent()->hadronicDecayQuarkBar()); };
  /// get leptonic top of the TtGenEvent
  const reco::GenParticle* leptonicDecayTop() const { return (!genEvt_ ? 0 : this->genEvent()->leptonicDecayTop()); };
  /// get leptonic b of the TtGenEvent
  const reco::GenParticle* leptonicDecayB() const { return (!genEvt_ ? 0 : this->genEvent()->leptonicDecayB()); };
  /// get leptonic W of the TtGenEvent
  const reco::GenParticle* leptonicDecayW() const { return (!genEvt_ ? 0 : this->genEvent()->leptonicDecayW()); };
  /// get lepton top of the TtGenEvent
  const reco::GenParticle* singleLepton() const { return (!genEvt_ ? 0 : this->genEvent()->singleLepton());   };
  /// get neutrino of the TtGenEvent
  const reco::GenParticle* singleNeutrino() const { return (!genEvt_ ? 0 : this->genEvent()->singleNeutrino()); };

  /// print full content of the structure as formated 
  /// LogInfo to the MessageLogger output for debugging
  void print(const int verbosity=1) const;

  /// get number of real neutrino solutions for a given hypo class
  const int numberOfRealNeutrinoSolutions(const HypoClassKey& key) const { return (numberOfRealNeutrinoSolutions_.find(key)==numberOfRealNeutrinoSolutions_.end() ? -999 : numberOfRealNeutrinoSolutions_.find(key)->second); };
  /// get number of real neutrino solutions for a given hypo class
  const int numberOfRealNeutrinoSolutions(const std::string& key) const { return numberOfRealNeutrinoSolutions(hypoClassKeyFromString(key)); };

  /// set number of real neutrino solutions for a given hypo class
  void setNumberOfRealNeutrinoSolutions(const HypoClassKey& key, const int& nr) { numberOfRealNeutrinoSolutions_[key] = nr; };

 protected:

  /// number of real neutrino solutions for all hypo classes
  std::map<HypoClassKey, int> numberOfRealNeutrinoSolutions_;

};

#endif
