#ifndef TopObjects_TtSemiLeptonicEvent_h
#define TopObjects_TtSemiLeptonicEvent_h

#include "AnalysisDataFormats/TopObjects/interface/TtEvent.h"

namespace TtSemiLepDaughter{
  // semileptonic daughter names
  static const std::string Nu  ="Nu"  , Lep ="Lep" , LepW="LepW", LepB="LepB", LepTop="LepTop";
  static const std::string HadQ="HadQ", HadP="HadP", HadW="HadW", HadB="HadB", HadTop="HadTop"; 
}

// ----------------------------------------------------------------------
// derived class for: 
//
//  * TtFullLeptonicEvent
//
//  the structure holds information on the leptonic decay channels, 
//  all event hypotheses of different classes (user defined during
//  production) and a reference to the TtGenEvent (if available) 
//  and provides access and administration; the derived class 
//  contains a few additional getters with respect to its base class
// ----------------------------------------------------------------------

class TtSemiLeptonicEvent: public TtEvent {
  
 public:

  /// empty constructor
  TtSemiLeptonicEvent(){};
  /// default destructor
  virtual ~TtSemiLeptonicEvent(){};

  /// get hadronic top of the given hypothesis
  const reco::Candidate* hadronicTop(const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : eventHypo  (key,cmb). daughter(TtSemiLepDaughter::HadTop); };
  /// get hadronic b of the given hypothesis
  const reco::Candidate* hadronicB  (const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : hadronicTop(key,cmb)->daughter(TtSemiLepDaughter::HadB  ); };
  /// get hadronic W of the given hypothesis
  const reco::Candidate* hadronicW  (const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : hadronicTop(key,cmb)->daughter(TtSemiLepDaughter::HadW  ); };
  /// get hadronic light quark of the given hypothesis
  const reco::Candidate* lightQuarkP(const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : hadronicW  (key,cmb)->daughter(TtSemiLepDaughter::HadP  ); };
  /// get hadronic light quark of the given hypothesis
  const reco::Candidate* lightQuarkQ(const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : hadronicW  (key,cmb)->daughter(TtSemiLepDaughter::HadQ  ); };
  /// get leptonic top of the given hypothesis
  const reco::Candidate* leptonicTop(const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : eventHypo  (key,cmb). daughter(TtSemiLepDaughter::LepTop); };
  /// get leptonic b of the given hypothesis
  const reco::Candidate* leptonicB  (const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : leptonicTop(key,cmb)->daughter(TtSemiLepDaughter::LepB  ); };
  /// get leptonic W of the given hypothesis
  const reco::Candidate* leptonicW  (const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : leptonicTop(key,cmb)->daughter(TtSemiLepDaughter::LepW  ); };
  /// get leptonic light quark of the given hypothesis
  const reco::Candidate* neutrino   (const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : leptonicW  (key,cmb)->daughter(TtSemiLepDaughter::Nu    ); };
  /// get leptonic light quark of the given hypothesis
  const reco::Candidate* lepton     (const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : leptonicW  (key,cmb)->daughter(TtSemiLepDaughter::Lep   ); };

  /// get hadronic top of the TtGenEvent
  const reco::GenParticle* genHadronicTop() const { return (!genEvt_ ? 0 : this->genEvent()->hadronicDecayTop()); };
  /// get hadronic b of the TtGenEvent
  const reco::GenParticle* genHadronicW()   const { return (!genEvt_ ? 0 : this->genEvent()->hadronicDecayW()); };
  /// get hadronic W of the TtGenEvent
  const reco::GenParticle* genHadronicB()   const { return (!genEvt_ ? 0 : this->genEvent()->hadronicDecayB()); };
  /// get hadronic light quark of the TtGenEvent
  const reco::GenParticle* genHadronicP()   const { return (!genEvt_ ? 0 : this->genEvent()->hadronicDecayQuark()); };
  /// get hadronic light quark of the TtGenEvent
  const reco::GenParticle* genHadronicQ()   const { return (!genEvt_ ? 0 : this->genEvent()->hadronicDecayQuarkBar()); };
  /// get leptonic top of the TtGenEvent
  const reco::GenParticle* genLeptonicTop() const { return (!genEvt_ ? 0 : this->genEvent()->leptonicDecayTop()); };
  /// get leptonic b of the TtGenEvent
  const reco::GenParticle* genLeptonicW()   const { return (!genEvt_ ? 0 : this->genEvent()->leptonicDecayW()); };
  /// get leptonic W of the TtGenEvent
  const reco::GenParticle* genLeptonicB()   const { return (!genEvt_ ? 0 : this->genEvent()->leptonicDecayB()); };
  /// get lepton top of the TtGenEvent
  const reco::GenParticle* genLepton()      const { return (!genEvt_ ? 0 : this->genEvent()->singleLepton());   };
  /// get neutrino of the TtGenEvent
  const reco::GenParticle* genNeutrino()    const { return (!genEvt_ ? 0 : this->genEvent()->singleNeutrino()); };
  
  /// print full content of the structure as formated 
  /// LogInfo to the MessageLogger output for debugging
  void print();
};

#endif
