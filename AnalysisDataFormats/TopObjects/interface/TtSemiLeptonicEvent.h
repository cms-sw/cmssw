#ifndef TopObjects_TtSemiLeptonicEvent_h
#define TopObjects_TtSemiLeptonicEvent_h

#include "AnalysisDataFormats/TopObjects/interface/TtEvent.h"

namespace TtSemiLepDaughter{
  // semileptonic daughter names
  static const std::string Nu  ="Nu"  , Lep ="Lep" , LepW="LepW", LepB="LepB", LepTop="LepTop";
  static const std::string HadQ="HadQ", HadP="HadP", HadW="HadW", HadB="HadB", HadTop="HadTop"; 
}

namespace TtSemiLepEvtPartonsFwd{
  // semileptonic parton names
  enum { LightQ, LightQBar, HadB, LepB, Lepton };
}

class TtSemiLeptonicEvent: public TtEvent {
  
 public:

  TtSemiLeptonicEvent(){};
  virtual ~TtSemiLeptonicEvent(){};

  // access objects according to corresponding event hypothesis
  const reco::Candidate* hadronicTop(const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : eventHypo  (key,cmb). daughter(TtSemiLepDaughter::HadTop); };
  const reco::Candidate* hadronicB  (const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : hadronicTop(key,cmb)->daughter(TtSemiLepDaughter::HadB  ); };
  const reco::Candidate* hadronicW  (const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : hadronicTop(key,cmb)->daughter(TtSemiLepDaughter::HadW  ); };
  const reco::Candidate* lightQuarkP(const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : hadronicW  (key,cmb)->daughter(TtSemiLepDaughter::HadP  ); };
  const reco::Candidate* lightQuarkQ(const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : hadronicW  (key,cmb)->daughter(TtSemiLepDaughter::HadQ  ); };
  const reco::Candidate* leptonicTop(const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : eventHypo  (key,cmb). daughter(TtSemiLepDaughter::LepTop); };
  const reco::Candidate* leptonicB  (const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : leptonicTop(key,cmb)->daughter(TtSemiLepDaughter::LepB  ); };
  const reco::Candidate* leptonicW  (const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : leptonicTop(key,cmb)->daughter(TtSemiLepDaughter::LepW  ); };
  const reco::Candidate* neutrino   (const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : leptonicW  (key,cmb)->daughter(TtSemiLepDaughter::Nu    ); };
  const reco::Candidate* lepton     (const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : leptonicW  (key,cmb)->daughter(TtSemiLepDaughter::Lep   ); };

  // access the matched gen particles
  const reco::GenParticle* genHadronicTop() const { return (!genEvt_ ? 0 : this->genEvent()->hadronicDecayTop()); };
  const reco::GenParticle* genHadronicW()   const { return (!genEvt_ ? 0 : this->genEvent()->hadronicDecayW()); };
  const reco::GenParticle* genHadronicB()   const { return (!genEvt_ ? 0 : this->genEvent()->hadronicDecayB()); };
  const reco::GenParticle* genHadronicP()   const { return (!genEvt_ ? 0 : this->genEvent()->hadronicDecayQuark()); };
  const reco::GenParticle* genHadronicQ()   const { return (!genEvt_ ? 0 : this->genEvent()->hadronicDecayQuarkBar()); };
  const reco::GenParticle* genLeptonicTop() const { return (!genEvt_ ? 0 : this->genEvent()->leptonicDecayTop()); };
  const reco::GenParticle* genLeptonicW()   const { return (!genEvt_ ? 0 : this->genEvent()->leptonicDecayW()); };
  const reco::GenParticle* genLeptonicB()   const { return (!genEvt_ ? 0 : this->genEvent()->leptonicDecayB()); };
  const reco::GenParticle* genLepton()      const { return (!genEvt_ ? 0 : this->genEvent()->singleLepton());   };
  const reco::GenParticle* genNeutrino()    const { return (!genEvt_ ? 0 : this->genEvent()->singleNeutrino()); };
  
  void print();
};

#endif
