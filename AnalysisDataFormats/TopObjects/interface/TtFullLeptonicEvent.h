#ifndef TopObjects_TtFullLeptonicEvent_h
#define TopObjects_TtFullLeptonicEvent_h

#include "AnalysisDataFormats/TopObjects/interface/TtEvent.h"

namespace TtFullLepDaughter{
  // fully-leptonic daughter names
  static const std::string Nu   ="Nu"   , LepBar="LepBar", WPlus ="WPlus" , B   ="B"   , Top   ="Top";
  static const std::string NuBar="NuBar", Lep   ="Lep"   , WMinus="WMinus", BBar="BBar", TopBar="TopBar"; 
}

namespace TtFullLepEvtPartons{
  // fully-leptonic parton names
  enum { B, BBar, Lepton, LeptonBar };
}

class TtFullLeptonicEvent: public TtEvent {
  
 public:

  TtFullLeptonicEvent(){};
  virtual ~TtFullLeptonicEvent(){};

  // access objects according to corresponding event hypothesis
  const reco::Candidate* top        (const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : eventHypo(key,cmb). daughter(TtFullLepDaughter::Top   ); };
  const reco::Candidate* b          (const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : top      (key,cmb)->daughter(TtFullLepDaughter::B     ); };
  const reco::Candidate* wPlus      (const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : top      (key,cmb)->daughter(TtFullLepDaughter::WPlus ); };
  const reco::Candidate* leptonBar  (const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : wPlus    (key,cmb)->daughter(TtFullLepDaughter::LepBar); };
  const reco::Candidate* neutrino   (const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : wPlus    (key,cmb)->daughter(TtFullLepDaughter::Nu    ); };
  const reco::Candidate* topBar     (const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : eventHypo(key,cmb). daughter(TtFullLepDaughter::TopBar); };
  const reco::Candidate* bBar       (const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : topBar   (key,cmb)->daughter(TtFullLepDaughter::BBar  ); };
  const reco::Candidate* wMinus     (const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : topBar   (key,cmb)->daughter(TtFullLepDaughter::WMinus); };
  const reco::Candidate* lepton     (const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : wMinus   (key,cmb)->daughter(TtFullLepDaughter::Lep   ); };
  const reco::Candidate* neutrinoBar(const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : wMinus   (key,cmb)->daughter(TtFullLepDaughter::NuBar ); };

  // access the matched gen particles
  const reco::GenParticle* genTop        () const { return (!genEvt_ ? 0 : this->genEvent()->top()        ); };
  const reco::GenParticle* genB          () const { return (!genEvt_ ? 0 : this->genEvent()->b()          ); };
  const reco::GenParticle* genWPlus      () const { return (!genEvt_ ? 0 : this->genEvent()->wPlus()      ); };
  const reco::GenParticle* genLeptonBar  () const { return (!genEvt_ ? 0 : this->genEvent()->leptonBar()  ); };
  const reco::GenParticle* genNeutrino   () const { return (!genEvt_ ? 0 : this->genEvent()->neutrino()   ); };
  const reco::GenParticle* genTopBar     () const { return (!genEvt_ ? 0 : this->genEvent()->topBar()     ); };
  const reco::GenParticle* genBBar       () const { return (!genEvt_ ? 0 : this->genEvent()->bBar()       ); };
  const reco::GenParticle* genWMinus     () const { return (!genEvt_ ? 0 : this->genEvent()->wMinus()     ); };
  const reco::GenParticle* genLepton     () const { return (!genEvt_ ? 0 : this->genEvent()->lepton()     ); };
  const reco::GenParticle* genNeutrinoBar() const { return (!genEvt_ ? 0 : this->genEvent()->neutrinoBar()); };
  
  void print();
};

#endif
