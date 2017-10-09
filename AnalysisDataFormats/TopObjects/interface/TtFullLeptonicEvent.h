#ifndef TopObjects_TtFullLeptonicEvent_h
#define TopObjects_TtFullLeptonicEvent_h

#include "AnalysisDataFormats/TopObjects/interface/TtEvent.h"

namespace TtFullLepDaughter{
  /// full leptonic daughter names for common
  /// use and use with the hypotheses
  static const std::string Nu   ="Nu"   , LepBar="LepBar", WPlus ="WPlus" , B   ="B"   , Top   ="Top";
  static const std::string NuBar="NuBar", Lep   ="Lep"   , WMinus="WMinus", BBar="BBar", TopBar="TopBar"; 
}

/**
   \class   TtFullLeptonicEvent TtFullLeptonicEvent.h "AnalysisDataFormats/TopObjects/interface/TtFullLeptonicEvent.h"

   \brief   Class derived from the TtEvent for the full leptonic decay channel

   The structure holds information on the leptonic decay channels, 
   all event hypotheses of different classes (user defined during
   production) and a reference to the TtGenEvent (if available). It 
   provides access and administration.
*/

class TtFullLeptonicEvent: public TtEvent {
  
 public:
  /// empty constructor
  TtFullLeptonicEvent(){};
  /// default destructor
  virtual ~TtFullLeptonicEvent(){};

  /// get top of the given hypothesis
  const reco::Candidate* top(const std::string& key, const unsigned& cmb=0) const { return top(hypoClassKeyFromString(key), cmb); };
  /// get top of the given hypothesis
  const reco::Candidate* top(const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : eventHypo(key,cmb). daughter(TtFullLepDaughter::Top); };
  /// get b of the given hypothesis
  const reco::Candidate* b(const std::string& key, const unsigned& cmb=0) const { return b(hypoClassKeyFromString(key), cmb); };
  /// get b of the given hypothesis
  const reco::Candidate* b(const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : top(key,cmb)->daughter(TtFullLepDaughter::B); };
  /// get Wplus of the given hypothesis
  const reco::Candidate* wPlus(const std::string& key, const unsigned& cmb=0) const { return wPlus(hypoClassKeyFromString(key), cmb); };
  /// get Wplus of the given hypothesis
  const reco::Candidate* wPlus(const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : top(key,cmb)->daughter(TtFullLepDaughter::WPlus); };
  /// get anti-lepton of the given hypothesis
  const reco::Candidate* leptonBar(const std::string& key, const unsigned& cmb=0) const { return leptonBar(hypoClassKeyFromString(key), cmb); };
  /// get anti-lepton of the given hypothesis
  const reco::Candidate* leptonBar(const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : wPlus(key,cmb)->daughter(TtFullLepDaughter::LepBar); };
  /// get neutrino of the given hypothesis
  const reco::Candidate* neutrino(const std::string& key, const unsigned& cmb=0) const { return neutrino(hypoClassKeyFromString(key), cmb); };
  /// get neutrino of the given hypothesis
  const reco::Candidate* neutrino(const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : wPlus(key,cmb)->daughter(TtFullLepDaughter::Nu    ); };
  /// get anti-top of the given hypothesis
  const reco::Candidate* topBar(const std::string& key, const unsigned& cmb=0) const { return topBar(hypoClassKeyFromString(key), cmb); };
  /// get anti-top of the given hypothesis
  const reco::Candidate* topBar(const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : eventHypo(key,cmb). daughter(TtFullLepDaughter::TopBar); };
  /// get anti-b of the given hypothesis
  const reco::Candidate* bBar(const std::string& key, const unsigned& cmb=0) const { return bBar(hypoClassKeyFromString(key), cmb); };
  /// get anti-b of the given hypothesis
  const reco::Candidate* bBar(const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : topBar(key,cmb)->daughter(TtFullLepDaughter::BBar  ); };
  /// get Wminus of the given hypothesis
  const reco::Candidate* wMinus(const std::string& key, const unsigned& cmb=0) const { return wMinus(hypoClassKeyFromString(key), cmb); };
  /// get Wminus of the given hypothesis
  const reco::Candidate* wMinus(const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : topBar(key,cmb)->daughter(TtFullLepDaughter::WMinus); };
  /// get lepton of the given hypothesis
  const reco::Candidate* lepton(const std::string& key, const unsigned& cmb=0) const { return lepton(hypoClassKeyFromString(key), cmb); };
  /// get lepton of the given hypothesis
  const reco::Candidate* lepton(const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : wMinus(key,cmb)->daughter(TtFullLepDaughter::Lep   ); };
  /// get anti-neutrino of the given hypothesis
  const reco::Candidate* neutrinoBar(const std::string& key, const unsigned& cmb=0) const { return neutrinoBar(hypoClassKeyFromString(key), cmb); };
  /// get anti-neutrino of the given hypothesis
  const reco::Candidate* neutrinoBar(const HypoClassKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : wMinus   (key,cmb)->daughter(TtFullLepDaughter::NuBar ); };

  /// get top of the TtGenEvent
  const reco::GenParticle* genTop        () const { return (!genEvt_ ? 0 : this->genEvent()->top()        ); };
  /// get b of the TtGenEvent
  const reco::GenParticle* genB          () const { return (!genEvt_ ? 0 : this->genEvent()->b()          ); };
  /// get Wplus of the TtGenEvent
  const reco::GenParticle* genWPlus      () const { return (!genEvt_ ? 0 : this->genEvent()->wPlus()      ); };
  /// get anti-lepton of the TtGenEvent
  const reco::GenParticle* genLeptonBar  () const { return (!genEvt_ ? 0 : this->genEvent()->leptonBar()  ); };
  /// get neutrino of the TtGenEvent
  const reco::GenParticle* genNeutrino   () const { return (!genEvt_ ? 0 : this->genEvent()->neutrino()   ); };
  /// get anti-top of the TtGenEvent
  const reco::GenParticle* genTopBar     () const { return (!genEvt_ ? 0 : this->genEvent()->topBar()     ); };
  /// get anti-b of the TtGenEvent
  const reco::GenParticle* genBBar       () const { return (!genEvt_ ? 0 : this->genEvent()->bBar()       ); };
  /// get Wminus of the TtGenEvent
  const reco::GenParticle* genWMinus     () const { return (!genEvt_ ? 0 : this->genEvent()->wMinus()     ); };
  /// get lepton of the TtGenEvent
  const reco::GenParticle* genLepton     () const { return (!genEvt_ ? 0 : this->genEvent()->lepton()     ); };
  /// get anti-neutrino of the TtGenEvent
  const reco::GenParticle* genNeutrinoBar() const { return (!genEvt_ ? 0 : this->genEvent()->neutrinoBar()); };

  /// return the weight of the kinematic solution of hypothesis 'cmb' if available; -1 else
  double solWeight(const unsigned& cmb=0) const { return (cmb<solWeight_.size() ? solWeight_[cmb] : -1.); }    
  /// return if the kinematic solution of hypothesis 'cmb' is right or wrong charge if available; -1 else
  bool isWrongCharge() const { return wrongCharge_; }

  /// set weight of kKinSolution hypothesis
  void setSolWeight(const std::vector<double>& val) { solWeight_=val; }; 
  /// set right or wrong charge combination of kKinSolution hypothesis
  void setWrongCharge(const bool& val) { wrongCharge_=val; }; 

  /// print full content of the structure as formated 
  /// LogInfo to the MessageLogger output for debugging  
  void print(const int verbosity=1) const;

 protected:

  /// result of kinematic solution
  std::vector<double> solWeight_; 
  /// right/wrong charge booleans
  bool wrongCharge_;

};

#endif
