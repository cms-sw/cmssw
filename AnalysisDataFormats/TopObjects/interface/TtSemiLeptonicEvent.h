#ifndef TopObjects_TtSemiLeptonicEvent_h
#define TopObjects_TtSemiLeptonicEvent_h

#include <vector>
#include <string>

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefProd.h"

#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"

#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/Candidate/interface/ShallowClonePtrCandidate.h"

namespace TtSemiDaughter{
  // semileptonic daughter names
  static const std::string Nu  ="Nu",   Lep ="Lep",  LepW="LepW", LepB="LepB", LepTop="LepTop";
  static const std::string HadQ="HadQ", HadP="HadP", HadW="HadW", HadB="HadB", HadTop="HadTop"; 
}

namespace TtSemiLepEvtPartons{
  // semileptonic parton names
  enum { LightQ, LightQBar, HadB, LepB};
}

class TtSemiLeptonicEvent {
  
 public:

  // semiletponic decay channels
  enum Decay {kNone, kElec, kMuon, kTau};

  // supported EventHypotheses
  enum HypoKey {kGeom, kWMassMaxSumPt, kMaxSumPtWMass, kGenMatch, kMVADisc, kKinFit};

  // pair of hypothesis' CompositeCandidate and corresponding JetComb
  typedef std::pair<reco::CompositeCandidate, std::vector<int> > HypoCombPair;

 public:

  TtSemiLeptonicEvent();
  virtual ~TtSemiLeptonicEvent(){};

  // access decay 
  Decay decay() const { return decay_;}

  // access objects according to corresponding event hypothesis
  const reco::CompositeCandidate& eventHypo(const HypoKey& key, const unsigned& cmb=0) const { return (evtHyp_.find(key)->second)[cmb].first; };
  const reco::Candidate* hadronicTop(const HypoKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : eventHypo  (key,cmb). daughter(TtSemiDaughter::HadTop); };
  const reco::Candidate* hadronicB  (const HypoKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : hadronicTop(key,cmb)->daughter(TtSemiDaughter::HadB  ); };
  const reco::Candidate* hadronicW  (const HypoKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : hadronicTop(key,cmb)->daughter(TtSemiDaughter::HadW  ); };
  const reco::Candidate* lightQuarkP(const HypoKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : hadronicW  (key,cmb)->daughter(TtSemiDaughter::HadP  ); };
  const reco::Candidate* lightQuarkQ(const HypoKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : hadronicW  (key,cmb)->daughter(TtSemiDaughter::HadQ  ); };
  const reco::Candidate* leptonicTop(const HypoKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : eventHypo  (key,cmb). daughter(TtSemiDaughter::LepTop); };
  const reco::Candidate* leptonicB  (const HypoKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : leptonicTop(key,cmb)->daughter(TtSemiDaughter::LepB  ); };
  const reco::Candidate* leptonicW  (const HypoKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : leptonicTop(key,cmb)->daughter(TtSemiDaughter::LepW  ); };
  const reco::Candidate* neutrino   (const HypoKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : leptonicW  (key,cmb)->daughter(TtSemiDaughter::Nu    ); };
  const reco::Candidate* lepton     (const HypoKey& key, const unsigned& cmb=0) const { return !isHypoValid(key,cmb) ? 0 : leptonicW  (key,cmb)->daughter(TtSemiDaughter::Lep   ); };

  // access the matched gen particles
  const edm::RefProd<TtGenEvent> & genEvent() const { return genEvt_; };
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
  
  // access meta information
  bool isHypoAvailable(const HypoKey& key) const { return (evtHyp_.find(key)!=evtHyp_.end()); };
  bool isHypoAvailable(const HypoKey& key, const unsigned& cmb) const { return isHypoAvailable(key) ? (cmb<evtHyp_.find(key)->second.size()) : false; };
  bool isHypoValid    (const HypoKey& key, const unsigned& cmb=0) const { return isHypoAvailable(key,cmb) ? !eventHypo(key,cmb).roles().empty() : false; };
  unsigned int numberOfAvailableHypos() const { return evtHyp_.size();};
  unsigned int numberOfAvailableCombs(const HypoKey& key) const { return isHypoAvailable(key) ? evtHyp_.find(key)->second.size() : 0;};
  std::vector<int> jetMatch(const HypoKey& key, const unsigned& cmb=0) const { return (evtHyp_.find(key)->second)[cmb].second; };
  double genMatchSumPt(const unsigned& cmb=0) const { return (cmb<genMatchSumPt_.size() ? genMatchSumPt_[cmb] : -1.); };
  double genMatchSumDR(const unsigned& cmb=0) const { return (cmb<genMatchSumDR_.size() ? genMatchSumDR_[cmb] : -1.); };
  std::string mvaMethod() const { return mvaMethod_; }
  double mvaDisc(const unsigned& cmb=0) const { return (cmb<mvaDisc_.size() ? mvaDisc_[cmb] : -1.); }
  double fitChi2(const unsigned& cmb=0) const { return (cmb<fitChi2_.size() ? fitChi2_[cmb] : -1.); }
  double fitProb(const unsigned& cmb=0) const { return (cmb<fitProb_.size() ? fitProb_[cmb] : -1.); }

  int correspondingJetMatch(const HypoKey& key1, const unsigned& cmb1, const HypoKey& key2) const;

  void print();

 public:

  // set decay
  void setDecay(const Decay& dec) { decay_=dec; };

  // set the generated event
  void setGenEvent(const edm::Handle<TtGenEvent>& evt) { genEvt_=edm::RefProd<TtGenEvent>(evt); };

  // add EventHypotheses
  void addEventHypo(const HypoKey& key, HypoCombPair hyp) { evtHyp_[key].push_back(hyp); };
  
  // set meta information
  void setGenMatchSumPt(const std::vector<double>& val) {genMatchSumPt_=val;};
  void setGenMatchSumDR(const std::vector<double>& val) {genMatchSumDR_=val;};
  void setMvaMethod(const std::string& name) { mvaMethod_=name; };
  void setMvaDiscriminators(const std::vector<double>& val) { mvaDisc_=val; };
  void setFitChi2(const std::vector<double>& val) { fitChi2_=val; };
  void setFitProb(const std::vector<double>& val) { fitProb_=val; };

 private:

  // event content
  Decay decay_;
  edm::RefProd<TtGenEvent> genEvt_;
  std::map<HypoKey, std::vector<HypoCombPair> > evtHyp_;
  
  //meta information
  std::vector<double> fitChi2_;         // result of kinematic fit
  std::vector<double> fitProb_;         // result of kinematic fit
  std::vector<double> genMatchSumPt_;   // result of gen match
  std::vector<double> genMatchSumDR_;   // result of gen match
  std::string mvaMethod_;               // result of MVA
  std::vector<double> mvaDisc_;         // result of MVA
};

#endif
