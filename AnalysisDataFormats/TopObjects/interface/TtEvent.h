#ifndef TopObjects_TtEvent_h
#define TopObjects_TtEvent_h

#include <vector>
#include <string>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefProd.h"

#include "DataFormats/Candidate/interface/CompositeCandidate.h"

#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"

////////////////////////////////////////////////////////////
// common base class for the TtSemiLeptonicEvent, the TtFullLeptonicEvent
// and the TtFullHadronicEvent (still to be implemented)
////////////////////////////////////////////////////////////

class TtEvent {

 public:

  // leptonic decay channels
  enum LepDecay {kNone, kElec, kMuon, kTau};

  // supported classes of event hypotheses
  enum HypoClassKey {kGeom, kWMassMaxSumPt, kMaxSumPtWMass, kGenMatch, kMVADisc, kKinFit};

  // pair of hypothesis' CompositeCandidate and corresponding JetLepComb
  typedef std::pair<reco::CompositeCandidate, std::vector<int> > HypoCombPair;
  
 public:

  TtEvent(){};
  virtual ~TtEvent(){};

  // access leptonic decay channels
  std::pair<LepDecay, LepDecay> lepDecays() const { return lepDecays_; }

  // access event hypothesis
  const reco::CompositeCandidate& eventHypo(const HypoClassKey& key, const unsigned& cmb=0) const { return (evtHyp_.find(key)->second)[cmb].first; };

  // access the TtGenEvent
  const edm::RefProd<TtGenEvent> & genEvent() const { return genEvt_; };

  // access meta information
  bool isHypoClassAvailable(const HypoClassKey& key) const { return (evtHyp_.find(key)!=evtHyp_.end()); };
  bool isHypoAvailable(const HypoClassKey& key, const unsigned& cmb=0) const { return isHypoClassAvailable(key) ? (cmb<evtHyp_.find(key)->second.size()) : false; };
  bool isHypoValid    (const HypoClassKey& key, const unsigned& cmb=0) const { return isHypoAvailable(key,cmb) ? !eventHypo(key,cmb).roles().empty() : false; };
  unsigned int numberOfAvailableHypoClasses() const { return evtHyp_.size();};
  unsigned int numberOfAvailableHypos(const HypoClassKey& key) const { return isHypoAvailable(key) ? evtHyp_.find(key)->second.size() : 0;};
  std::vector<int> jetLepComb(const HypoClassKey& key, const unsigned& cmb=0) const { return (evtHyp_.find(key)->second)[cmb].second; };
  double genMatchSumPt(const unsigned& cmb=0) const { return (cmb<genMatchSumPt_.size() ? genMatchSumPt_[cmb] : -1.); };
  double genMatchSumDR(const unsigned& cmb=0) const { return (cmb<genMatchSumDR_.size() ? genMatchSumDR_[cmb] : -1.); };
  std::string mvaMethod() const { return mvaMethod_; }
  double mvaDisc(const unsigned& cmb=0) const { return (cmb<mvaDisc_.size() ? mvaDisc_[cmb] : -1.); }
  double fitChi2(const unsigned& cmb=0) const { return (cmb<fitChi2_.size() ? fitChi2_[cmb] : -1.); }
  double fitProb(const unsigned& cmb=0) const { return (cmb<fitProb_.size() ? fitProb_[cmb] : -1.); }

  int correspondingHypo(const HypoClassKey& key1, const unsigned& hyp1, const HypoClassKey& key2) const;

 public:

  // set leptonic decay channels
  void setLepDecays(const LepDecay& lepDecTop1, const LepDecay& lepDecTop2) { lepDecays_=std::make_pair(lepDecTop1, lepDecTop2); };

  // set the generated event
  void setGenEvent(const edm::Handle<TtGenEvent>& evt) { genEvt_=edm::RefProd<TtGenEvent>(evt); };

  // add EventHypotheses
  void addEventHypo(const HypoClassKey& key, HypoCombPair hyp) { evtHyp_[key].push_back(hyp); };
  
  // set meta information
  void setGenMatchSumPt(const std::vector<double>& val) {genMatchSumPt_=val;};
  void setGenMatchSumDR(const std::vector<double>& val) {genMatchSumDR_=val;};
  void setMvaMethod(const std::string& name) { mvaMethod_=name; };
  void setMvaDiscriminators(const std::vector<double>& val) { mvaDisc_=val; };
  void setFitChi2(const std::vector<double>& val) { fitChi2_=val; };
  void setFitProb(const std::vector<double>& val) { fitProb_=val; };

 protected:

  // event content
  std::pair<LepDecay, LepDecay> lepDecays_;
  edm::RefProd<TtGenEvent> genEvt_;
  std::map<HypoClassKey, std::vector<HypoCombPair> > evtHyp_;
  
  //meta information
  std::vector<double> fitChi2_;         // result of kinematic fit
  std::vector<double> fitProb_;         // result of kinematic fit
  std::vector<double> genMatchSumPt_;   // result of gen match
  std::vector<double> genMatchSumDR_;   // result of gen match
  std::string mvaMethod_;               // result of MVA
  std::vector<double> mvaDisc_;         // result of MVA 
};

#endif
