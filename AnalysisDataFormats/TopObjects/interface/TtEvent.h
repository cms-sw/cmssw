#ifndef TopObjects_TtEvent_h
#define TopObjects_TtEvent_h

#include <vector>
#include <string>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"

/**
   \class   TtEvent TtEvent.h "AnalysisDataFormats/TopObjects/interface/TtEvent.h"

   \brief   Base class to hold information for ttbar event interpretation

   The structure holds information for ttbar event interpretation.
   All event hypotheses of different classes (user defined during
   production) and a reference to the TtGenEvent (if available). It 
   provides access and administration.
*/

namespace edm{
  class LogInfo;
}

class TtEvent {

 public:
  /// supported classes of event hypotheses
  enum HypoClassKey {kGeom, kWMassMaxSumPt, kMaxSumPtWMass, kGenMatch, kMVADisc, kKinFit, kKinSolution, kWMassDeltaTopMass};
  /// pair of hypothesis and lepton jet combinatorics for a given hypothesis
  typedef std::pair<reco::CompositeCandidate, std::vector<int> > HypoCombPair;

 protected:
   /// a lightweight map for selection type string label and enum value
   struct HypoClassKeyStringToEnum { const char* label; HypoClassKey value; };
   /// return the corresponding enum value from a string 
   HypoClassKey hypoClassKeyFromString(const std::string& label) const;
  
 public:
  /// empty constructor
  TtEvent(){};
  /// default destructor
  virtual ~TtEvent(){};

  /// get leptonic decay channels
  std::pair<WDecay::LeptonType, WDecay::LeptonType> lepDecays() const { return lepDecays_; }
  /// get event hypothesis; there can be more hypotheses of a certain 
  /// class (sorted by quality); per default the best hypothesis is returned
  const reco::CompositeCandidate& eventHypo(const HypoClassKey& key, const unsigned& cmb=0) const { return (evtHyp_.find(key)->second)[cmb].first; };
  /// get TtGenEvent
  const edm::RefProd<TtGenEvent>& genEvent() const { return genEvt_; };

  /// check if hypothesis class 'key' was added to the event structure
  bool isHypoClassAvailable(const std::string& key) const { return isHypoClassAvailable( hypoClassKeyFromString(key) ); };
  /// check if hypothesis class 'key' was added to the event structure
  bool isHypoClassAvailable(const HypoClassKey& key) const { return (evtHyp_.find(key)!=evtHyp_.end()); };
  // check if hypothesis 'cmb' is available within the hypothesis class
  bool isHypoAvailable(const std::string& key, const unsigned& cmb=0) const { return isHypoAvailable( hypoClassKeyFromString(key), cmb ); };
  /// check if hypothesis 'cmb' is available within the hypothesis class
  bool isHypoAvailable(const HypoClassKey& key, const unsigned& cmb=0) const { return isHypoClassAvailable(key) ? (cmb<evtHyp_.find(key)->second.size()) : false; };
  /// check if hypothesis 'cmb' within the hypothesis class was valid; if not it lead to an empty CompositeCandidate
  bool isHypoValid(const std::string& key, const unsigned& cmb=0) const { return isHypoValid( hypoClassKeyFromString(key), cmb ); };
  /// check if hypothesis 'cmb' within the hypothesis class was valid; if not it lead to an empty CompositeCandidate
  bool isHypoValid(const HypoClassKey& key, const unsigned& cmb=0) const { return isHypoAvailable(key, cmb) ? !eventHypo(key, cmb).roles().empty() : false; };
  /// return number of available hypothesis classes
  unsigned int numberOfAvailableHypoClasses() const { return evtHyp_.size(); };
  /// return number of available hypotheses within a given hypothesis class
  unsigned int numberOfAvailableHypos(const std::string& key) const { return numberOfAvailableHypos( hypoClassKeyFromString(key) ); };
  /// return number of available hypotheses within a given hypothesis class
  unsigned int numberOfAvailableHypos(const HypoClassKey& key) const { return isHypoAvailable(key) ? evtHyp_.find(key)->second.size() : 0; };
  /// return the vector of jet lepton combinatorics for a given hypothesis and class
  std::vector<int> jetLeptonCombination(const std::string& key, const unsigned& cmb=0) const { return jetLeptonCombination(hypoClassKeyFromString(key), cmb); };
  /// return the vector of jet lepton combinatorics for a given hypothesis and class
  std::vector<int> jetLeptonCombination(const HypoClassKey& key, const unsigned& cmb=0) const { return (evtHyp_.find(key)->second)[cmb].second; };
  /// return the sum pt of the generator match if available; -1 else
  double genMatchSumPt(const unsigned& cmb=0) const { return (cmb<genMatchSumPt_.size() ? genMatchSumPt_[cmb] : -1.); };
  /// return the sum dr of the generator match if available; -1 else
  double genMatchSumDR(const unsigned& cmb=0) const { return (cmb<genMatchSumDR_.size() ? genMatchSumDR_[cmb] : -1.); };
  /// return the label of the mva method in use for the jet parton association (if kMVADisc is not available the string is empty)
  std::string mvaMethod() const { return mvaMethod_; }
  /// return the mva discriminant value of hypothesis 'cmb' if available; -1 else
  double mvaDisc(const unsigned& cmb=0) const { return (cmb<mvaDisc_.size() ? mvaDisc_[cmb] : -1.); }
  /// return the chi2 of the kinemtaic fit of hypothesis 'cmb' if available; -1 else
  double fitChi2(const unsigned& cmb=0) const { return (cmb<fitChi2_.size() ? fitChi2_[cmb] : -1.); }
  /// return the fit probability of hypothesis 'cmb' if available; -1 else
  double fitProb(const unsigned& cmb=0) const { return (cmb<fitProb_.size() ? fitProb_[cmb] : -1.); }
  /// return the hypothesis in hypothesis class 'key2', which corresponds to hypothesis 'hyp1' in hypothesis class 'key1'
  int correspondingHypo(const std::string& key1, const unsigned& hyp1, const std::string& key2) const { return correspondingHypo(hypoClassKeyFromString(key1), hyp1, hypoClassKeyFromString(key2) ); };
  /// return the hypothesis in hypothesis class 'key2', which corresponds to hypothesis 'hyp1' in hypothesis class 'key1'
  int correspondingHypo(const HypoClassKey& key1, const unsigned& hyp1, const HypoClassKey& key2) const;

  /// print pt, eta, phi, mass of a given candidate into an existing LogInfo
  void printParticle(edm::LogInfo &log, const char* name, const reco::Candidate* cand) const;

  /// set leptonic decay channels
  void setLepDecays(const WDecay::LeptonType& lepDecTop1, const WDecay::LeptonType& lepDecTop2) { lepDecays_=std::make_pair(lepDecTop1, lepDecTop2); };
  /// set TtGenEvent
  void setGenEvent(const edm::Handle<TtGenEvent>& evt) { genEvt_=edm::RefProd<TtGenEvent>(evt); };
  /// add new hypotheses
  void addEventHypo(const HypoClassKey& key, HypoCombPair hyp) { evtHyp_[key].push_back(hyp); };
  /// set sum pt of kGenMatch hypothesis
  void setGenMatchSumPt(const std::vector<double>& val) {genMatchSumPt_=val;};
  /// set sum dr of kGenMatch hypothesis
  void setGenMatchSumDR(const std::vector<double>& val) {genMatchSumDR_=val;};
  /// set label of mva method for kMVADisc hypothesis
  void setMvaMethod(const std::string& name) { mvaMethod_=name; };
  /// set mva discriminant values of kMVADisc hypothesis
  void setMvaDiscriminators(const std::vector<double>& val) { mvaDisc_=val; };
  /// set chi2 of kKinFit hypothesis
  void setFitChi2(const std::vector<double>& val) { fitChi2_=val; };
  /// set fit probability of kKinFit hypothesis
  void setFitProb(const std::vector<double>& val) { fitProb_=val; };
  
 protected:

  /// leptonic decay channels
   std::pair<WDecay::LeptonType, WDecay::LeptonType> lepDecays_;
  /// reference to TtGenEvent (has to be kept in the event!)
  edm::RefProd<TtGenEvent> genEvt_;
  /// map of hypotheses; for each HypoClassKey a vector of 
  /// hypothesis and their lepton jet combinatorics are kept
  std::map<HypoClassKey, std::vector<HypoCombPair> > evtHyp_;
  
  /// result of kinematic fit
  std::vector<double> fitChi2_;        
  /// result of kinematic fit
  std::vector<double> fitProb_; 
  /// result of gen match
  std::vector<double> genMatchSumPt_;  
  /// result of gen match
  std::vector<double> genMatchSumDR_;   
  /// label of the MVA method
  std::string mvaMethod_;               
  /// MVA discriminants
  std::vector<double> mvaDisc_;         
};

#endif
