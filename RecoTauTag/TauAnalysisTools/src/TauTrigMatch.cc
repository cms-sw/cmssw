
#include "RecoTauTag/TauAnalysisTools/interface/TauTrigMatch.h"

TauTrigMatch::TauTrigMatch(const pat::Tau* tagTau, std::vector<const reco::Candidate*>* trigObj ,unsigned int index, unsigned int nTotalObjects):
  tagTau_(tagTau), trigObj_(trigObj),index_(index), nTotalObjects_(nTotalObjects){}

  
TauTrigMatch::TauTrigMatch(){}

unsigned int TauTrigMatch::index() const {
  return index_;
}

unsigned int TauTrigMatch::nTotalObjects() const {
  return nTotalObjects_;
}

const pat::Tau* TauTrigMatch::tagTau() const {
  return tagTau_;
}

bool TauTrigMatch::trigObjMatch(int a) const {
  return trigObj_->at(a) != NULL;
}

bool TauTrigMatch::tagTauID(std::string DiscriminatorName) const{

 return tagTau_->tauID(DiscriminatorName);  

}    

