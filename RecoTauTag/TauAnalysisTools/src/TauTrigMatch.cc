
#include "RecoTauTag/TauAnalysisTools/interface/TauTrigMatch.h"
#include <TLorentzVector.h>
#include "Math/GenVector/LorentzVector.h"

TauTrigMatch::TauTrigMatch(const pat::Tau* tagTau, std::vector<const reco::Candidate*>* trigObj, const reco::Candidate* GenTauMatch ,unsigned int index, unsigned int nTotalObjects):
  tagTau_(tagTau), trigObj_(trigObj),GenTauMatch_(GenTauMatch),index_(index), nTotalObjects_(nTotalObjects){
  
        // Create a dummy reco::Candidate Object with unrealistic LorentzVector values as a default output to return in case of a failed matching.  
        dummyCandidate_ = dynamic_cast<reco::Candidate* >( tagTau->clone());
        math::XYZTLorentzVector *v = new math::XYZTLorentzVector();
        v->SetPxPyPzE(-999.,-999.,-9999.,-999.);
        dummyCandidate_->setP4(((const math::XYZTLorentzVector)*v)); 
  }

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

const reco::Candidate* TauTrigMatch::GenTauMatch() const {
  if(GenTauMatch_ != NULL) return GenTauMatch_;
  else return dummyCandidate_; // Careful! Method return dummy object to ensure successfull termination of program. Only use GenTauMatch values if "bool TauTrigMatch::GenHadTauMatch()" returns "true"
}

const reco::Candidate* TauTrigMatch::GenTauJet() const {
  if(tagTau_->genJet() != NULL) return tagTau_->genJet();
  else return dummyCandidate_; // Careful!  Method return dummy object to ensure successfull termination of program. Only use GenTauJet values if "bool TauTrigMatch::GenHadTauMatch()" returns "true"

}

bool TauTrigMatch::GenTauMatchTest() const {
  return GenTauMatch_ != NULL;
}

bool TauTrigMatch::GenHadTauMatch() const {
  return tagTau_->genJet() != NULL;
}

bool TauTrigMatch::trigObjMatch(int a) const {
  return trigObj_->at(a) != NULL;
}

bool TauTrigMatch::tagTauID(std::string DiscriminatorName) const{

 return tagTau_->tauID(DiscriminatorName);  

}    

