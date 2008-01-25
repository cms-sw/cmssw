//
// $Id: TopMET.cc,v 1.6 2007/07/17 13:46:16 yumiceva Exp $
//

#include "AnalysisDataFormats/TopObjects/interface/TopMET.h"


/// default constructor
TopMET::TopMET(){
}


/// constructor from TopMETType
TopMET::TopMET(const TopMETType & aMET) : TopObject<TopMETType>(aMET) {
}


/// destructor
TopMET::~TopMET() {
}


/// return the generated MET from neutrinos
reco::GenParticle	TopMET::getGenMET() const {
  return (genMET_.size() > 0 ?
    genMET_.front() :
    reco::GenParticle(0, reco::GenParticle::LorentzVector(0, 0, 0, 0), reco::GenParticle::Point(0,0,0), 0, 0, true)
  );
}


/// method to set the generated MET
void TopMET::setGenMET(const reco::GenParticle & gm) {
  genMET_.clear();
  genMET_.push_back(gm);
}
