//
// $Id: TopMET.cc,v 1.5 2007/07/05 23:50:00 lowette Exp $
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
reco::Particle	TopMET::getGenMET() const {
  return (genMET_.size() > 0 ?
    genMET_.front() :
    reco::Particle(0, reco::Particle::LorentzVector(0, 0, 0, 0), reco::Particle::Point(0,0,0))
  );
}


/// method to set the generated MET
void TopMET::setGenMET(const reco::Particle & gm) {
  genMET_.clear();
  genMET_.push_back(gm);
}
