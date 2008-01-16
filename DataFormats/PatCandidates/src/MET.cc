//
// $Id: MET.cc,v 1.1 2008/01/15 12:59:32 lowette Exp $
//

#include "DataFormats/PatCandidates/interface/MET.h"


using namespace pat;


/// default constructor
MET::MET(){
}


/// constructor from METType
MET::MET(const METType & aMET) : PATObject<METType>(aMET) {
}


/// destructor
MET::~MET() {
}


/// return the generated MET from neutrinos
reco::Particle	MET::genMET() const {
  return (genMET_.size() > 0 ?
    genMET_.front() :
    reco::Particle(0, reco::Particle::LorentzVector(0, 0, 0, 0), reco::Particle::Point(0,0,0))
  );
}


/// method to set the generated MET
void MET::setGenMET(const reco::Particle & gm) {
  genMET_.clear();
  genMET_.push_back(gm);
}

