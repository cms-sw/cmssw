//
// $Id: MET.cc,v 1.3 2008/01/21 22:01:57 lowette Exp $
//

#include "DataFormats/PatCandidates/interface/MET.h"


using namespace pat;


/// default constructor
MET::MET(){
}


/// constructor from METType
MET::MET(const METType & aMET) : PATObject<METType>(aMET) {
}


/// constructor from ref to METType
MET::MET(const edm::Ref<std::vector<METType> > & aMETRef) : PATObject<METType>(aMETRef) {
}


/// destructor
MET::~MET() {
}


/// return the generated MET from neutrinos
const reco::Particle * MET::genMET() const {
  return (genMET_.size() > 0 ? &genMET_.front() : 0 );
}


/// method to set the generated MET
void MET::setGenMET(const reco::Particle & gm) {
  genMET_.clear();
  genMET_.push_back(gm);
}

