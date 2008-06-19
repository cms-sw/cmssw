//
// $Id: MET.cc,v 1.7 2008/06/03 22:28:07 gpetrucc Exp $
//

#include "DataFormats/PatCandidates/interface/MET.h"


using namespace pat;


/// default constructor
MET::MET(){
}


/// constructor from METType
MET::MET(const METType & aMET) : PATObject<METType>(aMET) {
    const reco::CaloMET * calo = dynamic_cast<const reco::CaloMET *>(&aMET);
    if (calo != 0) caloMET_.push_back(calo->getSpecific());
}


/// constructor from ref to METType
MET::MET(const edm::RefToBase<METType> & aMETRef) : PATObject<METType>(aMETRef) {
    const reco::CaloMET * calo = dynamic_cast<const reco::CaloMET *>(aMETRef.get());
    if (calo != 0) caloMET_.push_back(calo->getSpecific());
}

/// constructor from ref to METType
MET::MET(const edm::Ptr<METType> & aMETRef) : PATObject<METType>(aMETRef) {
    const reco::CaloMET * calo = dynamic_cast<const reco::CaloMET *>(aMETRef.get());
    if (calo != 0) caloMET_.push_back(calo->getSpecific());
}


/// destructor
MET::~MET() {
}


/// return the generated MET from neutrinos
const reco::GenMET * MET::genMET() const {
  return (genMET_.size() > 0 ? &genMET_.front() : 0 );
}


/// method to set the generated MET
void MET::setGenMET(const reco::GenMET & gm) {
  genMET_.clear();
  genMET_.push_back(gm);
}

