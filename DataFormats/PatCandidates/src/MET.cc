//
// $Id: MET.cc,v 1.16 2013/02/19 16:18:45 vadler Exp $
//

#include "DataFormats/PatCandidates/interface/MET.h"


using namespace pat;


/// default constructor
MET::MET(): uncorInfo_(0) {
}


/// constructor from reco::MET
MET::MET(const reco::MET & aMET) : PATObject<reco::MET>(aMET), uncorInfo_(0) {
    const reco::CaloMET * calo = dynamic_cast<const reco::CaloMET *>(&aMET);
    if (calo != 0) caloMET_.push_back(calo->getSpecific());
    const reco::PFMET * pf = dynamic_cast<const reco::PFMET *>(&aMET);
    if (pf != 0) pfMET_.push_back(pf->getSpecific());
}


/// constructor from ref to reco::MET
MET::MET(const edm::RefToBase<reco::MET> & aMETRef) : PATObject<reco::MET>(aMETRef), uncorInfo_(0) {
    const reco::CaloMET * calo = dynamic_cast<const reco::CaloMET *>(aMETRef.get());
    if (calo != 0) caloMET_.push_back(calo->getSpecific());
    const reco::PFMET * pf = dynamic_cast<const reco::PFMET *>(aMETRef.get());
    if (pf != 0) pfMET_.push_back(pf->getSpecific());
}

/// constructor from ref to reco::MET
MET::MET(const edm::Ptr<reco::MET> & aMETRef) : PATObject<reco::MET>(aMETRef), uncorInfo_(0) {
    const reco::CaloMET * calo = dynamic_cast<const reco::CaloMET *>(aMETRef.get());
    if (calo != 0) caloMET_.push_back(calo->getSpecific());
    const reco::PFMET * pf = dynamic_cast<const reco::PFMET *>(aMETRef.get());
    if (pf != 0) pfMET_.push_back(pf->getSpecific());
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

//! return uncorrrection related stuff
unsigned int MET::nCorrections() const { checkUncor_(); return nCorrections_; }

float MET::corEx(UncorrectionType ix) const {
  if (ix == uncorrNONE) return 0;
  checkUncor_(); return uncorInfo_[ix].corEx;
}
float MET::corEy(UncorrectionType ix) const {
  if (ix == uncorrNONE) return 0;
  checkUncor_(); return uncorInfo_[ix].corEy;
}
float MET::corSumEt(UncorrectionType ix) const {
  if (ix == uncorrNONE) return 0;
  checkUncor_(); return uncorInfo_[ix].corSumEt;
}
float MET::uncorrectedPt(UncorrectionType ix) const {
  if (ix == uncorrNONE) return pt();
  checkUncor_(); return uncorInfo_[ix].pt;
}
float MET::uncorrectedPhi(UncorrectionType ix) const {
  if (ix == uncorrNONE) return phi();
  checkUncor_(); return uncorInfo_[ix].phi;
}


//! check and set transients
void MET::checkUncor_() const {
  if (uncorInfo_.size() == uncorrMAXN && oldPt_ == pt() ) return;

  oldPt_ = pt();
  std::vector<CorrMETData> corrs(mEtCorr());
  nCorrections_ = corrs.size();

  uncorInfo_.resize(uncorrMAXN);
  UncorrectionType ix;

  //! ugly
  //! ALL
  ix = uncorrALL;
  uncorInfo_[ix] = UncorInfo();
  for (unsigned int iC=0; iC < nCorrections_; ++iC){
    uncorInfo_[ix].corEx +=    corrs[iC].mex;
    uncorInfo_[ix].corEy +=    corrs[iC].mey;
    uncorInfo_[ix].corSumEt += corrs[iC].sumet;
  }
  setPtPhi_(uncorInfo_[ix]);

  //! JES
  ix = uncorrJES;
  uncorInfo_[ix] = UncorInfo();
  if (nCorrections_ >=1 ){
    unsigned int iC = 0;
    uncorInfo_[ix].corEx +=    corrs[iC].mex;
    uncorInfo_[ix].corEy +=    corrs[iC].mey;
    uncorInfo_[ix].corSumEt += corrs[iC].sumet;
  }
  setPtPhi_(uncorInfo_[ix]);

  //! MUON
  ix = uncorrMUON;
  uncorInfo_[ix] = UncorInfo();
  if (nCorrections_ >=2 ){
    unsigned int iC = 1;
    uncorInfo_[ix].corEx +=    corrs[iC].mex;
    uncorInfo_[ix].corEy +=    corrs[iC].mey;
    uncorInfo_[ix].corSumEt += corrs[iC].sumet;
  }
  setPtPhi_(uncorInfo_[ix]);

  //! TAU
  ix = uncorrTAU;
  uncorInfo_[ix] = UncorInfo();
  if (nCorrections_ >=3 ){
    unsigned int iC = 2;
    uncorInfo_[ix].corEx +=    corrs[iC].mex;
    uncorInfo_[ix].corEy +=    corrs[iC].mey;
    uncorInfo_[ix].corSumEt += corrs[iC].sumet;
  }
  setPtPhi_(uncorInfo_[ix]);

}

void MET::setPtPhi_(UncorInfo& uci) const {
  float lpx = px() - uci.corEx;
  float lpy = py() - uci.corEy;
  uci.pt = sqrt(lpx*lpx + lpy*lpy);
  uci.phi = atan2(lpy, lpx);
}
