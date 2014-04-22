//
//

#include "DataFormats/PatCandidates/interface/MET.h"


using namespace pat;


/// default constructor
MET::MET(): uncorInfo_(nullptr) {
}


/// constructor from reco::MET
MET::MET(const reco::MET & aMET) : PATObject<reco::MET>(aMET), uncorInfo_(nullptr) {
    const reco::CaloMET * calo = dynamic_cast<const reco::CaloMET *>(&aMET);
    if (calo != 0) caloMET_.push_back(calo->getSpecific());
    const reco::PFMET * pf = dynamic_cast<const reco::PFMET *>(&aMET);
    if (pf != 0) pfMET_.push_back(pf->getSpecific());
}


/// constructor from ref to reco::MET
MET::MET(const edm::RefToBase<reco::MET> & aMETRef) : PATObject<reco::MET>(aMETRef), uncorInfo_(nullptr) {
    const reco::CaloMET * calo = dynamic_cast<const reco::CaloMET *>(aMETRef.get());
    if (calo != 0) caloMET_.push_back(calo->getSpecific());
    const reco::PFMET * pf = dynamic_cast<const reco::PFMET *>(aMETRef.get());
    if (pf != 0) pfMET_.push_back(pf->getSpecific());
}

/// constructor from ref to reco::MET
MET::MET(const edm::Ptr<reco::MET> & aMETRef) : PATObject<reco::MET>(aMETRef), uncorInfo_(nullptr) {
    const reco::CaloMET * calo = dynamic_cast<const reco::CaloMET *>(aMETRef.get());
    if (calo != 0) caloMET_.push_back(calo->getSpecific());
    const reco::PFMET * pf = dynamic_cast<const reco::PFMET *>(aMETRef.get());
    if (pf != 0) pfMET_.push_back(pf->getSpecific());
}

/// copy constructor
MET::MET(MET const& iOther):
PATObject<reco::MET>(iOther),
genMET_(iOther.genMET_),
caloMET_(iOther.caloMET_),
pfMET_(iOther.pfMET_),
uncorInfo_(nullptr)
{
   auto tmp = iOther.uncorInfo_.load(std::memory_order_acquire);
   if(tmp != nullptr) {
      //Only thread-safe to read iOther.nCorrections_ if iOther.uncorInfo_ != nullptr
      nCorrections_ = iOther.nCorrections_;
      uncorInfo_.store( new std::vector<UncorInfo>{*tmp},std::memory_order_release);
   }
}

/// destructor
MET::~MET() {
   delete uncorInfo_.load(std::memory_order_acquire);
}

MET& MET::operator=(MET const& iOther) {
   PATObject<reco::MET>::operator=(iOther);
   genMET_ = iOther.genMET_;
   caloMET_ =iOther.caloMET_;
   pfMET_ =iOther.pfMET_;
   auto tmp = iOther.uncorInfo_.load(std::memory_order_acquire);
   if(tmp != nullptr) {
      //Only thread-safe to read iOther.nCorrections_ if iOther.uncorInfo_ != nullptr
      nCorrections_ = iOther.nCorrections_;
      delete uncorInfo_.exchange( new std::vector<UncorInfo>{*tmp},std::memory_order_acq_rel);
   } else {
      nCorrections_ = 0;
      delete uncorInfo_.exchange(nullptr, std::memory_order_acq_rel);
   }
   return *this;
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
  checkUncor_(); return (*uncorInfo_.load(std::memory_order_acquire))[ix].corEx;
}
float MET::corEy(UncorrectionType ix) const {
  if (ix == uncorrNONE) return 0;
  checkUncor_(); return (*uncorInfo_.load(std::memory_order_acquire))[ix].corEy;
}
float MET::corSumEt(UncorrectionType ix) const {
  if (ix == uncorrNONE) return 0;
  checkUncor_(); return (*uncorInfo_.load(std::memory_order_acquire))[ix].corSumEt;
}
float MET::uncorrectedPt(UncorrectionType ix) const {
  if (ix == uncorrNONE) return pt();
  checkUncor_(); return (*uncorInfo_.load(std::memory_order_acquire))[ix].pt;
}
float MET::uncorrectedPhi(UncorrectionType ix) const {
  if (ix == uncorrNONE) return phi();
  checkUncor_(); return (*uncorInfo_.load(std::memory_order_acquire))[ix].phi;
}


//! check and set transients
void MET::checkUncor_() const {
  if (uncorInfo_.load(std::memory_order_acquire)!=nullptr ) return;

  const std::vector<CorrMETData>& corrs(mEtCorr());
  const auto nCorrectionsTmp = corrs.size();
  
  std::unique_ptr<std::vector<UncorInfo>> uncorInfoTmpPtr{ new std::vector<UncorInfo>{uncorrMAXN} };
  auto& uncorInfoTmp = *uncorInfoTmpPtr;

  UncorrectionType ix;

  //! ugly
  //! ALL
  ix = uncorrALL;
  uncorInfoTmp[ix] = UncorInfo();
  for (unsigned int iC=0; iC < nCorrectionsTmp; ++iC){
    uncorInfoTmp[ix].corEx +=    corrs[iC].mex;
    uncorInfoTmp[ix].corEy +=    corrs[iC].mey;
    uncorInfoTmp[ix].corSumEt += corrs[iC].sumet;
  }
  setPtPhi_(uncorInfoTmp[ix]);

  //! JES
  ix = uncorrJES;
  uncorInfoTmp[ix] = UncorInfo();
  if (nCorrectionsTmp >=1 ){
    unsigned int iC = 0;
    uncorInfoTmp[ix].corEx +=    corrs[iC].mex;
    uncorInfoTmp[ix].corEy +=    corrs[iC].mey;
    uncorInfoTmp[ix].corSumEt += corrs[iC].sumet;
  }
  setPtPhi_(uncorInfoTmp[ix]);

  //! MUON
  ix = uncorrMUON;
  uncorInfoTmp[ix] = UncorInfo();
  if (nCorrectionsTmp >=2 ){
    unsigned int iC = 1;
    uncorInfoTmp[ix].corEx +=    corrs[iC].mex;
    uncorInfoTmp[ix].corEy +=    corrs[iC].mey;
    uncorInfoTmp[ix].corSumEt += corrs[iC].sumet;
  }
  setPtPhi_(uncorInfoTmp[ix]);

  //! TAU
  ix = uncorrTAU;
  uncorInfoTmp[ix] = UncorInfo();
  if (nCorrectionsTmp >=3 ){
    unsigned int iC = 2;
    uncorInfoTmp[ix].corEx +=    corrs[iC].mex;
    uncorInfoTmp[ix].corEy +=    corrs[iC].mey;
    uncorInfoTmp[ix].corSumEt += corrs[iC].sumet;
  }
  setPtPhi_(uncorInfoTmp[ix]);

  //The compare_exchange_strong guarantees that the new value of nCorrections_ will be seen by other
  // threads
  nCorrections_ = nCorrectionsTmp;
  
  std::vector<UncorInfo>* expected=nullptr;
  if(uncorInfo_.compare_exchange_strong(expected,uncorInfoTmpPtr.get(),std::memory_order_acq_rel)) {
     uncorInfoTmpPtr.release();
  }
}

void MET::setPtPhi_(UncorInfo& uci) const {
  float lpx = px() - uci.corEx;
  float lpy = py() - uci.corEy;
  uci.pt = sqrt(lpx*lpx + lpy*lpy);
  uci.phi = atan2(lpy, lpx);
}

MET::Vector2 MET::shiftedP2(MET::METUncertainty shift, MET::METUncertaintyLevel level)  const {
    const std::vector<PackedMETUncertainty> &v = (level == Type1 ? uncertaintiesType1_ : (level == Type1p2 ? uncertaintiesType1p2_ : uncertaintiesRaw_));
    if (v.empty()) throw cms::Exception("Unsupported", "MET uncertainties not available for the specified correction type");
    Vector2 ret{ (px() + v[shift].dpx()), (py() + v[shift].dpy()) };
    return ret;
}
MET::Vector MET::shiftedP3(MET::METUncertainty shift, MET::METUncertaintyLevel level)  const {
    const std::vector<PackedMETUncertainty> &v = (level == Type1 ? uncertaintiesType1_ : (level == Type1p2 ? uncertaintiesType1p2_ : uncertaintiesRaw_));
    if (v.empty()) throw cms::Exception("Unsupported", "MET uncertainties not available for the specified correction type");
    return Vector(px() + v[shift].dpx(), py() + v[shift].dpy(), 0);
}
MET::LorentzVector MET::shiftedP4(METUncertainty shift, MET::METUncertaintyLevel level)  const {
    const std::vector<PackedMETUncertainty> &v = (level == Type1 ? uncertaintiesType1_ : (level == Type1p2 ? uncertaintiesType1p2_ : uncertaintiesRaw_));
    if (v.empty()) throw cms::Exception("Unsupported", "MET uncertainties not available for the specified correction type");
    double x = px() + v[shift].dpx(), y = py() + v[shift].dpy();
    return LorentzVector(x, y, 0, std::hypot(x,y));
}
double MET::shiftedSumEt(MET::METUncertainty shift, MET::METUncertaintyLevel level) const {
    const std::vector<PackedMETUncertainty> &v = (level == Type1 ? uncertaintiesType1_ : (level == Type1p2 ? uncertaintiesType1p2_ : uncertaintiesRaw_));
    if (v.empty()) throw cms::Exception("Unsupported", "MET uncertainties not available for the specified correction type");
    return sumEt() + v[shift].dsumEt();
}
void MET::setShift(double px, double py, double sumEt, MET::METUncertainty shift, MET::METUncertaintyLevel level) {
    std::vector<PackedMETUncertainty> &v = (level == Type1 ? uncertaintiesType1_ : (level == Type1p2 ? uncertaintiesType1p2_ : uncertaintiesRaw_));
    if (v.empty()) v.resize(METUncertaintySize);
    v[shift].set(px - this->px(), py - this->py(), sumEt - this->sumEt());
}
