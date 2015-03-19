//
//

#include "DataFormats/PatCandidates/interface/MET.h"


using namespace pat;


/// default constructor
MET::MET() {
}


/// constructor from reco::MET
MET::MET(const reco::MET & aMET) : PATObject<reco::MET>(aMET) {
    const reco::CaloMET * calo = dynamic_cast<const reco::CaloMET *>(&aMET);
    if (calo != 0) caloMET_.push_back(calo->getSpecific());
    const reco::PFMET * pf = dynamic_cast<const reco::PFMET *>(&aMET);
    if (pf != 0) pfMET_.push_back(pf->getSpecific());
}


/// constructor from ref to reco::MET
MET::MET(const edm::RefToBase<reco::MET> & aMETRef) : PATObject<reco::MET>(aMETRef) {
    const reco::CaloMET * calo = dynamic_cast<const reco::CaloMET *>(aMETRef.get());
    if (calo != 0) caloMET_.push_back(calo->getSpecific());
    const reco::PFMET * pf = dynamic_cast<const reco::PFMET *>(aMETRef.get());
    if (pf != 0) pfMET_.push_back(pf->getSpecific());
}

/// constructor from ref to reco::MET
MET::MET(const edm::Ptr<reco::MET> & aMETRef) : PATObject<reco::MET>(aMETRef) {
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
uncertaintiesRaw_(iOther.uncertaintiesRaw_),
uncertaintiesType1_(iOther.uncertaintiesType1_),
uncertaintiesType1p2_(iOther.uncertaintiesType1p2_),
caloPackedMet_(iOther.caloPackedMet_) {
}

/// destructor
MET::~MET() {

}

MET& MET::operator=(MET const& iOther) {
   PATObject<reco::MET>::operator=(iOther);
   genMET_ = iOther.genMET_;
   caloMET_ =iOther.caloMET_;
   pfMET_ =iOther.pfMET_;
   uncertaintiesRaw_ = iOther.uncertaintiesRaw_;
   uncertaintiesType1_ = iOther.uncertaintiesType1_;
   uncertaintiesType1p2_ = iOther.uncertaintiesType1p2_;
   caloPackedMet_ = iOther.caloPackedMet_;

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
//unsigned int MET::nCorrections() const { checkUncor_(); return nCorrections_; }

float MET::uncorrectedPt() const {
  return shiftedPt(MET::METUncertainty::NoShift, MET::METUncertaintyLevel::Raw);
}
float MET::uncorrectedPhi() const {
   return shiftedPt(MET::METUncertainty::NoShift, MET::METUncertaintyLevel::Raw);
}
float MET::uncorrectedSumEt() const {
  return shiftedSumEt(MET::METUncertainty::NoShift, MET::METUncertaintyLevel::Raw);
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
  if(level != Calo ) {
    std::vector<PackedMETUncertainty> &v = (level == Type1 ? uncertaintiesType1_ : (level == Type1p2 ? uncertaintiesType1p2_ : uncertaintiesRaw_));
    if (v.empty()) v.resize(METUncertaintySize);
    v[shift].set(px - this->px(), py - this->py(), sumEt - this->sumEt());
  } else {
    caloPackedMet_.set(px, py, sumEt);
  }
}

MET::Vector2 MET::caloMETP2() const {
  Vector2 ret{ caloPackedMet_.dpx(), caloPackedMet_.dpy() };
  return ret;
}

double MET::caloMETPt() const {
  return caloMETP2().pt();
}

double MET::caloMETPhi() const {
  return caloMETP2().phi();
}

double MET::caloMETSumEt() const {
  return caloPackedMet_.dsumEt();
}

#include "DataFormats/PatCandidates/interface/libminifloat.h"

void MET::PackedMETUncertainty::pack() {
  packedDpx_  =  MiniFloatConverter::float32to16(dpx_);
  packedDpy_  =  MiniFloatConverter::float32to16(dpy_);
  packedDSumEt_  =  MiniFloatConverter::float32to16(dsumEt_);
}
void  MET::PackedMETUncertainty::unpack() const {
  unpacked_=true;
  dpx_=MiniFloatConverter::float16to32(packedDpx_);
  dpy_=MiniFloatConverter::float16to32(packedDpy_);
  dsumEt_=MiniFloatConverter::float16to32(packedDSumEt_);

}

