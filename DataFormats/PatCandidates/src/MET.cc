//
//

#include "DataFormats/PatCandidates/interface/MET.h"

using namespace pat;

/// default constructor
MET::MET() { initCorMap(); }

/// constructor from reco::MET
MET::MET(const reco::MET &aMET) : PATObject<reco::MET>(aMET) {
  const reco::CaloMET *calo = dynamic_cast<const reco::CaloMET *>(&aMET);
  if (calo != nullptr)
    caloMET_.push_back(calo->getSpecific());
  const reco::PFMET *pf = dynamic_cast<const reco::PFMET *>(&aMET);
  if (pf != nullptr)
    pfMET_.push_back(pf->getSpecific());
  const pat::MET *pm = dynamic_cast<const pat::MET *>(&aMET);
  if (pm != nullptr)
    this->operator=(*pm);

  metSig_ = 0.;
  sumPtUnclustered_ = 0.;
  initCorMap();
}

/// constructor from ref to reco::MET
MET::MET(const edm::RefToBase<reco::MET> &aMETRef) : PATObject<reco::MET>(aMETRef) {
  const reco::CaloMET *calo = dynamic_cast<const reco::CaloMET *>(aMETRef.get());
  if (calo != nullptr)
    caloMET_.push_back(calo->getSpecific());
  const reco::PFMET *pf = dynamic_cast<const reco::PFMET *>(aMETRef.get());
  if (pf != nullptr)
    pfMET_.push_back(pf->getSpecific());
  const pat::MET *pm = dynamic_cast<const pat::MET *>(aMETRef.get());
  if (pm != nullptr)
    this->operator=(*pm);

  metSig_ = 0.;
  sumPtUnclustered_ = 0.;
  initCorMap();
}

/// constructor from ref to reco::MET
MET::MET(const edm::Ptr<reco::MET> &aMETRef) : PATObject<reco::MET>(aMETRef) {
  const reco::CaloMET *calo = dynamic_cast<const reco::CaloMET *>(aMETRef.get());
  if (calo != nullptr)
    caloMET_.push_back(calo->getSpecific());
  const reco::PFMET *pf = dynamic_cast<const reco::PFMET *>(aMETRef.get());
  if (pf != nullptr)
    pfMET_.push_back(pf->getSpecific());
  const pat::MET *pm = dynamic_cast<const pat::MET *>(aMETRef.get());
  if (pm != nullptr)
    this->operator=(*pm);

  metSig_ = 0.;
  sumPtUnclustered_ = 0.;
  initCorMap();
}

/// copy constructor
MET::MET(MET const &iOther)
    : PATObject<reco::MET>(iOther),
      genMET_(iOther.genMET_),
      caloMET_(iOther.caloMET_),
      pfMET_(iOther.pfMET_),
      metSig_(iOther.metSig_),
      sumPtUnclustered_(iOther.sumPtUnclustered_),
      uncertaintiesRaw_(iOther.uncertaintiesRaw_),          //74X reading compatibility
      uncertaintiesType1_(iOther.uncertaintiesType1_),      //74X compatibility
      uncertaintiesType1p2_(iOther.uncertaintiesType1p2_),  //74X compatibility
      uncertainties_(iOther.uncertainties_),
      corrections_(iOther.corrections_),
      caloPackedMet_(iOther.caloPackedMet_) {
  initCorMap();
}

/// constructor for corrected mets, keeping track of srcMET informations,
// old uncertainties discarded on purpose to avoid confusion
MET::MET(const reco::MET &corMET, const MET &srcMET)
    : PATObject<reco::MET>(corMET),
      genMET_(srcMET.genMET_),
      caloMET_(srcMET.caloMET_),
      pfMET_(srcMET.pfMET_),
      metSig_(srcMET.metSig_),
      sumPtUnclustered_(srcMET.sumPtUnclustered_),
      caloPackedMet_(srcMET.caloPackedMet_) {
  setSignificanceMatrix(srcMET.getSignificanceMatrix());

  initCorMap();
}

/// destructor
MET::~MET() {}

MET &MET::operator=(MET const &iOther) {
  PATObject<reco::MET>::operator=(iOther);
  genMET_ = iOther.genMET_;
  caloMET_ = iOther.caloMET_;
  pfMET_ = iOther.pfMET_;
  uncertaintiesRaw_ = iOther.uncertaintiesRaw_;  //74X compatibility
  uncertaintiesType1_ = iOther.uncertaintiesType1_;
  uncertaintiesType1p2_ = iOther.uncertaintiesType1p2_;
  uncertainties_ = iOther.uncertainties_;
  corrections_ = iOther.corrections_;
  metSig_ = iOther.metSig_;
  sumPtUnclustered_ = iOther.sumPtUnclustered_;
  caloPackedMet_ = iOther.caloPackedMet_;

  return *this;
}

/// return the generated MET from neutrinos
const reco::GenMET *MET::genMET() const { return (!genMET_.empty() ? &genMET_.front() : nullptr); }

/// method to set the generated MET
void MET::setGenMET(const reco::GenMET &gm) {
  genMET_.clear();
  genMET_.push_back(gm);
}

//Method to set the MET significance
void MET::setMETSignificance(const double &metSig) { metSig_ = metSig; }

double MET::metSignificance() const { return metSig_; }

void MET::setMETSumPtUnclustered(const double &sumPtUnclustered) { sumPtUnclustered_ = sumPtUnclustered; }

double MET::metSumPtUnclustered() const { return sumPtUnclustered_; }

void MET::initCorMap() {
  std::vector<MET::METCorrectionType> tmpRaw;
  std::vector<MET::METCorrectionType> tmpType1;
  std::vector<MET::METCorrectionType> tmpType01;
  std::vector<MET::METCorrectionType> tmpTypeXY;
  std::vector<MET::METCorrectionType> tmpType1XY;
  std::vector<MET::METCorrectionType> tmpType01XY;
  std::vector<MET::METCorrectionType> tmpType1Smear;
  std::vector<MET::METCorrectionType> tmpType01Smear;
  std::vector<MET::METCorrectionType> tmpType1SmearXY;
  std::vector<MET::METCorrectionType> tmpType01SmearXY;

  tmpRaw.push_back(MET::None);

  tmpType1.push_back(MET::T1);
  tmpType01.push_back(MET::T1);
  tmpType1XY.push_back(MET::T1);
  tmpType01XY.push_back(MET::T1);
  tmpType1Smear.push_back(MET::T1);
  tmpType01Smear.push_back(MET::T1);
  tmpType1SmearXY.push_back(MET::T1);
  tmpType01SmearXY.push_back(MET::T1);

  tmpType01.push_back(MET::T0);
  tmpType01XY.push_back(MET::T0);
  tmpType01Smear.push_back(MET::T0);
  tmpType01SmearXY.push_back(MET::T0);

  tmpType1Smear.push_back(MET::Smear);
  tmpType01Smear.push_back(MET::Smear);
  tmpType1SmearXY.push_back(MET::Smear);
  tmpType01SmearXY.push_back(MET::Smear);

  tmpTypeXY.push_back(MET::TXYForRaw);
  tmpType1XY.push_back(MET::TXY);
  tmpType01XY.push_back(MET::TXYForT01);
  tmpType1SmearXY.push_back(MET::TXYForT1Smear);
  tmpType01SmearXY.push_back(MET::TXYForT01Smear);

  corMap_[MET::Raw] = tmpRaw;
  corMap_[MET::Type1] = tmpType1;
  corMap_[MET::Type01] = tmpType01;
  corMap_[MET::TypeXY] = tmpTypeXY;
  corMap_[MET::Type1XY] = tmpType1XY;
  corMap_[MET::Type01XY] = tmpType01XY;
  corMap_[MET::Type1Smear] = tmpType1Smear;
  corMap_[MET::Type01Smear] = tmpType01Smear;
  corMap_[MET::Type1SmearXY] = tmpType1SmearXY;
  corMap_[MET::Type01SmearXY] = tmpType01SmearXY;

  //specific calo case
  std::vector<MET::METCorrectionType> tmpRawCalo;
  tmpRawCalo.push_back(MET::Calo);
  corMap_[MET::RawCalo] = tmpRawCalo;

  //specific chs case
  std::vector<MET::METCorrectionType> tmpRawChs;
  tmpRawChs.push_back(MET::Chs);
  corMap_[MET::RawChs] = tmpRawChs;

  //specific trk case
  std::vector<MET::METCorrectionType> tmpRawTrk;
  tmpRawTrk.push_back(MET::Trk);
  corMap_[MET::RawTrk] = tmpRawTrk;

  //specific deep response tune case
  std::vector<MET::METCorrectionType> tmpDeepResponse;
  tmpDeepResponse.push_back(MET::DeepResponseTune);
  corMap_[MET::RawDeepResponseTune] = tmpDeepResponse;

  //specific deep resolution tune case
  std::vector<MET::METCorrectionType> tmpDeepResolution;
  tmpDeepResolution.push_back(MET::DeepResolutionTune);
  corMap_[MET::RawDeepResolutionTune] = tmpDeepResolution;
}

MET::UnpackedMETUncertainty MET::findMETTotalShift(MET::METCorrectionLevel cor, MET::METUncertainty shift) const {
  //find corrections shifts =============================
  std::map<MET::METCorrectionLevel, std::vector<MET::METCorrectionType> >::const_iterator itCor_ = corMap_.find(cor);
  if (itCor_ == corMap_.end())
    throw cms::Exception("Unsupported", "Specified MET correction scheme does not exist");

  bool isSmeared = false;
  MET::UnpackedMETUncertainty totShift;
  unsigned int scor = itCor_->second.size();
  for (unsigned int i = 0; i < scor; i++) {
    auto up = corrections_[itCor_->second[i]].unpack();
    totShift.add(up.dpx(), up.dpy(), up.dsumEt());

    if (itCor_->first >= MET::Type1Smear)
      isSmeared = true;
  }

  //find uncertainty shift =============================

  if (uncertainties_.empty())
    return totShift;

  if (shift >= MET::METUncertaintySize)
    throw cms::Exception("Unsupported", "MET uncertainty does not exist");
  if (isSmeared && shift <= MET::JetResDown)
    shift = (MET::METUncertainty)(MET::METUncertaintySize + shift + 1);

  auto up = uncertainties_[shift].unpack();
  totShift.add(up.dpx(), up.dpy(), up.dsumEt());

  return totShift;
}

MET::Vector2 MET::shiftedP2(MET::METUncertainty shift, MET::METCorrectionLevel cor) const {
  Vector2 vo;

  //backward compatibility with 74X samples -> the only one
  // with uncertaintiesType1_/uncertaintiesRaw_ not empty
  //will be removed once 74X is not used anymore
  if (!uncertaintiesType1_.empty() || !uncertaintiesRaw_.empty()) {
    if (cor != MET::METCorrectionLevel::RawCalo) {
      vo = shiftedP2_74x(shift, cor);
    } else {
      Vector2 ret{caloPackedMet_.unpackDpx(), caloPackedMet_.unpackDpy()};
      vo = ret;
    }
  } else {
    auto v = findMETTotalShift(cor, shift);
    Vector2 ret{(px() + v.dpx()), (py() + v.dpy())};
    //return ret;
    vo = ret;
  }
  return vo;
}
MET::Vector MET::shiftedP3(MET::METUncertainty shift, MET::METCorrectionLevel cor) const {
  Vector vo;

  //backward compatibility with 74X samples -> the only one
  // with uncertaintiesType1_/uncertaintiesRaw_ not empty
  //will be removed once 74X is not used anymore
  if (!uncertaintiesType1_.empty() || !uncertaintiesRaw_.empty()) {
    if (cor != MET::METCorrectionLevel::RawCalo) {
      vo = shiftedP3_74x(shift, cor);
    } else {
      Vector tmp(caloPackedMet_.unpackDpx(), caloPackedMet_.unpackDpy(), 0);
      vo = tmp;
    }
  } else {
    const MET::UnpackedMETUncertainty &v = findMETTotalShift(cor, shift);
    //return Vector(px() + v.dpx(), py() + v.dpy(), 0);
    Vector tmp(px() + v.dpx(), py() + v.dpy(), 0);
    vo = tmp;
  }
  return vo;
}
MET::LorentzVector MET::shiftedP4(METUncertainty shift, MET::METCorrectionLevel cor) const {
  LorentzVector vo;

  //backward compatibility with 74X samples -> the only one
  // with uncertaintiesType1_/uncertaintiesRaw_ not empty
  //will be removed once 74X is not used anymore
  if (!uncertaintiesType1_.empty() || !uncertaintiesRaw_.empty()) {
    if (cor != MET::METCorrectionLevel::RawCalo) {
      vo = shiftedP4_74x(shift, cor);
    } else {
      double x = caloPackedMet_.unpackDpx(), y = caloPackedMet_.unpackDpy();
      LorentzVector tmp(x, y, 0, std::hypot(x, y));
      vo = tmp;
    }
  } else {
    const auto v = findMETTotalShift(cor, shift);
    double x = px() + v.dpx(), y = py() + v.dpy();
    //return LorentzVector(x, y, 0, std::hypot(x,y));
    LorentzVector tmp(x, y, 0, std::hypot(x, y));
    vo = tmp;
  }
  return vo;
}
double MET::shiftedSumEt(MET::METUncertainty shift, MET::METCorrectionLevel cor) const {
  double sumEto;

  //backward compatibility with 74X samples -> the only one
  // with uncertaintiesType1_/uncertaintiesRaw_ not empty
  //will be removed once 74X is not used anymore
  if (!uncertaintiesType1_.empty() || !uncertaintiesRaw_.empty()) {
    if (cor != MET::METCorrectionLevel::RawCalo) {
      sumEto = shiftedSumEt_74x(shift, cor);
    } else {
      sumEto = caloPackedMet_.unpackDSumEt();
    }
  } else {
    const auto v = findMETTotalShift(cor, shift);
    //return sumEt() + v.dsumEt();
    sumEto = sumEt() + v.dsumEt();
  }
  return sumEto;
}

MET::Vector2 MET::corP2(MET::METCorrectionLevel cor) const { return shiftedP2(MET::NoShift, cor); }
MET::Vector MET::corP3(MET::METCorrectionLevel cor) const { return shiftedP3(MET::NoShift, cor); }
MET::LorentzVector MET::corP4(MET::METCorrectionLevel cor) const { return shiftedP4(MET::NoShift, cor); }
double MET::corSumEt(MET::METCorrectionLevel cor) const { return shiftedSumEt(MET::NoShift, cor); }

MET::Vector2 MET::uncorP2() const { return shiftedP2(MET::NoShift, MET::Raw); }
MET::Vector MET::uncorP3() const { return shiftedP3(MET::NoShift, MET::Raw); }
MET::LorentzVector MET::uncorP4() const { return shiftedP4(MET::NoShift, MET::Raw); }
double MET::uncorSumEt() const { return shiftedSumEt(MET::NoShift, MET::Raw); }

void MET::setUncShift(double px, double py, double sumEt, METUncertainty shift, bool isSmeared) {
  if (uncertainties_.empty()) {
    uncertainties_.resize(METUncertainty::METFullUncertaintySize);
  }

  if (isSmeared && shift <= MET::JetResDown) {
    //changing reference to only get the uncertainty shift and not the smeared one
    // which is performed independently
    shift = (MET::METUncertainty)(METUncertainty::METUncertaintySize + shift + 1);
    const PackedMETUncertainty &ref = corrections_[METCorrectionType::Smear];
    uncertainties_[shift].set(px - ref.unpackDpx() - this->px(),
                              py - ref.unpackDpy() - this->py(),
                              sumEt - ref.unpackDSumEt() - this->sumEt());
  } else
    uncertainties_[shift].set(px - this->px(), py - this->py(), sumEt - this->sumEt());
}

void MET::setCorShift(double px, double py, double sumEt, MET::METCorrectionType level) {
  if (corrections_.empty()) {
    corrections_.resize(MET::METCorrectionType::METCorrectionTypeSize);
  }

  corrections_[level].set(px - this->px(), py - this->py(), sumEt - this->sumEt());
}

MET::Vector2 MET::caloMETP2() const {
  return shiftedP2(MET::METUncertainty::NoShift, MET::METCorrectionLevel::RawCalo);
}

double MET::caloMETPt() const { return caloMETP2().pt(); }

double MET::caloMETPhi() const { return caloMETP2().phi(); }

double MET::caloMETSumEt() const { return shiftedSumEt(MET::NoShift, MET::RawCalo); }

// functions to access to 74X samples ========================================================
MET::Vector2 MET::shiftedP2_74x(MET::METUncertainty shift, MET::METCorrectionLevel level) const {
  if (level != Type1 && level != Raw)
    throw cms::Exception("Unsupported", "MET uncertainties only supported for Raw and Type1 in 74X samples \n");
  const std::vector<PackedMETUncertainty> &v = (level == Type1 ? uncertaintiesType1_ : uncertaintiesRaw_);
  if (v.empty())
    throw cms::Exception("Unsupported", "MET uncertainties not available for the specified correction type\n");
  if (v.size() == 1) {
    if (shift != MET::METUncertainty::NoShift)
      throw cms::Exception(
          "Unsupported",
          "MET uncertainties not available for the specified correction type (only central value available)\n");
    auto const &p = v.front();
    return Vector2{(px() + p.unpackDpx()), (py() + p.unpackDpy())};
  }
  auto const &p = v[shift];
  Vector2 ret{(px() + p.unpackDpx()), (py() + p.unpackDpy())};
  return ret;
}

MET::Vector MET::shiftedP3_74x(MET::METUncertainty shift, MET::METCorrectionLevel level) const {
  if (level != Type1 && level != Raw)
    throw cms::Exception("Unsupported", "MET uncertainties only supported for Raw and Type1 in 74X samples \n");
  const std::vector<PackedMETUncertainty> &v = (level == Type1 ? uncertaintiesType1_ : uncertaintiesRaw_);
  if (v.empty())
    throw cms::Exception("Unsupported", "MET uncertainties not available for the specified correction type\n");
  if (v.size() == 1) {
    if (shift != MET::METUncertainty::NoShift)
      throw cms::Exception(
          "Unsupported",
          "MET uncertainties not available for the specified correction type (only central value available)\n");
    auto const &p = v.front();
    return Vector(px() + p.unpackDpx(), py() + p.unpackDpy(), 0);
  }
  auto const &p = v[shift];
  return Vector(px() + p.unpackDpx(), py() + p.unpackDpy(), 0);
}

MET::LorentzVector MET::shiftedP4_74x(METUncertainty shift, MET::METCorrectionLevel level) const {
  if (level != Type1 && level != Raw)
    throw cms::Exception("Unsupported", "MET uncertainties only supported for Raw and Type1 in 74X samples\n");
  const std::vector<PackedMETUncertainty> &v = (level == Type1 ? uncertaintiesType1_ : uncertaintiesRaw_);
  if (v.empty())
    throw cms::Exception("Unsupported", "MET uncertainties not available for the specified correction type\n");
  if (v.size() == 1) {
    if (shift != MET::METUncertainty::NoShift)
      throw cms::Exception(
          "Unsupported",
          "MET uncertainties not available for the specified correction type (only central value available)\n");
    auto const &p = v.front();
    double x = px() + p.unpackDpx(), y = py() + p.unpackDpy();
    return LorentzVector(x, y, 0, std::hypot(x, y));
  }
  auto const &p = v[shift];
  double x = px() + p.unpackDpx(), y = py() + p.unpackDpy();
  return LorentzVector(x, y, 0, std::hypot(x, y));
}

double MET::shiftedSumEt_74x(MET::METUncertainty shift, MET::METCorrectionLevel level) const {
  if (level != Type1 && level != Raw)
    throw cms::Exception("Unsupported", "MET uncertainties only supported for Raw and Type1 in 74X samples\n");
  const std::vector<PackedMETUncertainty> &v = (level == Type1 ? uncertaintiesType1_ : uncertaintiesRaw_);
  if (v.empty())
    throw cms::Exception("Unsupported", "MET uncertainties not available for the specified correction type\n");
  if (v.size() == 1) {
    if (shift != MET::METUncertainty::NoShift)
      throw cms::Exception(
          "Unsupported",
          "MET uncertainties not available for the specified correction type (only central value available)\n");
    return sumEt() + v.front().unpackDSumEt();
  }
  return sumEt() + v[shift].unpackDSumEt();
}

#include "DataFormats/Math/interface/libminifloat.h"

MET::UnpackedMETUncertainty MET::PackedMETUncertainty::unpack() const {
  auto dpx = MiniFloatConverter::float16to32(packedDpx_);
  auto dpy = MiniFloatConverter::float16to32(packedDpy_);
  auto dsumEt = MiniFloatConverter::float16to32(packedDSumEt_);
  return UnpackedMETUncertainty(dpx, dpy, dsumEt);
}

float MET::PackedMETUncertainty::unpackDpx() const { return MiniFloatConverter::float16to32(packedDpx_); }

float MET::PackedMETUncertainty::unpackDpy() const { return MiniFloatConverter::float16to32(packedDpy_); }

float MET::PackedMETUncertainty::unpackDSumEt() const { return MiniFloatConverter::float16to32(packedDSumEt_); }

void MET::PackedMETUncertainty::pack(float dpx, float dpy, float dsumEt) {
  packedDpx_ = MiniFloatConverter::float32to16(dpx);
  packedDpy_ = MiniFloatConverter::float32to16(dpy);
  packedDSumEt_ = MiniFloatConverter::float32to16(dsumEt);
}
