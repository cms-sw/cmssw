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
    const pat::MET * pm = dynamic_cast<const pat::MET *>(&aMET);
    if (pm != 0) this->operator=(*pm);

    initCorMap();
}


/// constructor from ref to reco::MET
MET::MET(const edm::RefToBase<reco::MET> & aMETRef) : PATObject<reco::MET>(aMETRef) {
    const reco::CaloMET * calo = dynamic_cast<const reco::CaloMET *>(aMETRef.get());
    if (calo != 0) caloMET_.push_back(calo->getSpecific());
    const reco::PFMET * pf = dynamic_cast<const reco::PFMET *>(aMETRef.get());
    if (pf != 0) pfMET_.push_back(pf->getSpecific());
    const pat::MET * pm = dynamic_cast<const pat::MET *>(aMETRef.get());
    if (pm != 0) this->operator=(*pm);

    initCorMap();
}

/// constructor from ref to reco::MET
MET::MET(const edm::Ptr<reco::MET> & aMETRef) : PATObject<reco::MET>(aMETRef) {
    const reco::CaloMET * calo = dynamic_cast<const reco::CaloMET *>(aMETRef.get());
    if (calo != 0) caloMET_.push_back(calo->getSpecific());
    const reco::PFMET * pf = dynamic_cast<const reco::PFMET *>(aMETRef.get());
    if (pf != 0) pfMET_.push_back(pf->getSpecific());
    const pat::MET * pm = dynamic_cast<const pat::MET *>(aMETRef.get());
    if (pm != 0) this->operator=(*pm);

    initCorMap();
}

/// copy constructor
MET::MET(MET const& iOther):
PATObject<reco::MET>(iOther),
genMET_(iOther.genMET_),
caloMET_(iOther.caloMET_),
pfMET_(iOther.pfMET_),
uncertainties_(iOther.uncertainties_),
corrections_(iOther.corrections_),
caloPackedMet_(iOther.caloPackedMet_) {

  initCorMap();
}

/// constructor for corrected mets, keeping track of srcMET informations, 
// old uncertainties discarded on purpose to avoid confusion
MET::MET(const reco::MET & corMET, const MET& srcMET ):
PATObject<reco::MET>(corMET),
genMET_(srcMET.genMET_),
caloMET_(srcMET.caloMET_),
pfMET_(srcMET.pfMET_),
caloPackedMet_(srcMET.caloPackedMet_) {

  initCorMap();
}

/// destructor
MET::~MET() {

}

MET& MET::operator=(MET const& iOther) {
   PATObject<reco::MET>::operator=(iOther);
   genMET_ = iOther.genMET_;
   caloMET_ =iOther.caloMET_;
   pfMET_ =iOther.pfMET_;
   uncertainties_ = iOther.uncertainties_;
   corrections_ = iOther.corrections_;
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


//Method to set the MET significance
void MET::setMETSignificance(const double& metSig) {
  metSig_ = metSig;
}

double MET::metSignificance() const {
  return metSig_;
}


void
MET::initCorMap() {

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

}

const MET::PackedMETUncertainty
MET::findMETTotalShift(MET::METCorrectionLevel cor, MET::METUncertainty shift) const {

  //find corrections shifts =============================
  std::map<MET::METCorrectionLevel, std::vector<MET::METCorrectionType> >::const_iterator itCor_ = corMap_.find(cor);
  if(itCor_==corMap_.end() ) throw cms::Exception("Unsupported", "Specified MET correction scheme does not exist");

  bool isSmeared=false;
  MET::PackedMETUncertainty totShift;
  unsigned int scor=itCor_->second.size();
  for(unsigned int i=0; i<scor;i++) {
    totShift.add( corrections_[ itCor_->second[i] ].dpx(),
		  corrections_[ itCor_->second[i] ].dpy(),
		  corrections_[ itCor_->second[i] ].dsumEt() );

    if(itCor_->first>=MET::Type1Smear)
      isSmeared=true;
  }

  //find uncertainty shift =============================
  if(shift>=MET::METUncertaintySize) throw cms::Exception("Unsupported", "MET uncertainty does not exist");
  if(isSmeared && shift<=MET::JetResDown) shift = (MET::METUncertainty)(MET::METUncertaintySize+shift+1);
							  
  totShift.add( uncertainties_[ shift ].dpx(),
		uncertainties_[ shift ].dpy(),
		uncertainties_[ shift ].dsumEt() );

  return totShift;
}


MET::Vector2 MET::shiftedP2(MET::METUncertainty shift, MET::METCorrectionLevel cor)  const {
  const MET::PackedMETUncertainty& v = findMETTotalShift(cor,shift);
  Vector2 ret{ (px() + v.dpx()), (py() + v.dpy()) };
  return ret;
}
MET::Vector MET::shiftedP3(MET::METUncertainty shift, MET::METCorrectionLevel cor)  const {
  const MET::PackedMETUncertainty& v = findMETTotalShift(cor,shift);
  return Vector(px() + v.dpx(), py() + v.dpy(), 0);
}
MET::LorentzVector MET::shiftedP4(METUncertainty shift, MET::METCorrectionLevel cor)  const {
  const MET::PackedMETUncertainty& v = findMETTotalShift(cor,shift);
  double x = px() + v.dpx(), y = py() + v.dpy();
  return LorentzVector(x, y, 0, std::hypot(x,y));
}
double MET::shiftedSumEt(MET::METUncertainty shift, MET::METCorrectionLevel cor) const {
  const MET::PackedMETUncertainty& v = findMETTotalShift(cor,shift);
  return sumEt() + v.dsumEt();
}

MET::Vector2 MET::corP2(MET::METCorrectionLevel cor)  const {
  return shiftedP2(MET::NoShift, cor );
}
MET::Vector MET::corP3(MET::METCorrectionLevel cor)  const {
  return shiftedP3(MET::NoShift, cor );
}
MET::LorentzVector MET::corP4(MET::METCorrectionLevel cor)  const {
  return shiftedP4(MET::NoShift, cor );
}
double MET::corSumEt(MET::METCorrectionLevel cor) const {
  return shiftedSumEt(MET::NoShift, cor );
}

MET::Vector2 MET::uncorP2()  const {
  return shiftedP2(MET::NoShift, MET::Raw );
}
MET::Vector MET::uncorP3()  const {
  return shiftedP3(MET::NoShift, MET::Raw );
}
MET::LorentzVector MET::uncorP4()  const {
  return shiftedP4(MET::NoShift, MET::Raw );
}
double MET::uncorSumEt() const {
  return shiftedSumEt(MET::NoShift, MET::Raw );
}


void MET::setUncShift(double px, double py, double sumEt, METUncertainty shift, bool isSmeared) {
  if (uncertainties_.empty()) uncertainties_.resize(METUncertainty::METFullUncertaintySize);
  
  if(isSmeared && shift<=MET::JetResDown) {
    //changing reference to only get the uncertainty shift and not the smeared one
    // which is performed independently
    shift = (MET::METUncertainty)(METUncertainty::METUncertaintySize+shift+1);
    const PackedMETUncertainty& ref = uncertainties_[METUncertainty::NoShift];
    uncertainties_[shift].set(px + ref.dpx() - this->px(), py + ref.dpy() - this->py(), sumEt + ref.dsumEt() - this->sumEt() );
  }
  else
    uncertainties_[shift].set(px - this->px(), py - this->py(), sumEt - this->sumEt());
  
}

void MET::setCorShift(double px, double py, double sumEt, MET::METCorrectionType level) {
  if (corrections_.empty()) corrections_.resize(MET::METCorrectionType::METCorrectionTypeSize);
  corrections_[level].set(px - this->px(), py - this->py(), sumEt - this->sumEt());
  
}


MET::Vector2 MET::caloMETP2() const {
  return shiftedP2(MET::METUncertainty::NoShift, MET::METCorrectionLevel::RawCalo );
}

double MET::caloMETPt() const {
  return caloMETP2().pt();
}

double MET::caloMETPhi() const {
  return caloMETP2().phi();
}

double MET::caloMETSumEt() const {
  return shiftedSumEt(MET::NoShift, MET::RawCalo );
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

