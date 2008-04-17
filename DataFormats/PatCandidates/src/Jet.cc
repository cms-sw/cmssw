//
// $Id: Jet.cc,v 1.11 2008/04/03 19:22:00 lowette Exp $
//

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace pat;


/// default constructor
Jet::Jet() :
  PATObject<JetType>(JetType(reco::Particle::LorentzVector(0, 0, 0, 0), reco::Particle::Point(0,0,0), reco::CaloJet::Specific(), reco::Jet::Constituents())),
  embeddedCaloTowers_(false),
  partonFlavour_(0), lrPhysicsJetLRval_(-999.), lrPhysicsJetProb_(-1),
  jetCharge_(0.0) {
}


/// constructor from a JetType
Jet::Jet(const JetType & aJet) :
  PATObject<JetType>(aJet),
  embeddedCaloTowers_(false),
  partonFlavour_(0), lrPhysicsJetLRval_(-999.), lrPhysicsJetProb_(-1),
  jetCharge_(0.0) {
}


/// constructor from ref to JetType
Jet::Jet(const edm::RefToBase<JetType> & aJetRef) :
  PATObject<JetType>(aJetRef),
  embeddedCaloTowers_(false),
  partonFlavour_(0), lrPhysicsJetLRval_(-999.), lrPhysicsJetProb_(-1),
  jetCharge_(0.0) {
}


/// destructor
Jet::~Jet() {
}


/// override the getConstituent method from CaloJet, to access the internal storage of the constituents
/// this returns a transient Ref which *should never be persisted*!
CaloTowerRef Jet::getConstituent(unsigned int idx) const {
  if (embeddedCaloTowers_) {
    return CaloTowerRef(&caloTowers_, idx);
  } else {
    return JetType::getConstituent(idx);
  }
}


/// override the getConstituents method from CaloJet, to access the internal storage of the constituents
/// this returns a transient RefVector which *should never be persisted*!
std::vector<CaloTowerRef> Jet::getConstituents() const {
  std::vector<CaloTowerRef> caloTowerRefs;
  for (unsigned int i = 0; i < caloTowers_.size(); ++i) {
    caloTowerRefs.push_back(CaloTowerRef(&caloTowers_, i));
  }
  return caloTowerRefs;
}


/// return the matched generated parton
const reco::Particle * Jet::genParton() const {
  return (genParton_.size() > 0 ? &genParton_.front() : 0);
}


/// return the matched generated jet
const reco::GenJet * Jet::genJet() const {
  return (genJet_.size() > 0 ? &genJet_.front() : 0);
}


/// return the flavour of the parton underlying the jet
int Jet::partonFlavour() const {
  return partonFlavour_;
}


/// return the correction factor to go to a non-calibrated jet
JetCorrFactors Jet::jetCorrFactors() const {
  return jetCorrF_;
}


/// return the original non-calibrated jet
JetType Jet::recJet() const {
  JetType recJet(*this);
  recJet.setP4(noCorrF_*this->p4());
  return recJet;
}


/// return the associated non-calibrated jet
Jet Jet::noCorrJet() const {
  Jet jet(*this);
  jet.setP4(noCorrF_ * this->p4());
  // fix the factor to uncalibrate for the fact that we change the scale of the actual jet
  jet.setNoCorrFactor(1.);
  return jet;
}


/// return the associated default-calibrated jet
Jet Jet::defaultCorrJet() const {
  Jet jet(*this);
  jet.setP4(jetCorrF_.scaleDefault() * noCorrF_ * this->p4());
  // fix the factor to uncalibrate for the fact that we change the scale of the actual jet
  jet.setNoCorrFactor(1. / jetCorrF_.scaleDefault());
  return jet;
}


/// return the associated uds-calibrated jet
Jet Jet::udsCorrJet() const {
  Jet jet(*this);
  jet.setP4(jetCorrF_.scaleUds() * noCorrF_ * this->p4());
  // fix the factor to uncalibrate for the fact that we change the scale of the actual jet
  jet.setNoCorrFactor(1. / jetCorrF_.scaleUds());
  return jet;
}


/// return the associated gluon-calibrated jet
Jet Jet::gluCorrJet() const {
  Jet jet(*this);
  jet.setP4(jetCorrF_.scaleGlu() * noCorrF_ * this->p4());
  // fix the factor to uncalibrate for the fact that we change the scale of the actual jet
  jet.setNoCorrFactor(1. / jetCorrF_.scaleGlu());
  return jet;
}


/// return the associated c-calibrated jet
Jet Jet::cCorrJet() const {
  Jet jet(*this);
  jet.setP4(jetCorrF_.scaleC() * noCorrF_ * this->p4());
  // fix the factor to uncalibrate for the fact that we change the scale of the actual jet
  jet.setNoCorrFactor(1. / jetCorrF_.scaleC());
  return jet;
}


/// return the associated b-calibrated jet
Jet Jet::bCorrJet() const {
  Jet jet(*this);
  // set the corrected 4-vector
  jet.setP4(jetCorrF_.scaleB() * noCorrF_ * this->p4());
  // fix the factor to uncalibrate for the fact that we change the scale of the actual jet
  jet.setNoCorrFactor(1. / jetCorrF_.scaleB());
  // set the resolutions assuming this jet to be a b-jet
  jet.setResolutionA(bResA_);
  jet.setResolutionB(bResB_);
  jet.setResolutionC(bResC_);
  jet.setResolutionD(bResD_);
  jet.setResolutionEt(bResEt_);
  jet.setResolutionEta(bResEta_);
  jet.setResolutionPhi(bResPhi_);
  jet.setResolutionTheta(bResTheta_);
  jet.setCovMatrix(bCovM_);
  return jet;
}


/// return the jet calibrated according to the MC flavour truth
Jet Jet::mcFlavCorrJet() const {
  // determine the correction factor to use depending on MC flavour truth
  float corrF = jetCorrF_.scaleGlu(); // default, also for unidentified flavour
  if (abs(partonFlavour_) == 1 || abs(partonFlavour_) == 2 || abs(partonFlavour_) == 3) corrF = jetCorrF_.scaleUds();
  if (abs(partonFlavour_) == 4) corrF = jetCorrF_.scaleC();
  if (abs(partonFlavour_) == 5) corrF = jetCorrF_.scaleB();
  Jet jet(*this);
  jet.setP4(corrF * noCorrF_ * this->p4());
  // fix the factor to uncalibrate for the fact that we change the scale of the actual jet
  jet.setNoCorrFactor(1. / corrF);
  return jet;
}


/// return the jet calibrated with weights assuming W decay
Jet Jet::wCorrJet() const {
  Jet jet(*this);
  // set the corrected 4-vector weighting for the c-content in W decays
  jet.setP4((3*jetCorrF_.scaleUds() + jetCorrF_.scaleC()) / 4 * noCorrF_ * this->p4());
  // fix the factor to uncalibrate for the fact that we change the scale of the actual jet
  jet.setNoCorrFactor(4. / (3*jetCorrF_.scaleUds() + jetCorrF_.scaleC()));
  return jet;
}


/// get b discriminant from label name
float Jet::bDiscriminator(std::string theLabel) const {
  float discriminator = -10.;
  if (theLabel == "" || theLabel == "default") theLabel = "trackCountingJetTags";
  for(unsigned int i=0; i!=pairDiscriVector_.size(); i++){
    if(pairDiscriVector_[i].first == theLabel){
      discriminator = pairDiscriVector_[i].second;
    }
  }
  return discriminator;
}


/// get TagInfo ref IP
const std::vector<reco::TrackIPTagInfoRef> Jet::bTagIPTagInfoRef() const {
  return bTagIPTagInfoRef_;
}


/// get TagInfo ref soft letpon electron   
const std::vector<reco::SoftLeptonTagInfoRef> Jet::bTagSoftLeptonERef() const {
  return bTagSoftLeptonERef_;
}


/// get TagInfo ref soft letpon muon   
const std::vector<reco::SoftLeptonTagInfoRef> Jet::bTagSoftLeptonMRef() const {
  return bTagSoftLeptonMRef_;
}


/// get TagInfo ref SV  
const std::vector<reco::SecondaryVertexTagInfoRef> Jet::bTagSecondaryVertexTagInfoRef() const {
  return bTagSecondaryVertexTagInfoRef_;
}


/// get the value of the i'th jet cleaning variable
float Jet::lrPhysicsJetVar(unsigned int i) const {
  return (i < lrPhysicsJetVarVal_.size() ? lrPhysicsJetVarVal_[i].first  : 0);
}


/// get the likelihood ratio corresponding to the i'th jet cleaning variable
float Jet::lrPhysicsJetVal(unsigned int i) const {
  return (i < lrPhysicsJetVarVal_.size() ? lrPhysicsJetVarVal_[i].second : 1);
}


/// get the overall jet cleaning likelihood ratio
float Jet::lrPhysicsJetLRval() const {
  return lrPhysicsJetLRval_;
}


/// get the overall jet cleaning probability
float Jet::lrPhysicsJetProb() const {
  return lrPhysicsJetProb_;
}


/// method to return the JetCharge computed when creating the Jet
float Jet::jetCharge() const {
  return jetCharge_;
}


/// method to return a vector of refs to the tracks associated to this jet
const reco::TrackRefVector & Jet::associatedTracks() const {
  return associatedTracks_;
}


/// method to store the CaloJet constituents internally
void Jet::setCaloTowers(const std::vector<CaloTowerRef> & caloTowers) {
  for(unsigned int i = 0; i < caloTowers.size(); ++i) {
    caloTowers_.push_back(*caloTowers.at(i));
  }
  embeddedCaloTowers_ = true;
}


/// method to set the matched parton
void Jet::setGenParton(const reco::Particle & gp) {
  genParton_.clear();
  genParton_.push_back(gp);
}


/// method to set the matched generated jet
void Jet::setGenJet(const reco::GenJet & gj) {
  genJet_.clear();
  genJet_.push_back(gj);
}


/// method to set the flavour of the parton underlying the jet
void Jet::setPartonFlavour(int partonFl) {
  partonFlavour_ = partonFl;
}


/// method to set the energy scale correction factors
void Jet::setJetCorrFactors(const JetCorrFactors & jetCorrF) {
  jetCorrF_ = jetCorrF;
}


/// method to set correction factor to go back to an uncorrected jet
void Jet::setNoCorrFactor(float noCorrF) {
  noCorrF_ = noCorrF;
}


/// method to set the resolutions under the assumption this is a b-jet
void Jet::setBResolutions(float bResEt, float bResEta, float bResPhi, float bResA, float bResB, float bResC, float bResD, float bResTheta) {
  bResEt_ = bResEt;
  bResEta_ = bResEta;
  bResPhi_ = bResPhi;
  bResA_ = bResA;
  bResB_ = bResB;
  bResC_ = bResC;
  bResD_ = bResD;
  bResTheta_ = bResTheta;
}


/// method to add a algolabel-discriminator pair
void Jet::addBDiscriminatorPair(std::pair<std::string, float> & thePair) {
  pairDiscriVector_.push_back(thePair);
}


/// method to add tag ref IP taggers
void Jet::addBTagIPTagInfoRef(const reco::TrackIPTagInfoRef & tagRef) {
  bTagIPTagInfoRef_.push_back(tagRef);
}


/// method to add tag ref soft lepton taggers electron
void Jet::addBTagSoftLeptonERef(const reco::SoftLeptonTagInfoRef & tagRef) {
  bTagSoftLeptonERef_.push_back(tagRef);
}


/// method to add tag ref soft lepton taggers muon
void Jet::addBTagSoftLeptonMRef(const reco::SoftLeptonTagInfoRef & tagRef) {
  bTagSoftLeptonMRef_.push_back(tagRef);
}


/// method to add tag ref soft lepton taggers
void Jet::addBTagSecondaryVertexTagInfoRef(const reco::SecondaryVertexTagInfoRef & tagRef) {
  bTagSecondaryVertexTagInfoRef_.push_back(tagRef);
}


/// method to set all jet cleaning variable + LR pairs
void Jet::setLRPhysicsJetVarVal(const std::vector<std::pair<float, float> > & varValVec) {
  for (size_t i = 0; i<varValVec.size(); i++) lrPhysicsJetVarVal_.push_back(varValVec[i]);
}


/// method to set the combined jet cleaning likelihood ratio value
void Jet::setLRPhysicsJetLRval(float clr) {
  lrPhysicsJetLRval_ = clr;
}


/// method to set the jet cleaning probability
void Jet::setLRPhysicsJetProb(float plr) {
  lrPhysicsJetProb_ = plr;
}


/// method to set the jet charge
void Jet::setJetCharge(float jetCharge) {
  jetCharge_ = jetCharge;
}

/// correction factor from correction type
float
Jet::correctionFactor (CorrectionType type) const
{
  switch ( type ) {
  case NoCorrection :      return noCorrF_;
  case DefaultCorrection : return jetCorrF_.scaleDefault();
  case udsCorrection :     return jetCorrF_.scaleUds();
  case cCorrection :       return jetCorrF_.scaleC();
  case bCorrection :       return jetCorrF_.scaleB();
  case gCorrection :       return jetCorrF_.scaleGlu();
  default :                return jetCorrF_.scaleDefault();
  }
}

/// auxiliary method to convert a string to a correction type enum
Jet::CorrectionType
Jet::correctionType (const std::string& correctionName) 
{
  for ( unsigned int i=0; i<NrOfCorrections; ++i ) {
    if ( correctionName == correctionNames_[i] )  
      return static_cast<CorrectionType>(i);
  }
  // should include an error message ..
  edm::LogError("pat::Jet") << "Unknown correction type " << correctionName
			    << " - going to use default";
  return DefaultCorrection;
}

const std::string pat::Jet::correctionNames_[] = { "none", "default", 
						   "uds", "c", "b", "g" };
