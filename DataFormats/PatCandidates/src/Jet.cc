//
// $Id: Jet.cc,v 1.16 2008/05/26 11:22:13 arizzi Exp $
//

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

using namespace pat;


/// default constructor
Jet::Jet() :
  PATObject<JetType>(JetType()),
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
    tryImportSpecific(aJet);
}

/// constructor from ref to JetType
Jet::Jet(const edm::Ptr<JetType> & aJetRef) :
  PATObject<JetType>(aJetRef),
  embeddedCaloTowers_(false),
  partonFlavour_(0), lrPhysicsJetLRval_(-999.), lrPhysicsJetProb_(-1),
  jetCharge_(0.0) {
    tryImportSpecific(*aJetRef);
}

/// constructor from ref to JetType
Jet::Jet(const edm::RefToBase<JetType> & aJetRef) :
  PATObject<JetType>(aJetRef),
  embeddedCaloTowers_(false),
  partonFlavour_(0), lrPhysicsJetLRval_(-999.), lrPhysicsJetProb_(-1),
  jetCharge_(0.0) {
    tryImportSpecific(*aJetRef);
}

/// constructor helper that tries to import the specific info from the source jet
void Jet::tryImportSpecific(const JetType &source) {
    const std::type_info & type = typeid(source);
    if (type == typeid(reco::CaloJet)) {
        specificCalo_.push_back( (static_cast<const reco::CaloJet &>(source)).getSpecific() );
    } else if (type == typeid(reco::PFJet)) {
        specificPF_.push_back( (static_cast<const reco::PFJet &>(source)).getSpecific() );
    }
}

/// destructor
Jet::~Jet() {
}


/// ============= CaloJet methods ============

CaloTowerPtr Jet::getCaloConstituent (unsigned fIndex) const {
    if (embeddedCaloTowers_) {
        return (fIndex < caloTowers_.size() ? CaloTowerPtr(&caloTowers_, fIndex) : CaloTowerPtr());
    } else {
            Constituent dau = daughterPtr (fIndex);
	    const CaloTower* towerCandidate = dynamic_cast <const CaloTower*> (dau.get());
  	    if (towerCandidate) {
              return edm::Ptr<CaloTower> (dau.id(), towerCandidate, dau.key() );
            } 
   	    else {
      		throw cms::Exception("Invalid Constituent") << "CaloJet constituent is not of CaloTowere type";
	    }

    } 
   
   return CaloTowerPtr ();
}

std::vector<CaloTowerPtr> Jet::getCaloConstituents () const {
  std::vector<CaloTowerPtr> result;
  for (unsigned i = 0;  i <  numberOfDaughters (); i++) result.push_back (getCaloConstituent (i));
  return result;
}

/// ============= PFJet methods ============
const reco::PFCandidate* Jet::getPFCandidate (const reco::Candidate* fConstituent) {
  if (!fConstituent) return 0;
  const reco::Candidate* base = fConstituent;
  if (fConstituent->hasMasterClone ())
    base = fConstituent->masterClone().get();
  if (!base) return 0; // not in the event
  const reco::PFCandidate* candidate = dynamic_cast <const reco::PFCandidate*> (base);
  if (!candidate) {
    throw cms::Exception("Invalid Constituent") << "Jet constituent is not of PFCandidate type."
                                                << "Actual type is " << typeid (*base).name();
  }
  return candidate;
}

const reco::PFCandidate* Jet::getPFConstituent (unsigned fIndex) const {
  return getPFCandidate (daughter (fIndex));
}

std::vector <const reco::PFCandidate*> Jet::getPFConstituents () const {
  std::vector <const reco::PFCandidate*> result;
  for (unsigned i = 0;  i <  numberOfDaughters (); i++) result.push_back (getPFConstituent (i));
  return result;
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


const std::vector<std::pair<std::string, float> > & Jet::getPairDiscri() const {
   return pairDiscriVector_;
}


/// get b discriminant from label name
float Jet::bDiscriminator(const std::string & aLabel) const {
  float discriminator = -1000.;
  const std::string & theLabel = ((aLabel == "" || aLabel == "default")) ? "trackCountingHighEffBJetTags" : aLabel;
  for(unsigned int i=0; i!=pairDiscriVector_.size(); i++){
    if(pairDiscriVector_[i].first == theLabel){
      discriminator = pairDiscriVector_[i].second;
    }
  }
  return discriminator;
}

const reco::BaseTagInfo * Jet::tagInfo(const std::string &label) const {
    std::vector<std::string>::const_iterator it = std::find(tagInfoLabels_.begin(), tagInfoLabels_.end(), label);
    if (it != tagInfoLabels_.end()) {
        return tagInfos_[it - tagInfoLabels_.begin()].get();
    }
    return 0;
}
template<typename T> 
const T *  Jet::tagInfoByType() const {
    for (size_t i = 0, n = tagInfos_.size(); i < n; ++i) {
        if (tagInfos_[i].isAvailable() && (typeid(*tagInfos_[i]) == typeid(T)) )
             return static_cast<const T *>(tagInfos_[i].get());
    }
    return 0;
}
const reco::TrackIPTagInfo * 
Jet::tagInfoTrackIP(const std::string &label) const {
    return (label.empty() ? tagInfoByType<reco::TrackIPTagInfo>() 
                          : dynamic_cast<const reco::TrackIPTagInfo *>(tagInfo(label)) );
}
const reco::SoftLeptonTagInfo * 
Jet::tagInfoSoftLepton(const std::string &label) const {
    return (label.empty() ? tagInfoByType<reco::SoftLeptonTagInfo>()
                          : dynamic_cast<const reco::SoftLeptonTagInfo *>(tagInfo(label)) );
}
const reco::SecondaryVertexTagInfo * 
Jet::tagInfoSecondaryVertex(const std::string &label) const {
    return (label.empty() ? tagInfoByType<reco::SecondaryVertexTagInfo>()
                          : dynamic_cast<const reco::SecondaryVertexTagInfo *>(tagInfo(label)) );
}
void
Jet::addTagInfo(const std::string &label, const edm::Ptr<reco::BaseTagInfo> &info) {
    std::string::size_type idx = label.find("TagInfos");
    if (idx == std::string::npos) {
        tagInfoLabels_.push_back(label);
    } else {
        tagInfoLabels_.push_back(label.substr(0,idx));
    }
    tagInfos_.push_back(info);
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

/// method to set the vector of refs to the tracks associated to this jet
void Jet::setAssociatedTracks(const reco::TrackRefVector &tracks) {
    associatedTracks_ = tracks;
}

/// method to store the CaloJet constituents internally
void Jet::setCaloTowers(const std::vector<CaloTowerPtr> & caloTowers) {
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

#ifdef PATJet_OldTagInfo

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

#endif

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
  // No MessageLogger in DataFormats 
  throw cms::Exception("pat::Jet") << "Unknown correction type '" << correctionName << "' ";
}

const std::string pat::Jet::correctionNames_[] = { "none", "default", 
						   "uds", "c", "b", "g" };
