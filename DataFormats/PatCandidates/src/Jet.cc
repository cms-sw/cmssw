//
// $Id: Jet.cc,v 1.33 2009/09/29 17:39:40 srappocc Exp $
//

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

using namespace pat;


/// default constructor
Jet::Jet() :
  PATObject<reco::Jet>(reco::Jet()),
  embeddedCaloTowers_(false),
  partonFlavour_(0), 
  jetCharge_(0.)
{      
  jetID_.fHPD = 0.;
  jetID_.fRBX = 0.;
  jetID_.n90Hits = 0;
  jetID_.fSubDetector1 = 0.;
  jetID_.fSubDetector2 = 0.;
  jetID_.fSubDetector3 = 0.;
  jetID_.fSubDetector4 = 0.;
  jetID_.restrictedEMF = 0.;
  jetID_.nHCALTowers = 0;
  jetID_.nECALTowers = 0;  
}

/// constructor from a reco::Jet
Jet::Jet(const reco::Jet & aJet) :
  PATObject<reco::Jet>(aJet),
  embeddedCaloTowers_(false),
  partonFlavour_(0), 
  jetCharge_(0.0)
{
  jetID_.fHPD = 0.;
  jetID_.fRBX = 0.;
  jetID_.n90Hits = 0;
  jetID_.fSubDetector1 = 0.;
  jetID_.fSubDetector2 = 0.;
  jetID_.fSubDetector3 = 0.;
  jetID_.fSubDetector4 = 0.;
  jetID_.restrictedEMF = 0.;
  jetID_.nHCALTowers = 0;
  jetID_.nECALTowers = 0;
  tryImportSpecific(aJet);
}

/// constructor from ref to reco::Jet
Jet::Jet(const edm::Ptr<reco::Jet> & aJetRef) :
  PATObject<reco::Jet>(aJetRef),
  embeddedCaloTowers_(false),
  partonFlavour_(0), 
  jetCharge_(0.0)
{
  jetID_.fHPD = 0.;
  jetID_.fRBX = 0.;
  jetID_.n90Hits = 0;
  jetID_.fSubDetector1 = 0.;
  jetID_.fSubDetector2 = 0.;
  jetID_.fSubDetector3 = 0.;
  jetID_.fSubDetector4 = 0.;
  jetID_.restrictedEMF = 0.;
  jetID_.nHCALTowers = 0;
  jetID_.nECALTowers = 0;
  tryImportSpecific(*aJetRef);
}

/// constructor from ref to reco::Jet
Jet::Jet(const edm::RefToBase<reco::Jet> & aJetRef) :
  PATObject<reco::Jet>(aJetRef),
  embeddedCaloTowers_(false),
  partonFlavour_(0), 
  jetCharge_(0.0)
{
  jetID_.fHPD = 0.;
  jetID_.fRBX = 0.;
  jetID_.n90Hits = 0;
  jetID_.fSubDetector1 = 0.;
  jetID_.fSubDetector2 = 0.;
  jetID_.fSubDetector3 = 0.;
  jetID_.fSubDetector4 = 0.;
  jetID_.restrictedEMF = 0.;
  jetID_.nHCALTowers = 0;
  jetID_.nECALTowers = 0;
  tryImportSpecific(*aJetRef);
}

/// constructor helper that tries to import the specific info from the source jet
void Jet::tryImportSpecific(const reco::Jet &source) {
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

/// return the matched generated jet
const reco::GenJet * Jet::genJet() const {
  if (genJet_.size()) return  &(genJet_.front());
  return (genJetRef_.size() > 0 ? genJetRef_.begin()->get() : 0);
}

/// return the flavour of the parton underlying the jet
int Jet::partonFlavour() const {
  return partonFlavour_;
}

/// ============= Jet Energy Correction methods ============

/// method to set the energy scale correction factors
void Jet::setCorrFactors(const JetCorrFactors & jetCorrF) {
  jetEnergyCorrections_.clear();
  jetEnergyCorrections_.push_back(jetCorrF);
  activeJetCorrIndex_ = 0;
}

/// method to add more sets of energy scale correction factors
void Jet::addCorrFactors(const JetCorrFactors& jetCorrF) {
  jetEnergyCorrections_.push_back(jetCorrF);
}

/// method to set the energy scale correction factors
void Jet::setCorrStep(JetCorrFactors::CorrStep step) {
  jetEnergyCorrectionStep_ = step;
  setP4(corrFactors_()->correction( step ) * p4());
}

/// copy of the jet with correction factor to target step for
/// the set of correction factors, which is currently in use 
Jet Jet::correctedJet(const std::string& step, const std::string& flavour) const {
    Jet ret(*this);
    ret.setP4(p4() * corrFactors_()->correction(corrFactors_()->corrStep(step, flavour), jetEnergyCorrectionStep_));
    ret.jetEnergyCorrectionStep_ = corrFactors_()->corrStep(step, flavour);
    return ret;
}

/// copy of the jet with correction factor to target step for
/// the set of correction factors, which is currently in use 
Jet Jet::correctedJet(const JetCorrFactors::CorrStep& step) const {
    Jet ret(*this);
    ret.setP4(p4() * corrFactors_()->correction(step, jetEnergyCorrectionStep_));
    ret.jetEnergyCorrectionStep_ = step;
    return ret;
}

/// copy of the jet with correction factor to target step for
/// the set of correction factors, which is currently in use 
Jet Jet::correctedJet(const std::string& step, const std::string& flavour, const std::string& set) const {
    Jet ret(*this);
    const JetCorrFactors * jetCorrFac = corrFactors_(set);
    if (!jetCorrFac)
      throw cms::Exception("InvalidRequest") 
	<< "invalid JetCorrectionModule label '" << set << "' requested in Jet::correctedJet!";
    //uncorrect from jec factor from current set first; then correct starting from Raw from new set
    ret.setP4(p4() * corrFactors_()->correction( JetCorrFactors::Raw, jetEnergyCorrectionStep_) * jetCorrFac->correction(jetCorrFac->corrStep(step, flavour)) );
    ret.jetEnergyCorrectionStep_ = jetCorrFac->corrStep(step, flavour);
    return ret;
}

/// copy of the jet with correction factor to target step for
/// the set of correction factors, which is currently in use 
Jet Jet::correctedJet(const JetCorrFactors::CorrStep& step, const std::string& set) const {
    Jet ret(*this);
    const JetCorrFactors * jetCorrFac = corrFactors_(set);
    if (!jetCorrFac)
      throw cms::Exception("InvalidRequest") 
	<< "invalid JetCorrectionModule label '" << set << "' requested in Jet::correctedJet!";
    //uncorrect from jec factor from current set first; then correct starting from Raw from new set
    ret.setP4(p4() * corrFactors_()->correction( JetCorrFactors::Raw, jetEnergyCorrectionStep_) * jetCorrFac->correction(step) );
    ret.jetEnergyCorrectionStep_ = step;
    return ret;
}

/// return true if this jet carries the jet correction factors of a different set, for systematic studies
bool Jet::hasCorrFactorSet(const std::string& set) const 
{
  for (std::vector<pat::JetCorrFactors>::const_iterator it=jetEnergyCorrections_.begin();
       it!=jetEnergyCorrections_.end(); ++it)
    if (it->getLabel()==set)
      return true;
  return false;      
}

/// return the jet correction factors of a different set, for systematic studies
const JetCorrFactors * Jet::corrFactors_(const std::string& set) const 
{
  const JetCorrFactors * result = 0;
  for (std::vector<pat::JetCorrFactors>::const_iterator it=jetEnergyCorrections_.begin();
       it!=jetEnergyCorrections_.end(); ++it)
    if (it->getLabel()==set){
      result = &(*it);
      break;
    }  
  return result;      
}

/// return the correction factor for this jet. Throws if they're not available.
const JetCorrFactors * Jet::corrFactors_() const {
  return &jetEnergyCorrections_.at( activeJetCorrIndex_ );
}

/// return the name of the current level of jet energy corrections
std::string Jet::corrStep() const { 
  return corrFactors_()->corrStep( jetEnergyCorrectionStep_ ); 
}

/// return flavour of the current level of jet energy corrections
std::string Jet::corrFlavour() const {
  return corrFactors_()->flavour( jetEnergyCorrectionStep_ ); 
}

/// total correction factor to target step, starting from jetCorrStep(),
/// for the set of correction factors, which is currently in use
float Jet::corrFactor(const std::string& step, const std::string& flavour) const {
  return corrFactors_()->correction(corrFactors_()->corrStep(step, flavour), jetEnergyCorrectionStep_);
}

/// total correction factor to target step, starting from jetCorrStep(),
/// for a specific set of correction factors
float Jet::corrFactor(const std::string& step, const std::string& flavour, const std::string& set) const {
  const JetCorrFactors * jetCorrFac = corrFactors_(set);
  if (!jetCorrFac) 
  throw cms::Exception("InvalidRequest") 
  	<< "invalid JetCorrectionModule label '" << set << "' requested in Jet::jetCorrFactor!"<<std::endl;;
  return jetCorrFac->correction(jetCorrFac->corrStep(step, flavour), jetEnergyCorrectionStep_);
}

/// all available label-names of all sets of jet energy corrections
const std::vector<std::string> Jet::corrFactorSetLabels() const
{
  std::vector<std::string> result;
  for (std::vector<pat::JetCorrFactors>::const_iterator it=jetEnergyCorrections_.begin();
       it!=jetEnergyCorrections_.end(); ++it)
    result.push_back(it->getLabel());
  return result;      
}

const std::vector<std::pair<std::string, float> > & Jet::getPairDiscri() const {
   return pairDiscriVector_;
}

///returns uncertainty of currently applied jet-correction level 
///for plus or minus 1 sigma defined by direction ("UP" or "DOWN")
float Jet::relCorrUncert(const std::string& direction) const
{
   return corrFactors_()->uncertainty( jetEnergyCorrectionStep_, direction );
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
        return & tagInfos_[it - tagInfoLabels_.begin()];
    }
    return 0;
}

template<typename T> 
const T *  Jet::tagInfoByType() const {
    for (size_t i = 0, n = tagInfos_.size(); i < n; ++i) {
        if ( typeid(tagInfos_[i]) == typeid(T) )
             return static_cast<const T *>(&tagInfos_[i]);
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
    tagInfos_.push_back(info->clone());
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
  // possibly, if we really want to squeeze out bytes, we could clear the
  // daughters when the calotowers are embedded. The methods of the
  // CompositePtrCandidate that access this daughters would be srewed up though.
  // this->clearDaughters();
  embeddedCaloTowers_ = true;
}

/// method to set the matched generated jet reference, embedding if requested
void Jet::setGenJetRef(const reco::GenJetRef &ref, bool embed) {
  genJet_   .clear(); 
  genJetRef_.clear();
  if (embed)  genJet_   .push_back(*ref);
  else        genJetRef_.push_back(ref);
}


/// method to set the flavour of the parton underlying the jet
void Jet::setPartonFlavour(int partonFl) {
  partonFlavour_ = partonFl;
}

/// method to add a algolabel-discriminator pair
void Jet::addBDiscriminatorPair(const std::pair<std::string, float> & thePair) {
  pairDiscriVector_.push_back(thePair);
}

/// method to set the jet charge
void Jet::setJetCharge(float jetCharge) {
  jetCharge_ = jetCharge;
}
