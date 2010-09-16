//
// $Id: Jet.cc,v 1.39 2010/06/15 19:18:55 srappocc Exp $
//

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace pat;

/// default constructor
Jet::Jet() :
  PATObject<reco::Jet>(reco::Jet()),
  embeddedCaloTowers_(false),
  embeddedPFCandidates_(false),
  partonFlavour_(0), 
  jetCharge_(0.),
  isCaloTowerCached_(false),
  isPFCandidateCached_(false)
{      
}

/// constructor from a reco::Jet
Jet::Jet(const reco::Jet & aJet) :
  PATObject<reco::Jet>(aJet),
  embeddedCaloTowers_(false),
  embeddedPFCandidates_(false),
  partonFlavour_(0), 
  jetCharge_(0.0),
  isCaloTowerCached_(false),
  isPFCandidateCached_(false)
{
  tryImportSpecific(aJet);
}

/// constructor from ref to reco::Jet
Jet::Jet(const edm::Ptr<reco::Jet> & aJetRef) :
  PATObject<reco::Jet>(aJetRef),
  embeddedCaloTowers_(false),
  embeddedPFCandidates_(false),
  partonFlavour_(0), 
  jetCharge_(0.0),
  isCaloTowerCached_(false),
  isPFCandidateCached_(false)
{
  tryImportSpecific(*aJetRef);
}

/// constructor from ref to reco::Jet
Jet::Jet(const edm::RefToBase<reco::Jet> & aJetRef) :
  PATObject<reco::Jet>(aJetRef),
  embeddedCaloTowers_(false),
  embeddedPFCandidates_(false),
  partonFlavour_(0), 
  jetCharge_(0.0),
  isCaloTowerCached_(false),
  isPFCandidateCached_(false)
{
  tryImportSpecific(*aJetRef);
}

/// constructor helper that tries to import the specific info from the source jet
void Jet::tryImportSpecific(const reco::Jet& source) 
{
  const std::type_info & type = typeid(source);
  if( type == typeid(reco::CaloJet) ){
    specificCalo_.push_back( (static_cast<const reco::CaloJet&>(source)).getSpecific() );
  } else if( type == typeid(reco::JPTJet) ){
    reco::JPTJet const & jptJet = static_cast<reco::JPTJet const &>(source);
    specificJPT_.push_back( jptJet.getSpecific() );
    reco::CaloJet const * caloJet = 0;
    if ( jptJet.getCaloJetRef().isNonnull() && jptJet.getCaloJetRef().isAvailable() ) {
      caloJet = dynamic_cast<reco::CaloJet const *>( jptJet.getCaloJetRef().get() );
    } 
    if ( caloJet != 0 ) {
      specificCalo_.push_back( caloJet->getSpecific() );
    } 
    else {
      edm::LogWarning("OptionalProductNotFound") << " in pat::Jet, Attempted to add Calo Specifics to JPT Jets, but failed."
						 << " Jet ID for JPT Jets will not be available for you." << std::endl;
    }
  } else if( type == typeid(reco::PFJet) ){
    specificPF_.push_back( (static_cast<const reco::PFJet&>(source)).getSpecific() );
  } 
}

/// destructor
Jet::~Jet() {
}

/// ============= CaloJet methods ============

CaloTowerPtr Jet::getCaloConstituent (unsigned fIndex) const {
    if (embeddedCaloTowers_) {
      // Refactorized PAT access
      if ( caloTowersFwdPtr_.size() > 0 ) {
	return (fIndex < caloTowersFwdPtr_.size() ? 
		caloTowersFwdPtr_[fIndex].ptr() : CaloTowerPtr());
      }
      // Compatibility PAT access
      else {
	if ( caloTowers_.size() > 0 ) {
	  return (fIndex < caloTowers_.size() ? 
		  CaloTowerPtr(&caloTowers_, fIndex) : CaloTowerPtr());

	}
      }
    }
    // Non-embedded access
    else {
      Constituent dau = daughterPtr (fIndex);
      const CaloTower* caloTower = dynamic_cast <const CaloTower*> (dau.get());
      if (caloTower != 0) {
	return CaloTowerPtr(dau.id(), caloTower, dau.key() );
      } 
      else {
	throw cms::Exception("Invalid Constituent") << "CaloJet constituent is not of CaloTower type";
      }
      
    } 
    
    return CaloTowerPtr ();
}



std::vector<CaloTowerPtr> const & Jet::getCaloConstituents () const { 
  if ( !isCaloTowerCached_ || caloTowers_.size() > 0 ) cacheCaloTowers();
  return caloTowersTemp_;
}


/// ============= PFJet methods ============

reco::PFCandidatePtr Jet::getPFConstituent (unsigned fIndex) const {
    if (embeddedPFCandidates_) {
      // Refactorized PAT access
      if ( pfCandidatesFwdPtr_.size() > 0 ) {
	return (fIndex < pfCandidatesFwdPtr_.size() ? 
		pfCandidatesFwdPtr_[fIndex].ptr() : reco::PFCandidatePtr());
      }
      // Compatibility PAT access
      else {
	if ( pfCandidates_.size() > 0 ) {
	  return (fIndex < pfCandidates_.size() ? 
		  reco::PFCandidatePtr(&pfCandidates_, fIndex) : reco::PFCandidatePtr());

	}
      }
    }
    // Non-embedded access
    else {
      Constituent dau = daughterPtr (fIndex);
      const reco::PFCandidate* pfCandidate = dynamic_cast <const reco::PFCandidate*> (dau.get());
      if (pfCandidate) {
	return reco::PFCandidatePtr(dau.id(), pfCandidate, dau.key() );
      } 
      else {
	throw cms::Exception("Invalid Constituent") << "PFJet constituent is not of PFCandidate type";
      }
      
    } 
    
    return reco::PFCandidatePtr ();
}

std::vector<reco::PFCandidatePtr> const & Jet::getPFConstituents () const { 
  if ( !isPFCandidateCached_ || pfCandidates_.size() > 0 ) cachePFCandidates();
  return pfCandidatesTemp_;
}



/// return the matched generated jet
const reco::GenJet * Jet::genJet() const {
  if (genJet_.size()) return  &(genJet_.front());
  else if ( genJetRef_.size() ) return genJetRef_[0].get();
  else return genJetFwdRef_.get();
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
      if ( tagInfosFwdPtr_.size() > 0 ) return tagInfosFwdPtr_[it - tagInfoLabels_.begin()].get();
      else if ( tagInfos_.size() > 0 )  return & tagInfos_[it - tagInfoLabels_.begin()];
      return 0;
    }
    return 0;
}


template<typename T> 
const T *  Jet::tagInfoByType() const {
  // First check the factorized PAT version
    for (size_t i = 0, n = tagInfosFwdPtr_.size(); i < n; ++i) {
      TagInfoFwdPtrCollection::value_type const & val = tagInfosFwdPtr_[i];
      reco::BaseTagInfo const * baseTagInfo = val.get();
      if ( typeid(*baseTagInfo) == typeid(T) ) {
	return static_cast<const T *>( baseTagInfo );
      }
    }
    // Then check compatibility version
    for (size_t i = 0, n = tagInfos_.size(); i < n; ++i) {
      edm::OwnVector<reco::BaseTagInfo>::value_type const & val = tagInfos_[i];
      reco::BaseTagInfo const * baseTagInfo = &val;
      if ( typeid(*baseTagInfo) == typeid(T) ) {
	return static_cast<const T *>( baseTagInfo );
      }
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
Jet::addTagInfo(const std::string &label, 
		const TagInfoFwdPtrCollection::value_type &info) {
    std::string::size_type idx = label.find("TagInfos");
    if (idx == std::string::npos) {
      tagInfoLabels_.push_back(label);
    } else {
        tagInfoLabels_.push_back(label.substr(0,idx));
    }
    tagInfosFwdPtr_.push_back(info);
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
void Jet::setCaloTowers(const CaloTowerFwdPtrCollection & caloTowers) {
  for(unsigned int i = 0; i < caloTowers.size(); ++i) {
    caloTowersFwdPtr_.push_back( caloTowers.at(i) );
  }
  embeddedCaloTowers_ = true;
  isCaloTowerCached_ = false;
}


/// method to store the CaloJet constituents internally
void Jet::setPFCandidates(const PFCandidateFwdPtrCollection & pfCandidates) {
  for(unsigned int i = 0; i < pfCandidates.size(); ++i) {
    pfCandidatesFwdPtr_.push_back(pfCandidates.at(i));
  }
  embeddedPFCandidates_ = true;
  isPFCandidateCached_ = false;
}


/// method to set the matched generated jet reference, embedding if requested
void Jet::setGenJetRef(const edm::FwdRef<reco::GenJetCollection> & gj)
{
  genJetFwdRef_ = gj;
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



/// method to cache the constituents to allow "user-friendly" access
void Jet::cacheCaloTowers() const {
  // Clear the cache
  caloTowersTemp_.clear();
  // Here is where we've embedded constituents
  if ( embeddedCaloTowers_ ) {
    // Refactorized PAT access
    if ( caloTowersFwdPtr_.size() > 0 ) {
      for ( CaloTowerFwdPtrVector::const_iterator ibegin=caloTowersFwdPtr_.begin(),
	      iend = caloTowersFwdPtr_.end(),
	      icalo = ibegin;
	    icalo != iend; ++icalo ) {
	caloTowersTemp_.push_back( CaloTowerPtr(icalo->ptr() ) );
      }      
    }
    // Compatibility access
    else if ( caloTowers_.size() > 0 ) {
      for ( CaloTowerCollection::const_iterator ibegin=caloTowers_.begin(),
	      iend = caloTowers_.end(),
	      icalo = ibegin;
	    icalo != iend; ++icalo ) {
	caloTowersTemp_.push_back( CaloTowerPtr(&caloTowers_, icalo - ibegin ) );
      }
    }
  } 
  // Non-embedded access
  else {
    for ( unsigned fIndex = 0; fIndex < numberOfDaughters(); ++fIndex ) {
      Constituent const & dau = daughterPtr (fIndex);
      const CaloTower* caloTower = dynamic_cast <const CaloTower*> (dau.get());
      if (caloTower) {
	caloTowersTemp_.push_back( CaloTowerPtr(dau.id(), caloTower,dau.key() ) );
      } 
      else {
	throw cms::Exception("Invalid Constituent") << "CaloJet constituent is not of CaloTower type";
      }
    }
  }
  // Set the cache flag
  isCaloTowerCached_=true;
}

/// method to cache the constituents to allow "user-friendly" access
void Jet::cachePFCandidates() const {
  // Clear the cache
  pfCandidatesTemp_.clear();
  // Here is where we've embedded constituents
  if ( embeddedPFCandidates_ ) {
    // Refactorized PAT access
    if ( pfCandidatesFwdPtr_.size() > 0 ) {
      for ( PFCandidateFwdPtrCollection::const_iterator ibegin=pfCandidatesFwdPtr_.begin(),
	      iend = pfCandidatesFwdPtr_.end(),
	      ipf = ibegin;
	    ipf != iend; ++ipf ) {
	pfCandidatesTemp_.push_back( reco::PFCandidatePtr(ipf->ptr() ) );
      }      
    }
    // Compatibility access
    else if ( pfCandidates_.size() > 0 ) {
      for ( reco::PFCandidateCollection::const_iterator ibegin=pfCandidates_.begin(),
	      iend = pfCandidates_.end(),
	      ipf = ibegin;
	    ipf != iend; ++ipf ) {
	pfCandidatesTemp_.push_back( reco::PFCandidatePtr(&pfCandidates_, ipf - ibegin ) );
      }
    }
  } 
  // Non-embedded access
  else {
    for ( unsigned fIndex = 0; fIndex < numberOfDaughters(); ++fIndex ) {
      Constituent const & dau = daughterPtr (fIndex);
      const reco::PFCandidate* pfCandidate = dynamic_cast <const reco::PFCandidate*> (dau.get());
      if (pfCandidate) {
	pfCandidatesTemp_.push_back( reco::PFCandidatePtr(dau.id(), pfCandidate,dau.key() ) );
      } 
      else {
	throw cms::Exception("Invalid Constituent") << "PFJet constituent is not of PFCandidate type";
      }
    }
  }
  // Set the cache flag
  isPFCandidateCached_=true;
}
