//
// $Id: Jet.cc,v 1.44 2011/06/08 20:40:19 rwolf Exp $
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

std::ostream& 
reco::operator<<(std::ostream& out, const pat::Jet& obj) 
{
  if(!out) return out;
  
  out << "\tpat::Jet: ";
  out << std::setiosflags(std::ios::right);
  out << std::setiosflags(std::ios::fixed);
  out << std::setprecision(3);
  out << " E/pT/eta/phi " 
      << obj.energy()<<"/"
      << obj.pt()<<"/"
      << obj.eta()<<"/"
      << obj.phi();
  return out; 
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

// initialize the jet to a given JEC level during creation starting from Uncorrected
void Jet::initializeJEC(unsigned int level, const JetCorrFactors::Flavor& flavor, unsigned int set)
{
  currentJECSet(set);
  currentJECLevel(level);
  currentJECFlavor(flavor);
  setP4(jec_[set].correction(level, flavor)*p4());
}

/// return true if this jet carries the jet correction factors of a different set, for systematic studies
int Jet::jecSet(const std::string& set) const
{
  for(std::vector<pat::JetCorrFactors>::const_iterator corrFactor=jec_.begin(); corrFactor!=jec_.end(); ++corrFactor)
    if( corrFactor->jecSet()==set ){ return (corrFactor-jec_.begin()); }
  return -1;
}

/// all available label-names of all sets of jet energy corrections
const std::vector<std::string> Jet::availableJECSets() const
{
  std::vector<std::string> sets;
  for(std::vector<pat::JetCorrFactors>::const_iterator corrFactor=jec_.begin(); corrFactor!=jec_.end(); ++corrFactor)
    sets.push_back(corrFactor->jecSet());
  return sets;
}

const std::vector<std::string> Jet::availableJECLevels(const int& set) const
{
  return set>=0 ? jec_.at(set).correctionLabels() : std::vector<std::string>();
}

/// correction factor to the given level for a specific set
/// of correction factors, starting from the current level
float Jet::jecFactor(const std::string& level, const std::string& flavor, const std::string& set) const
{
  for(unsigned int idx=0; idx<jec_.size(); ++idx){
    if(set.empty() || jec_.at(idx).jecSet()==set){
      if(jec_[idx].jecLevel(level)>=0){
	return jecFactor(jec_[idx].jecLevel(level), jec_[idx].jecFlavor(flavor), idx);
      }
      else{
	throw cms::Exception("InvalidRequest") << "This JEC level " << level << " does not exist. \n";
      }
    }
  }
  throw cms::Exception("InvalidRequest") << "This jet does not carry any jet energy correction factor information \n"
					 << "for a jet energy correction set with label " << set << "\n";
}

/// correction factor to the given level for a specific set
/// of correction factors, starting from the current level
float Jet::jecFactor(const unsigned int& level, const JetCorrFactors::Flavor& flavor, const unsigned int& set) const
{
  if(!jecSetsAvailable()){
    throw cms::Exception("InvalidRequest") << "This jet does not carry any jet energy correction factor information \n";
  }
  if(!jecSetAvailable(set)){
    throw cms::Exception("InvalidRequest") << "This jet does not carry any jet energy correction factor information \n"
					   << "for a jet energy correction set with index " << set << "\n";
  }
  return jec_.at(set).correction(level, flavor)/jec_.at(currentJECSet_).correction(currentJECLevel_, currentJECFlavor_);
}

/// copy of the jet with correction factor to target step for
/// the set of correction factors, which is currently in use
Jet Jet::correctedJet(const std::string& level, const std::string& flavor, const std::string& set) const
{
  // rescale p4 of the jet; the update of current values is
  // done within the called jecFactor function
  for(unsigned int idx=0; idx<jec_.size(); ++idx){
    if(set.empty() || jec_.at(idx).jecSet()==set){
      if(jec_[idx].jecLevel(level)>=0){
	return correctedJet(jec_[idx].jecLevel(level), jec_[idx].jecFlavor(flavor), idx);
      }
      else{
	throw cms::Exception("InvalidRequest") << "This JEC level " << level << " does not exist. \n";
      }
    }
  }
  throw cms::Exception("InvalidRequest") << "This JEC set " << set << " does not exist. \n";
}

/// copy of the jet with correction factor to target step for
/// the set of correction factors, which is currently in use
Jet Jet::correctedJet(const unsigned int& level, const JetCorrFactors::Flavor& flavor, const unsigned int& set) const
{
  Jet correctedJet(*this);
  //rescale p4 of the jet
  correctedJet.setP4(jecFactor(level, flavor, set)*p4());
  // update current level, flavor and set
  correctedJet.currentJECSet(set); correctedJet.currentJECLevel(level); correctedJet.currentJECFlavor(flavor);
  return correctedJet;
}


/// ============= BTag information methods ============

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
