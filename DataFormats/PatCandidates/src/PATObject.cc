#include "DataFormats/PatCandidates/interface/PATObject.h"

using namespace pat;

void PATObject::setOriginalObjectRef(const edm::RefToBase<reco::Candidate> & ref)
{
  // correct way to convert RefToBase=>Ptr, if ref is guaranteed to be available
  // which happens to be true, otherwise the line before this throws ex. already
  refToOrig_ = edm::Ptr<reco::Candidate>(ref.id(), ref.get(), ref.key()) ;
}

void PATObject::setOriginalObjectRef(const edm::Ptr<reco::Candidate> & ref)
{
  refToOrig_ = ref;
}


const reco::Candidate * PATObject::originalObject() const {
  if (refToOrig_.isNull()) {
    // this object was not produced from a reference, so no link to the
    // original object exists -> return a 0-pointer
    return 0;
  } else if (!refToOrig_.isAvailable()) {
    throw edm::Exception(edm::errors::ProductNotFound) << "The original collection from which this PAT object was made is not present any more in the event, hence you cannot access the originating object anymore.";
  } else {
    return refToOrig_.get();
  }
}


const edm::Ptr<reco::Candidate> & PATObject::originalObjectRef() const { return refToOrig_; }


const TriggerObjectStandAloneCollection & PATObject::triggerObjectMatches() const { return triggerObjectMatchesEmbedded_; }


const TriggerObjectStandAloneCollection PATObject::triggerObjectMatchesByFilterID( const unsigned id ) const {
  TriggerObjectStandAloneCollection matches;
  for ( size_t i = 0; i < triggerObjectMatches().size(); ++i ) {
    if ( triggerObjectMatches().at( i ).hasFilterId( id ) ) matches.push_back( triggerObjectMatches().at( i ) );
  }
  return matches;
}


const TriggerObjectStandAloneCollection PATObject::triggerObjectMatchesByCollection( const std::string & coll ) const {
  TriggerObjectStandAloneCollection matches;
  for ( size_t i = 0; i < triggerObjectMatches().size(); ++i ) {
    if ( triggerObjectMatches().at( i ).collection() == coll ) matches.push_back( triggerObjectMatches().at( i ) );
  }
  return matches;
}


const TriggerObjectStandAloneCollection PATObject::triggerObjectMatchesByFilter( const std::string & labelFilter ) const {
  TriggerObjectStandAloneCollection matches;
  for ( size_t i = 0; i < triggerObjectMatches().size(); ++i ) {
    if ( triggerObjectMatches().at( i ).hasFilterLabel( labelFilter ) ) matches.push_back( triggerObjectMatches().at( i ) );
  }
  return matches;
}


const TriggerObjectStandAloneCollection PATObject::triggerObjectMatchesByPath( const std::string & namePath ) const {
  TriggerObjectStandAloneCollection matches;
  for ( size_t i = 0; i < triggerObjectMatches().size(); ++i ) {
    if ( triggerObjectMatches().at( i ).hasPathName( namePath ) ) matches.push_back( triggerObjectMatches().at( i ) );
  }
  return matches;
}


void PATObject::addTriggerObjectMatch( const TriggerObjectStandAlone & trigObj ) {
  triggerObjectMatchesEmbedded_.push_back( trigObj );
}


const pat::LookupTableRecord &  
PATObject::efficiency(const std::string &name) const {
  // find the name in the (sorted) list of names
  std::vector<std::string>::const_iterator it = std::lower_bound(efficiencyNames_.begin(), efficiencyNames_.end(), name);
  if ((it == efficiencyNames_.end()) || (*it != name)) {
    throw cms::Exception("Invalid Label") << "There is no efficiency with name '" << name << "' in this PAT Object\n";
  }
  return efficiencyValues_[it - efficiencyNames_.begin()];
}


std::vector<std::pair<std::string,pat::LookupTableRecord> > 
PATObject::efficiencies() const {
  std::vector<std::pair<std::string,pat::LookupTableRecord> > ret;
  std::vector<std::string>::const_iterator itn = efficiencyNames_.begin(), edn = efficiencyNames_.end();
  std::vector<pat::LookupTableRecord>::const_iterator itv = efficiencyValues_.begin();
  for ( ; itn != edn; ++itn, ++itv) {
    ret.push_back( std::pair<std::string,pat::LookupTableRecord>(*itn, *itv) );
  }
  return ret;
}


void PATObject::setEfficiency(const std::string &name, const pat::LookupTableRecord & value) {
  // look for the name, or to the place where we can insert it without violating the alphabetic order
  std::vector<std::string>::iterator it = std::lower_bound(efficiencyNames_.begin(), efficiencyNames_.end(), name);
  if (it == efficiencyNames_.end()) { // insert at the end
    efficiencyNames_.push_back(name);
    efficiencyValues_.push_back(value);
  } else if (*it == name) {           // replace existing
    efficiencyValues_[it - efficiencyNames_.begin()] = value;
  } else {                            // insert in the middle :-(
    efficiencyNames_. insert(it, name);
    efficiencyValues_.insert( efficiencyValues_.begin() + (it - efficiencyNames_.begin()), value );
  }
}


void PATObject::setGenParticleRef(const reco::GenParticleRef &ref, bool embed) {
  genParticleRef_ = std::vector<reco::GenParticleRef>(1,ref);
  genParticleEmbedded_.clear(); 
  if (embed) embedGenParticle();
}


void PATObject::addGenParticleRef(const reco::GenParticleRef &ref) {
  if (!genParticleEmbedded_.empty()) { // we're embedding
    if (ref.isNonnull()) genParticleEmbedded_.push_back(*ref);
  } else {
    genParticleRef_.push_back(ref);
  }
}
  

void PATObject::setGenParticle( const reco::GenParticle &particle ) {
  genParticleEmbedded_.clear(); 
  genParticleEmbedded_.push_back(particle);
  genParticleRef_.clear();
}


void PATObject::embedGenParticle() {
  genParticleEmbedded_.clear(); 
  for (std::vector<reco::GenParticleRef>::const_iterator it = genParticleRef_.begin(); it != genParticleRef_.end(); ++it) {
    if (it->isNonnull()) genParticleEmbedded_.push_back(**it); 
  }
  genParticleRef_.clear();
}


std::vector<reco::GenParticleRef> PATObject::genParticleRefs() const {
  if (genParticleEmbedded_.empty()) return genParticleRef_;
  std::vector<reco::GenParticleRef> ret(genParticleEmbedded_.size());
  for (size_t i = 0, n = ret.size(); i < n; ++i) {
    ret[i] = reco::GenParticleRef(&genParticleEmbedded_, i);
  }
  return ret;
}


reco::GenParticleRef PATObject::genParticleById(int pdgId, int status) const {
  // get a vector, avoiding an unneeded copy if there is no embedding
  const std::vector<reco::GenParticleRef> & vec = (genParticleEmbedded_.empty() ? genParticleRef_ : genParticleRefs());
  for (std::vector<reco::GenParticleRef>::const_iterator ref = vec.begin(), end = vec.end(); ref != end; ++ref) {
    if (ref->isNonnull() && ((*ref)->pdgId() == pdgId) && ((*ref)->status() == status)) return *ref;
  }
  return reco::GenParticleRef();
}


bool PATObject::hasOverlaps(const std::string &label) const {
  return std::find(overlapLabels_.begin(), overlapLabels_.end(), label) != overlapLabels_.end();
}


const reco::CandidatePtrVector & PATObject::overlaps(const std::string &label) const {
  static const reco::CandidatePtrVector EMPTY;
  std::vector<std::string>::const_iterator match = std::find(overlapLabels_.begin(), overlapLabels_.end(), label);
  if (match == overlapLabels_.end()) return EMPTY;
  return overlapItems_[match - overlapLabels_.begin()];
}


void PATObject::setOverlaps(const std::string &label, const reco::CandidatePtrVector & overlaps) {
  if (!overlaps.empty()) {
    std::vector<std::string>::const_iterator match = std::find(overlapLabels_.begin(), overlapLabels_.end(), label);
    if (match == overlapLabels_.end()) {
      overlapLabels_.push_back(label);
      overlapItems_.push_back(overlaps);
    } else {
      overlapItems_[match - overlapLabels_.begin()] = overlaps;
    }
  }
}

const pat::UserData * PATObject::userDataObject_( const std::string & key ) const
{
  std::vector<std::string>::const_iterator it = std::find(userDataLabels_.begin(), userDataLabels_.end(), key);
  if (it != userDataLabels_.end()) {
    return & userDataObjects_[it - userDataLabels_.begin()];
  }
  return 0;
}


float PATObject::userFloat( const std::string &key ) const
{
  std::vector<std::string>::const_iterator it = std::find(userFloatLabels_.begin(), userFloatLabels_.end(), key);
  if (it != userFloatLabels_.end()) {
    return userFloats_[it - userFloatLabels_.begin()];
  }
  return 0.0;
}

void PATObject::addUserFloat( const std::string & label,
					  float data )
{
  userFloatLabels_.push_back(label);
  userFloats_.push_back( data );
}


int PATObject::userInt( const std::string & key ) const
{
  std::vector<std::string>::const_iterator it = std::find(userIntLabels_.begin(), userIntLabels_.end(), key);
  if (it != userIntLabels_.end()) {
    return userInts_[it - userIntLabels_.begin()];
  }
  return 0;
}

void PATObject::addUserInt( const std::string &label,
					int data )
{
  userIntLabels_.push_back(label);
  userInts_.push_back( data );
}
