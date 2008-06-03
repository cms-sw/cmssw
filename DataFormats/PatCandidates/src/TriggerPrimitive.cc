//
// $Id: TriggerPrimitive.cc,v 1.1.4.2 2008/05/30 12:23:59 vadler Exp $
//

#include "DataFormats/PatCandidates/interface/TriggerPrimitive.h"


using namespace pat;

/// default constructor
TriggerPrimitive::TriggerPrimitive() :
  reco::LeafCandidate(),
  filterName_( "" ),
  triggerObjectType_( 0 ) {
}

/// copy constructor
TriggerPrimitive::TriggerPrimitive( const TriggerPrimitive & aTrigPrim ) :
  reco::LeafCandidate( 0, aTrigPrim.p4(), reco::Particle::Point( 0., 0., 0. ), aTrigPrim.triggerObjectId() ),
  filterName_( aTrigPrim.filterName() ),
  triggerObjectType_( aTrigPrim.triggerObjectType() ) {
}

/// constructor from values
TriggerPrimitive::TriggerPrimitive( const reco::Particle::LorentzVector & aVec, const std::string aFilt, const int aType, const int id ) :
  reco::LeafCandidate( 0, aVec, reco::Particle::Point( 0., 0., 0. ), id ),
  filterName_( aFilt ),
  triggerObjectType_( aType ) {
}
TriggerPrimitive::TriggerPrimitive( const reco::Particle::PolarLorentzVector & aVec, const std::string aFilt, const int aType, const int id ) :
  reco::LeafCandidate( 0, aVec, reco::Particle::Point( 0., 0., 0. ), id ),
  filterName_( aFilt ),
  triggerObjectType_( aType ) {
}

/// destructor
TriggerPrimitive::~TriggerPrimitive() {
}

/// clone method
TriggerPrimitive * TriggerPrimitive::clone() const {
  return new TriggerPrimitive( * this );
}

/// return filter name
const std::string & TriggerPrimitive::filterName() const {
  return filterName_;
}

/// return trigger object type
const int TriggerPrimitive::triggerObjectType() const {
  return triggerObjectType_;
}

/// return trigger object id
const int TriggerPrimitive::triggerObjectId() const {
  return pdgId();
}

/// set filter name 
void TriggerPrimitive::setFilterName( const std::string aFilt ) {
  filterName_ = aFilt;
}

/// set trigger object type
void TriggerPrimitive::setTriggerObjectType( const int aType ) {
  triggerObjectType_ = aType;
}

/// set trigger object id
void TriggerPrimitive::setTriggerObjectId( const int id ) {
  reco::Particle::setPdgId( id );
}
