//
// $Id: TriggerPrimitive.cc,v 1.1.2.1 2008/04/08 09:40:53 vadler Exp $
//

#include "DataFormats/PatCandidates/interface/TriggerPrimitive.h"


using namespace pat;

/// default constructor
TriggerPrimitive::TriggerPrimitive() :
  reco::LeafCandidate(),
  triggerName_( "" ),
  filterName_( "" ) {
}

/// copy constructor
TriggerPrimitive::TriggerPrimitive( const TriggerPrimitive & aTrigPrim ) :
  reco::LeafCandidate( 0, aTrigPrim.p4() ),
  triggerName_( aTrigPrim.triggerName() ),
  filterName_( aTrigPrim.filterName() ) {
}

/// constructor from values
TriggerPrimitive::TriggerPrimitive( const reco::Particle::LorentzVector & aVec, const std::string aTrig, const std::string aFilt ) :
  reco::LeafCandidate( 0, aVec ),
  triggerName_( aTrig ),
  filterName_( aFilt ) {
}
TriggerPrimitive::TriggerPrimitive( const reco::Particle::PolarLorentzVector & aVec, const std::string aTrig, const std::string aFilt ) :
  reco::LeafCandidate( 0, aVec ),
  triggerName_( aTrig ),
  filterName_( aFilt ) {
}

/// destructor
TriggerPrimitive::~TriggerPrimitive() {
}

/// clone method
TriggerPrimitive * TriggerPrimitive::clone() const {
  return new TriggerPrimitive( * this );
}

/// return trigger name
const std::string & TriggerPrimitive::triggerName() const {
  return triggerName_;
}
const std::string & TriggerPrimitive::pathName() const {
  return triggerName();
}

/// return filter name
const std::string & TriggerPrimitive::filterName() const {
  return filterName_;
}
const std::string & TriggerPrimitive::objectType() const {
  return filterName();
}

/// set trigger name 
void TriggerPrimitive::setTriggerName( const std::string aTrig ) {
  triggerName_ = aTrig;
}
void TriggerPrimitive::setPathName( const std::string aPath ) {
  setTriggerName( aPath );
}

/// set filter name 
void TriggerPrimitive::setFilterName( const std::string aFilt ) {
  filterName_ = aFilt;
}
void TriggerPrimitive::setObjectType( const std::string anObj ) {
  setFilterName( anObj );
}
