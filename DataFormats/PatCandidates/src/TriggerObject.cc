//
// $Id: TriggerObject.cc,v 1.12 2010/12/20 20:05:52 vadler Exp $
//

#include "DataFormats/PatCandidates/interface/TriggerObject.h"

#include "FWCore/Utilities/interface/EDMException.h"


using namespace pat;


// Constructors and Destructor


// Default constructor
TriggerObject::TriggerObject() :
  reco::LeafCandidate()
{
  triggerObjectTypes_.clear();
}


// Constructor from trigger::TriggerObject
TriggerObject::TriggerObject( const trigger::TriggerObject & trigObj ) :
  reco::LeafCandidate( 0, trigObj.particle().p4(), reco::Particle::Point( 0., 0., 0. ), trigObj.id() ),
  refToOrig_()
{
  triggerObjectTypes_.clear();
}


// Constructors from base class object
TriggerObject::TriggerObject( const reco::LeafCandidate & leafCand ) :
  reco::LeafCandidate( leafCand ),
  refToOrig_()
{
  triggerObjectTypes_.clear();
}


// Constructors from base candidate reference (for 'l1extra' particles)
TriggerObject::TriggerObject( const reco::CandidateBaseRef & candRef ) :
  reco::LeafCandidate( *candRef ),
  refToOrig_( candRef )
{
  triggerObjectTypes_.clear();
}


// Constructors from Lorentz-vectors and (optional) PDG ID
TriggerObject::TriggerObject( const reco::Particle::LorentzVector & vec, int id ) :
  reco::LeafCandidate( 0, vec, reco::Particle::Point( 0., 0., 0. ), id ),
  refToOrig_()
{
  triggerObjectTypes_.clear();
}
TriggerObject::TriggerObject( const reco::Particle::PolarLorentzVector & vec, int id ) :
  reco::LeafCandidate( 0, vec, reco::Particle::Point( 0., 0., 0. ), id ),
  refToOrig_()
{
  triggerObjectTypes_.clear();
}


// Methods


// Get all trigger object type identifiers
std::vector< int > TriggerObject::triggerObjectTypes() const
{
  std::vector< int > triggerObjectTypes;
  for ( size_t iTo = 0; iTo < triggerObjectTypes_.size(); ++iTo ) {
    triggerObjectTypes.push_back( triggerObjectTypes_.at( iTo ) );
  }
  return triggerObjectTypes;
}


// Checks, if a certain label of original collection is assigned
bool TriggerObject::hasCollection( const std::string & collName ) const
{
  // True, if collection name is simply fine
  if ( collName == collection_ ) return true;
  // Check, if collection name possibly fits in an edm::InputTag approach
  const edm::InputTag collectionTag( collection_ );
  const edm::InputTag collTag( collName );
  // If evaluated collection tag contains a process name, it must have been found already by identity check
  if ( collTag.process().empty() ) {
    // Check instance ...
    if ( ( collTag.instance().empty() && collectionTag.instance().empty() ) || collTag.instance() == collectionTag.instance() ) {
      // ... and label
      return ( collTag.label() == collectionTag.label() );
    }
  }
  return false;
}


// Checks, if a certain trigger object type identifier is assigned
bool TriggerObject::hasTriggerObjectType( trigger::TriggerObjectType triggerObjectType ) const
{
  for ( size_t iF = 0; iF < triggerObjectTypes_.size(); ++iF ) {
    if ( triggerObjectType == triggerObjectTypes_.at( iF ) ) return true;
  }
  return false;
}


// Special methods for 'l1extra' particles


// Getters specific to the 'l1extra' particle types
// Exceptions of type 'edm::errors::InvalidReference' are thrown,
// if wrong particle type is requested

// EM
const l1extra::L1EmParticleRef TriggerObject::origL1EmRef() const
{
  l1extra::L1EmParticleRef l1Ref;
  try {
    l1Ref = origObjRef().castTo< l1extra::L1EmParticleRef >();
  } catch ( edm::Exception X ) {
    if ( X.categoryCode() != edm::errors::InvalidReference ) throw X;
  }
  return l1Ref;
}

// EtMiss
const l1extra::L1EtMissParticleRef TriggerObject::origL1EtMissRef() const
{
  l1extra::L1EtMissParticleRef l1Ref;
  try {
    l1Ref = origObjRef().castTo< l1extra::L1EtMissParticleRef >();
  } catch ( edm::Exception X ) {
    if ( X.categoryCode() != edm::errors::InvalidReference ) throw X;
  }
  return l1Ref;
}

// Jet
const l1extra::L1JetParticleRef TriggerObject::origL1JetRef() const
{
  l1extra::L1JetParticleRef l1Ref;
  try {
    l1Ref = origObjRef().castTo< l1extra::L1JetParticleRef >();
  } catch ( edm::Exception X ) {
    if ( X.categoryCode() != edm::errors::InvalidReference ) throw X;
  }
  return l1Ref;
}

// Muon
const l1extra::L1MuonParticleRef TriggerObject::origL1MuonRef() const
{
  l1extra::L1MuonParticleRef l1Ref;
  try {
    l1Ref = origObjRef().castTo< l1extra::L1MuonParticleRef >();
  } catch ( edm::Exception X ) {
    if ( X.categoryCode() != edm::errors::InvalidReference ) throw X;
  }
  return l1Ref;
}
