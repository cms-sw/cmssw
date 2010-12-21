//
// $Id: TriggerObject.cc,v 1.9 2010/12/11 21:25:44 vadler Exp $
//

#include "DataFormats/PatCandidates/interface/TriggerObject.h"

#include "FWCore/Utilities/interface/EDMException.h"


using namespace pat;


/// Constructors and Destructor


TriggerObject::TriggerObject() :
  reco::LeafCandidate()
{
  filterIds_.clear();
}


TriggerObject::TriggerObject( const reco::Particle::LorentzVector & vec, int id ) :
  reco::LeafCandidate( 0, vec, reco::Particle::Point( 0., 0., 0. ), id ),
  refToOrig_()
{
  filterIds_.clear();
}


TriggerObject::TriggerObject( const reco::Particle::PolarLorentzVector & vec, int id ) :
  reco::LeafCandidate( 0, vec, reco::Particle::Point( 0., 0., 0. ), id ),
  refToOrig_()
{
  filterIds_.clear();
}


TriggerObject::TriggerObject( const trigger::TriggerObject & trigObj ) :
  reco::LeafCandidate( 0, trigObj.particle().p4(), reco::Particle::Point( 0., 0., 0. ), trigObj.id() ),
  refToOrig_()
{
  filterIds_.clear();
}


TriggerObject::TriggerObject( const reco::LeafCandidate & leafCand ) :
  reco::LeafCandidate( leafCand ),
  refToOrig_()
{
  filterIds_.clear();
}


TriggerObject::TriggerObject( const reco::CandidateBaseRef & candRef ) :
  reco::LeafCandidate( *candRef ),
  refToOrig_( candRef )
{
  filterIds_.clear();
}


/// Methods


bool TriggerObject::hasCollection( const std::string & coll ) const
{
  if ( collection() == coll ) return true;
  const edm::InputTag collectionTag( collection() );
  const edm::InputTag collTag( coll );
  if ( collTag.process().empty() ) {
    if ( ( collTag.instance().empty() && collectionTag.instance().empty() ) || collTag.instance() == collectionTag.instance() ) {
      if ( collTag.label() == collectionTag.label() ) return true;
    }
  }
  return false;
}


bool TriggerObject::hasFilterId( trigger::TriggerObjectType filterId ) const
{
  for ( size_t iF = 0; iF < filterIds().size(); ++iF ) {
    if ( filterIds().at( iF ) == filterId ) {
      return true;
    }
  }
  return false;
}


/// Special methods for 'l1extra' particles


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
