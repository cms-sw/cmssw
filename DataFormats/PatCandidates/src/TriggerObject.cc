//
// $Id: TriggerObject.cc,v 1.6 2010/04/20 21:39:46 vadler Exp $
//

#include "DataFormats/PatCandidates/interface/TriggerObject.h"


using namespace pat;


/// default constructor

TriggerObject::TriggerObject() :
  reco::LeafCandidate()
{
  filterIds_.clear();
}

/// constructors from values

TriggerObject::TriggerObject( const reco::Particle::LorentzVector & vec, int id ) :
  reco::LeafCandidate( 0, vec, reco::Particle::Point( 0., 0., 0. ), id )
{
  filterIds_.clear();
}
TriggerObject::TriggerObject( const reco::Particle::PolarLorentzVector & vec, int id ) :
  reco::LeafCandidate( 0, vec, reco::Particle::Point( 0., 0., 0. ), id )
{
  filterIds_.clear();
}
TriggerObject::TriggerObject( const trigger::TriggerObject & trigObj ) :
  reco::LeafCandidate( 0, trigObj.particle().p4(), reco::Particle::Point( 0., 0., 0. ), trigObj.id() )
{
  filterIds_.clear();
}
TriggerObject::TriggerObject( const reco::LeafCandidate & leafCand ) :
  reco::LeafCandidate( leafCand )
{
  filterIds_.clear();
}

/// getters

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

bool TriggerObject::hasFilterId( int filterId ) const
{
  for ( size_t iF = 0; iF < filterIds().size(); ++iF ) {
    if ( filterIds().at( iF ) == filterId ) {
      return true;
    }
  }
  return false;
}
