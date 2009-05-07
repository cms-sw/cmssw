//
// $Id: TriggerObject.cc,v 1.1.2.2 2009/03/15 12:24:13 vadler Exp $
//

#include "DataFormats/PatCandidates/interface/TriggerObject.h"


using namespace pat;

/// default constructor

TriggerObject::TriggerObject() :
  reco::LeafCandidate()
{
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
  reco::LeafCandidate( trigObj.particle() )
{
  filterIds_.clear();
}

/// getters

bool TriggerObject::hasFilterId( unsigned filterId ) const
{
  for ( unsigned iF = 0; iF < filterIds().size(); ++iF ) {
    if ( filterIds().at( iF ) == filterId ) {
      return true;
    }
  }
  return false;
}
