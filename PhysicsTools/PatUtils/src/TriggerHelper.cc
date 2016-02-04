//
// $Id: TriggerHelper.cc,v 1.5 2011/04/05 19:41:33 vadler Exp $
//


#include "PhysicsTools/PatUtils/interface/TriggerHelper.h"

#include "DataFormats/Common/interface/AssociativeIterator.h"


using namespace pat;
using namespace pat::helper;



// Methods


// Get a reference to the trigger objects matched to a certain physics object given by a reference for a certain matcher module

// ... by resulting association
TriggerObjectRef TriggerMatchHelper::triggerMatchObject( const reco::CandidateBaseRef & candRef, const TriggerObjectMatch * matchResult, const edm::Event & event, const TriggerEvent & triggerEvent ) const
{
  if ( matchResult ) {
    edm::AssociativeIterator< reco::CandidateBaseRef, TriggerObjectMatch > it( *matchResult, edm::EdmEventItemGetter< reco::CandidateBaseRef >( event ) ), itEnd( it.end() );
    while ( it != itEnd ) {
      if ( it->first.isNonnull() && it->second.isNonnull() && it->second.isAvailable() ) {
        if ( it->first.id() == candRef.id() && it->first.key() == candRef.key() ) {
          return TriggerObjectRef( it->second );
        }
      }
      ++it;
    }
  }
  return TriggerObjectRef();
}

// ... by matcher module label
TriggerObjectRef TriggerMatchHelper::triggerMatchObject( const reco::CandidateBaseRef & candRef, const std::string & labelMatcher, const edm::Event & event, const TriggerEvent & triggerEvent ) const
{
  return triggerMatchObject( candRef, triggerEvent.triggerObjectMatchResult( labelMatcher ), event, triggerEvent );
}


// Get a table of references to all trigger objects matched to a certain physics object given by a reference
TriggerObjectMatchMap TriggerMatchHelper::triggerMatchObjects( const reco::CandidateBaseRef & candRef, const edm::Event & event, const TriggerEvent & triggerEvent ) const
{
  TriggerObjectMatchMap theContainer;
  const std::vector< std::string > matchers( triggerEvent.triggerMatchers() );
  for ( size_t iMatch = 0; iMatch < matchers.size(); ++iMatch ) {
    theContainer[ matchers.at( iMatch ) ] = triggerMatchObject( candRef, matchers.at( iMatch ), event, triggerEvent );
  }
  return theContainer;
}


// Get a vector of references to the phyics objects matched to a certain trigger object given by a reference for a certain matcher module

// ... by resulting association
reco::CandidateBaseRefVector TriggerMatchHelper::triggerMatchCandidates( const TriggerObjectRef & objectRef, const TriggerObjectMatch * matchResult, const edm::Event & event, const TriggerEvent & triggerEvent ) const
{
  reco::CandidateBaseRefVector theCands;
  if ( matchResult ) {
    edm::AssociativeIterator< reco::CandidateBaseRef, TriggerObjectMatch > it( *matchResult, edm::EdmEventItemGetter< reco::CandidateBaseRef >( event ) ), itEnd( it.end() );
    while ( it != itEnd ) {
      if ( it->first.isNonnull() && it->second.isNonnull() && it->second.isAvailable() ) {
        if ( it->second == objectRef ) {
          theCands.push_back( it->first );
        }
      }
      ++it;
    }
  }
  return theCands;
}

// ... by matcher module label
reco::CandidateBaseRefVector TriggerMatchHelper::triggerMatchCandidates( const TriggerObjectRef & objectRef, const std::string & labelMatcher, const edm::Event & event, const TriggerEvent & triggerEvent ) const
{
  return triggerMatchCandidates( objectRef, triggerEvent.triggerObjectMatchResult( labelMatcher ), event, triggerEvent );
}


// Get a vector of references to the phyics objects matched to a certain trigger object given by a collection and index for a certain matcher module

// ... by resulting association
reco::CandidateBaseRefVector TriggerMatchHelper::triggerMatchCandidates( const edm::Handle< TriggerObjectCollection > & trigCollHandle, const size_t iTrig, const TriggerObjectMatch * matchResult, const edm::Event & event, const TriggerEvent & triggerEvent ) const
{
  return triggerMatchCandidates( TriggerObjectRef( trigCollHandle, iTrig ), matchResult, event, triggerEvent );
}

// ... by matcher module label
reco::CandidateBaseRefVector TriggerMatchHelper::triggerMatchCandidates( const edm::Handle< TriggerObjectCollection > & trigCollHandle, const size_t iTrig, const std::string & labelMatcher, const edm::Event & event, const TriggerEvent & triggerEvent ) const
{
  return triggerMatchCandidates( TriggerObjectRef( trigCollHandle, iTrig ), triggerEvent.triggerObjectMatchResult( labelMatcher ), event, triggerEvent );
}
