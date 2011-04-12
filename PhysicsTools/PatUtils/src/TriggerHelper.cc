//
// $Id: TriggerHelper.cc,v 1.1.2.2 2009/06/16 21:21:24 vadler Exp $
//


#include "PhysicsTools/PatUtils/interface/TriggerHelper.h"

#include "DataFormats/Common/interface/AssociativeIterator.h"


using namespace pat;
using namespace pat::helper;



/// functions

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
TriggerObjectRef TriggerMatchHelper::triggerMatchObject( const reco::CandidateBaseRef & candRef, const std::string & labelMatcher, const edm::Event & event, const TriggerEvent & triggerEvent ) const
{
  return triggerMatchObject( candRef, triggerEvent.triggerObjectMatchResult( labelMatcher ), event, triggerEvent );
}

TriggerObjectMatchMap TriggerMatchHelper::triggerMatchObjects( const reco::CandidateBaseRef & candRef, const edm::Event & event, const TriggerEvent & triggerEvent ) const
{
  TriggerObjectMatchMap theContainer;
  const std::vector< std::string > matchers( triggerEvent.triggerMatchers() );
  for ( size_t iMatch = 0; iMatch < matchers.size(); ++iMatch ) {
    theContainer[ matchers.at( iMatch ) ] = triggerMatchObject( candRef, matchers.at( iMatch ), event, triggerEvent );
  }
  return theContainer;
}

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
reco::CandidateBaseRefVector TriggerMatchHelper::triggerMatchCandidates( const TriggerObjectRef & objectRef, const std::string & labelMatcher, const edm::Event & event, const TriggerEvent & triggerEvent ) const
{
  return triggerMatchCandidates( objectRef, triggerEvent.triggerObjectMatchResult( labelMatcher ), event, triggerEvent );
}
