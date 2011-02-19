//
// $Id: TriggerEvent.cc,v 1.11 2010/09/24 21:16:41 vadler Exp $
//


#include "DataFormats/PatCandidates/interface/TriggerEvent.h"


using namespace pat;


/// constructor from values
TriggerEvent::TriggerEvent( const std::string & nameHltTable, bool run, bool accept, bool error, bool physDecl ) :
  nameHltTable_( nameHltTable ),
  run_( run ),
  accept_( accept ),
  error_( error ),
  physDecl_( physDecl )
{
  objectMatchResults_.clear();
}

TriggerEvent::TriggerEvent( const std::string & nameL1Menu, const std::string & nameHltTable, bool run, bool accept, bool error, bool physDecl ) :
  nameL1Menu_( nameL1Menu ),
  nameHltTable_( nameHltTable ),
  run_( run ),
  accept_( accept ),
  error_( error ),
  physDecl_( physDecl )
{
  objectMatchResults_.clear();
}


/// algorithms related

/// returns a NULL pointer, if the PAT trigger algorithm is not in the event
const TriggerAlgorithm * TriggerEvent::algorithm( const std::string & nameAlgorithm ) const
{
  for ( TriggerAlgorithmCollection::const_iterator iAlgorithm = algorithms()->begin(); iAlgorithm != algorithms()->end(); ++iAlgorithm ) {
    if ( nameAlgorithm == iAlgorithm->name() ) {
      return &*iAlgorithm;
    }
  }
  return 0;
}

/// returns the size of the PAT trigger algorithm collection, if the algorithm is not in the event
unsigned TriggerEvent::indexAlgorithm( const std::string & nameAlgorithm ) const
{
  unsigned iAlgorithm = 0;
  while ( iAlgorithm < algorithms()->size() && algorithms()->at( iAlgorithm ).name() != nameAlgorithm ) {
    ++iAlgorithm;
  }
  return iAlgorithm;
}

TriggerAlgorithmRefVector TriggerEvent::acceptedAlgorithms() const
{
  TriggerAlgorithmRefVector theAcceptedAlgorithms;
  for ( TriggerAlgorithmCollection::const_iterator iAlgorithm = algorithms()->begin(); iAlgorithm != algorithms()->end(); ++iAlgorithm ) {
    if ( iAlgorithm->decision() ) {
      const std::string nameAlgorithm( iAlgorithm->name() );
      const TriggerAlgorithmRef algorithmRef( algorithms(), indexAlgorithm( nameAlgorithm ) );
      theAcceptedAlgorithms.push_back( algorithmRef );
    }
  }
  return theAcceptedAlgorithms;
}

TriggerAlgorithmRefVector TriggerEvent::techAlgorithms() const
{
  TriggerAlgorithmRefVector theTechAlgorithms;
  for ( TriggerAlgorithmCollection::const_iterator iAlgorithm = algorithms()->begin(); iAlgorithm != algorithms()->end(); ++iAlgorithm ) {
    if ( iAlgorithm->techTrigger() ) {
      const std::string nameAlgorithm( iAlgorithm->name() );
      const TriggerAlgorithmRef algorithmRef( algorithms(), indexAlgorithm( nameAlgorithm ) );
      theTechAlgorithms.push_back( algorithmRef );
    }
  }
  return theTechAlgorithms;
}

TriggerAlgorithmRefVector TriggerEvent::acceptedTechAlgorithms() const
{
  TriggerAlgorithmRefVector theAcceptedTechAlgorithms;
  for ( TriggerAlgorithmCollection::const_iterator iAlgorithm = algorithms()->begin(); iAlgorithm != algorithms()->end(); ++iAlgorithm ) {
    if ( iAlgorithm->techTrigger() && iAlgorithm->decision() ) {
      const std::string nameAlgorithm( iAlgorithm->name() );
      const TriggerAlgorithmRef algorithmRef( algorithms(), indexAlgorithm( nameAlgorithm ) );
      theAcceptedTechAlgorithms.push_back( algorithmRef );
    }
  }
  return theAcceptedTechAlgorithms;
}

TriggerAlgorithmRefVector TriggerEvent::physAlgorithms() const
{
  TriggerAlgorithmRefVector thePhysAlgorithms;
  for ( TriggerAlgorithmCollection::const_iterator iAlgorithm = algorithms()->begin(); iAlgorithm != algorithms()->end(); ++iAlgorithm ) {
    if ( ! iAlgorithm->techTrigger() ) {
      const std::string nameAlgorithm( iAlgorithm->name() );
      const TriggerAlgorithmRef algorithmRef( algorithms(), indexAlgorithm( nameAlgorithm ) );
      thePhysAlgorithms.push_back( algorithmRef );
    }
  }
  return thePhysAlgorithms;
}

TriggerAlgorithmRefVector TriggerEvent::acceptedPhysAlgorithms() const
{
  TriggerAlgorithmRefVector theAcceptedPhysAlgorithms;
  for ( TriggerAlgorithmCollection::const_iterator iAlgorithm = algorithms()->begin(); iAlgorithm != algorithms()->end(); ++iAlgorithm ) {
    if ( ! iAlgorithm->techTrigger() && iAlgorithm->decision() ) {
      const std::string nameAlgorithm( iAlgorithm->name() );
      const TriggerAlgorithmRef algorithmRef( algorithms(), indexAlgorithm( nameAlgorithm ) );
      theAcceptedPhysAlgorithms.push_back( algorithmRef );
    }
  }
  return theAcceptedPhysAlgorithms;
}


/// paths related

/// returns a NULL pointer, if the PAT trigger path is not in the event
const TriggerPath * TriggerEvent::path( const std::string & namePath ) const
{
  for ( TriggerPathCollection::const_iterator iPath = paths()->begin(); iPath != paths()->end(); ++iPath ) {
    if ( namePath == iPath->name() ) {
      return &*iPath;
    }
  }
  return 0;
}

/// returns the size of the PAT trigger path collection, if the path is not in the event
unsigned TriggerEvent::indexPath( const std::string & namePath ) const
{
  unsigned iPath = 0;
  while ( iPath < paths()->size() && paths()->at( iPath ).name() != namePath ) {
    ++iPath;
  }
  return iPath;
}

TriggerPathRefVector TriggerEvent::acceptedPaths() const
{
  TriggerPathRefVector theAcceptedPaths;
  for ( TriggerPathCollection::const_iterator iPath = paths()->begin(); iPath != paths()->end(); ++iPath ) {
    if ( iPath->wasAccept() ) {
      const std::string namePath( iPath->name() );
      const TriggerPathRef pathRef( paths(), indexPath( namePath ) );
      theAcceptedPaths.push_back( pathRef );
    }
  }
  return theAcceptedPaths;
}

/// filters related

/// returns a NULL pointer, if the PAT trigger filter is not in the event
const TriggerFilter * TriggerEvent::filter( const std::string & labelFilter ) const
{
  for ( TriggerFilterCollection::const_iterator iFilter = filters()->begin(); iFilter != filters()->end(); ++iFilter ) {
    if ( iFilter->label() == labelFilter ) {
      return &*iFilter;
    }
  }
  return 0;
}

/// returns the size of the PAT trigger filter collection, if the filter is not in the event
unsigned TriggerEvent::indexFilter( const std::string & labelFilter ) const
{
  unsigned iFilter = 0;
  while ( iFilter < filters()->size() && filters()->at( iFilter ).label() != labelFilter ) {
    ++iFilter;
  }
  return iFilter;
}

TriggerFilterRefVector TriggerEvent::acceptedFilters() const
{
  TriggerFilterRefVector theAcceptedFilters;
  for ( TriggerFilterCollection::const_iterator iFilter = filters()->begin(); iFilter != filters()->end(); ++iFilter ) {
    if ( iFilter->status() == 1 ) {
      const std::string labelFilter( iFilter->label() );
      const TriggerFilterRef filterRef( filters(), indexFilter( labelFilter ) );
      theAcceptedFilters.push_back( filterRef );
    }
  }
  return theAcceptedFilters;
}

/// objects related

/// returns 'false', if a PAT trigger match with the given name exists already
bool TriggerEvent::addObjectMatchResult( const TriggerObjectMatchRefProd & trigMatches, const std::string & labelMatcher )
{
  if ( triggerObjectMatchResults()->find( labelMatcher ) == triggerObjectMatchResults()->end() ) {
    objectMatchResults_[ labelMatcher ] = trigMatches;
    return true;
  }
  return false;
}
bool TriggerEvent::addObjectMatchResult( const edm::Handle< TriggerObjectMatch > & trigMatches, const std::string & labelMatcher )
{
  return addObjectMatchResult( TriggerObjectMatchRefProd( trigMatches ), labelMatcher );
}
bool TriggerEvent::addObjectMatchResult( const edm::OrphanHandle< TriggerObjectMatch > & trigMatches, const std::string & labelMatcher )
{
  return addObjectMatchResult( TriggerObjectMatchRefProd( trigMatches ), labelMatcher );
}

TriggerObjectRefVector TriggerEvent::objects( int filterId ) const
{
  TriggerObjectRefVector theObjects;
  for ( unsigned iObject = 0; iObject < objects()->size(); ++iObject ) {
    if ( objects()->at( iObject ).hasFilterId( filterId ) ) {
      const TriggerObjectRef objectRef( objects(), iObject );
      theObjects.push_back( objectRef );
    }
  }
  return theObjects;
}

/// x-collection related

TriggerFilterRefVector TriggerEvent::pathModules( const std::string & namePath, bool all ) const
{
  TriggerFilterRefVector thePathFilters;
  if ( path( namePath ) && path( namePath )->modules().size() > 0 ) {
    const unsigned onePastLastFilter = all ? path( namePath )->modules().size() : path( namePath )->lastActiveFilterSlot() + 1;
    for ( unsigned iM = 0; iM < onePastLastFilter; ++iM ) {
      const std::string labelFilter( path( namePath )->modules().at( iM ) );
      const TriggerFilterRef filterRef( filters(), indexFilter( labelFilter ) ); // NULL, if filter was not in trigger::TriggerEvent
      thePathFilters.push_back( filterRef );
    }
  }
  return thePathFilters;
}

TriggerFilterRefVector TriggerEvent::pathFilters( const std::string & namePath ) const
{
  TriggerFilterRefVector thePathFilters;
  if ( path( namePath ) ) {
    for ( unsigned iF = 0; iF < path( namePath )->filterIndices().size(); ++iF ) {
      const TriggerFilterRef filterRef( filters(), path( namePath )->filterIndices().at( iF ) );
      thePathFilters.push_back( filterRef );
    }
  }
  return thePathFilters;
}

bool TriggerEvent::filterInPath( const TriggerFilterRef & filterRef, const std::string & namePath ) const
{
  TriggerFilterRefVector theFilters = pathFilters( namePath );
  for ( TriggerFilterRefVectorIterator iFilter = theFilters.begin(); iFilter != theFilters.end(); ++iFilter ) {
    if ( filterRef == *iFilter ) {
      return true;
    }
  }
  return false;
}

TriggerPathRefVector TriggerEvent::filterPaths( const TriggerFilterRef & filterRef ) const
{
  TriggerPathRefVector theFilterPaths;
  size_t cPaths( 0 );
  for ( TriggerPathCollection::const_iterator iPath = paths()->begin(); iPath != paths()->end(); ++iPath ) {
    const std::string namePath( iPath->name() );
    if ( filterInPath( filterRef, namePath ) ) {
      const TriggerPathRef pathRef( paths(), cPaths );
      theFilterPaths.push_back( pathRef );
    }
    ++cPaths;
  }
  return theFilterPaths;
}

std::vector< std::string > TriggerEvent::filterCollections( const std::string & labelFilter ) const
{
  std::vector< std::string > theFilterCollections;
  if ( filter( labelFilter ) ) {
    for ( unsigned iObject = 0; iObject < objects()->size(); ++iObject ) {
      if ( filter( labelFilter )->hasObjectKey( iObject ) ) {
        bool found( false );
        std::string objectCollection( objects()->at( iObject ).collection() );
        for ( std::vector< std::string >::const_iterator iC = theFilterCollections.begin(); iC != theFilterCollections.end(); ++iC ) {
          if ( *iC == objectCollection ) {
            found = true;
            break;
          }
        }
        if ( ! found ) {
          theFilterCollections.push_back( objectCollection );
        }
      }
    }
  }
  return theFilterCollections;
}

TriggerObjectRefVector TriggerEvent::filterObjects( const std::string & labelFilter ) const
{
  TriggerObjectRefVector theFilterObjects;
  if ( filter( labelFilter ) ) {
    for ( unsigned iObject = 0; iObject < objects()->size(); ++iObject ) {
      if ( filter( labelFilter )->hasObjectKey( iObject ) ) {
        const TriggerObjectRef objectRef( objects(), iObject );
        theFilterObjects.push_back( objectRef );
      }
    }
  }
  return theFilterObjects;
}

bool TriggerEvent::objectInFilter( const TriggerObjectRef & objectRef, const std::string & labelFilter ) const {
  if ( filter( labelFilter ) ) return filter( labelFilter )->hasObjectKey( objectRef.key() );
  return false;
}

TriggerFilterRefVector TriggerEvent::objectFilters( const TriggerObjectRef & objectRef ) const
{
  TriggerFilterRefVector theObjectFilters;
  for ( TriggerFilterCollection::const_iterator iFilter = filters()->begin(); iFilter != filters()->end(); ++iFilter ) {
    const std::string labelFilter( iFilter->label() );
    if ( objectInFilter( objectRef, labelFilter ) ) {
      const TriggerFilterRef filterRef( filters(), indexFilter( labelFilter ) );
      theObjectFilters.push_back( filterRef );
    }
  }
  return theObjectFilters;
}

TriggerObjectRefVector TriggerEvent::pathObjects( const std::string & namePath ) const
{
  TriggerObjectRefVector thePathObjects;
  TriggerFilterRefVector theFilters = pathFilters( namePath );
  for ( TriggerFilterRefVectorIterator iFilter = theFilters.begin(); iFilter != theFilters.end(); ++iFilter ) {
    const std::string labelFilter( ( *iFilter )->label() );
    TriggerObjectRefVector theObjects = filterObjects( labelFilter );
    for ( TriggerObjectRefVectorIterator iObject = theObjects.begin(); iObject != theObjects.end(); ++iObject ) {
      thePathObjects.push_back( *iObject );
    }
  }
  return thePathObjects;
}

bool TriggerEvent::objectInPath( const TriggerObjectRef & objectRef, const std::string & namePath ) const
{
  TriggerFilterRefVector theFilters = pathFilters( namePath );
  for ( TriggerFilterRefVectorIterator iFilter = theFilters.begin(); iFilter != theFilters.end(); ++iFilter ) {
    if ( objectInFilter( objectRef, ( *iFilter )->label() ) ) {
      return true;
    }
  }
  return false;
}

TriggerPathRefVector TriggerEvent::objectPaths( const TriggerObjectRef & objectRef ) const
{
  TriggerPathRefVector theObjectPaths;
  for ( TriggerPathCollection::const_iterator iPath = paths()->begin(); iPath != paths()->end(); ++iPath ) {
    const std::string namePath( iPath->name() );
    if ( objectInPath( objectRef, namePath ) ) {
      const TriggerPathRef pathRef( paths(), indexPath( namePath ) );
      theObjectPaths.push_back( pathRef );
    }
  }
  return theObjectPaths;
}

/// trigger matches

std::vector< std::string > TriggerEvent::triggerMatchers() const
{
  std::vector< std::string > theMatchers;
  for ( TriggerObjectMatchContainer::const_iterator iMatch = triggerObjectMatchResults()->begin(); iMatch != triggerObjectMatchResults()->end(); ++iMatch ) {
    theMatchers.push_back( iMatch->first );
  }
  return theMatchers;
}

const TriggerObjectMatch * TriggerEvent::triggerObjectMatchResult( const std::string & labelMatcher ) const
{
  const TriggerObjectMatchContainer::const_iterator iMatch( triggerObjectMatchResults()->find( labelMatcher ) );
  if ( iMatch != triggerObjectMatchResults()->end() ) {
    return iMatch->second.get();
  }
  return 0;
}
