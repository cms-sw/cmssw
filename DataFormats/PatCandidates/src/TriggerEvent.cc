//
// $Id: TriggerEvent.cc,v 1.17 2010/12/15 19:44:28 vadler Exp $
//


#include "DataFormats/PatCandidates/interface/TriggerEvent.h"


using namespace pat;


// Constructors and Destructor


// Constructor from values, HLT only
TriggerEvent::TriggerEvent( const std::string & nameHltTable, bool run, bool accept, bool error, bool physDecl ) :
  nameHltTable_( nameHltTable ),
  run_( run ),
  accept_( accept ),
  error_( error ),
  physDecl_( physDecl )
{
  objectMatchResults_.clear();
}


// Constructor from values, HLT and L1/GT
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


// Methods


// Get a pointer to a certain L1 algorithm by name
const TriggerAlgorithm * TriggerEvent::algorithm( const std::string & nameAlgorithm ) const
{
  for ( TriggerAlgorithmCollection::const_iterator iAlgorithm = algorithms()->begin(); iAlgorithm != algorithms()->end(); ++iAlgorithm ) {
    if ( nameAlgorithm == iAlgorithm->name() ) {
      return &*iAlgorithm;
    }
  }
  return 0;
}


// Get the index of a certain L1 algorithm in the event collection by name
unsigned TriggerEvent::indexAlgorithm( const std::string & nameAlgorithm ) const
{
  unsigned iAlgorithm( 0 );
  while ( iAlgorithm < algorithms()->size() && algorithms()->at( iAlgorithm ).name() != nameAlgorithm ) {
    ++iAlgorithm;
  }
  return iAlgorithm;
}


// Get a vector of references to all succeeding L1 algorithms
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


// Get a vector of references to all technical L1 algorithms
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


// Get a vector of references to all succeeding technical L1 algorithms
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


// Get a vector of references to all physics L1 algorithms
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


// Get a vector of references to all succeeding physics L1 algorithms
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


// Get a pointer to a certain HLT path by name
const TriggerPath * TriggerEvent::path( const std::string & namePath ) const
{
  for ( TriggerPathCollection::const_iterator iPath = paths()->begin(); iPath != paths()->end(); ++iPath ) {
    if ( namePath == iPath->name() ) {
      return &*iPath;
    }
  }
  return 0;
}


// Get the index of a certain HLT path in the event collection by name
unsigned TriggerEvent::indexPath( const std::string & namePath ) const
{
  unsigned iPath( 0 );
  while ( iPath < paths()->size() && paths()->at( iPath ).name() != namePath ) {
    ++iPath;
  }
  return iPath;
}


// Get a vector of references to all succeeding HLT paths
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


// Get a pointer to a certain HLT filter by label
const TriggerFilter * TriggerEvent::filter( const std::string & labelFilter ) const
{
  for ( TriggerFilterCollection::const_iterator iFilter = filters()->begin(); iFilter != filters()->end(); ++iFilter ) {
    if ( iFilter->label() == labelFilter ) {
      return &*iFilter;
    }
  }
  return 0;
}


// Get the index of a certain HLT filter in the event collection by label
unsigned TriggerEvent::indexFilter( const std::string & labelFilter ) const
{
  unsigned iFilter( 0 );
  while ( iFilter < filters()->size() && filters()->at( iFilter ).label() != labelFilter ) {
    ++iFilter;
  }
  return iFilter;
}


// Get a vector of references to all succeeding HLT filters
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


// Add a pat::TriggerObjectMatch association
bool TriggerEvent::addObjectMatchResult( const TriggerObjectMatchRefProd & trigMatches, const std::string & labelMatcher )
{
  if ( triggerObjectMatchResults()->find( labelMatcher ) == triggerObjectMatchResults()->end() ) {
    objectMatchResults_[ labelMatcher ] = trigMatches;
    return true;
  }
  return false;
}


// Get a vector of references to all trigger objects by trigger object type
TriggerObjectRefVector TriggerEvent::objects( trigger::TriggerObjectType triggerObjectType ) const
{
  TriggerObjectRefVector theObjects;
  for ( unsigned iObject = 0; iObject < objects()->size(); ++iObject ) {
    if ( objects()->at( iObject ).hasTriggerObjectType( triggerObjectType ) ) {
      const TriggerObjectRef objectRef( objects(), iObject );
      theObjects.push_back( objectRef );
    }
  }
  return theObjects;
}


// Get a vector of references to all modules assigned to a certain path given by name
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


// Get a vector of references to all active HLT filters assigned to a certain path given by name
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


// Checks, if a filter is assigned to and was run in a certain path given by name
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


// Get a vector of references to all paths, which have a certain filter assigned
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


// Get a list of all trigger object collections used in a certain filter given by name
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


// Get a vector of references to all objects, which were used in a certain filter given by name
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


// Checks, if an object was used in a certain filter given by name
bool TriggerEvent::objectInFilter( const TriggerObjectRef & objectRef, const std::string & labelFilter ) const {
  if ( filter( labelFilter ) ) return filter( labelFilter )->hasObjectKey( objectRef.key() );
  return false;
}


// Get a vector of references to all filters, which have a certain object assigned
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


// Get a vector of references to all objects, which wree used in a certain path given by name
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


// Checks, if an object was used in a certain path given by name
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


// Get a vector of references to all paths, which have a certain object assigned
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


// Get a list of all linked trigger matches
std::vector< std::string > TriggerEvent::triggerMatchers() const
{
  std::vector< std::string > theMatchers;
  for ( TriggerObjectMatchContainer::const_iterator iMatch = triggerObjectMatchResults()->begin(); iMatch != triggerObjectMatchResults()->end(); ++iMatch ) {
    theMatchers.push_back( iMatch->first );
  }
  return theMatchers;
}


// Get a pointer to a certain trigger match given by label
const TriggerObjectMatch * TriggerEvent::triggerObjectMatchResult( const std::string & labelMatcher ) const
{
  const TriggerObjectMatchContainer::const_iterator iMatch( triggerObjectMatchResults()->find( labelMatcher ) );
  if ( iMatch != triggerObjectMatchResults()->end() ) {
    return iMatch->second.get();
  }
  return 0;
}
