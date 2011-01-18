//
// $Id: TriggerObjectStandAlone.cc,v 1.5 2010/12/14 10:46:18 gpetrucc Exp $
//

#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"
#include <algorithm>


using namespace pat;


/// Methods


void TriggerObjectStandAlone::addPathName( const std::string & pathName, bool pathLastFilterAccepted )
{
  if ( ! hasPathName( pathName, false ) ) {
    pathNames_.push_back( pathName );
    pathLastFilterAccepted_.push_back( pathLastFilterAccepted );
  }
}


std::vector< std::string > TriggerObjectStandAlone::pathNames( bool pathLastFilterAccepted ) const
{
  if ( ! pathLastFilterAccepted || ! hasPathLastFilterAccepted() ) return pathNames_;
  std::vector< std::string > paths;
  for ( unsigned iPath = 0; iPath < pathNames_.size(); ++iPath ) {
    if ( pathLastFilterAccepted_.at( iPath ) ) paths.push_back( pathNames_.at( iPath ) );
  }
  return paths;
}


TriggerObject TriggerObjectStandAlone::triggerObject()
{
  TriggerObject theObj( p4(), pdgId() );
  theObj.setCollection( collection() );
  for ( size_t i = 0; i < filterIds().size(); ++i ) theObj.addFilterId( filterIds().at( i ) );
  return theObj;
}


bool TriggerObjectStandAlone::hasPathName( const std::string & pathName, bool pathLastFilterAccepted ) const
{
  if ( ! hasPathLastFilterAccepted() ) pathLastFilterAccepted = false;
  std::vector< std::string >::const_iterator match( std::find( pathNames_.begin(), pathNames_.end(), pathName ) );
  if ( match == pathNames_.end() ) return false;
  return ( pathLastFilterAccepted ? pathLastFilterAccepted_.at( match - pathNames_.begin() ) : true );
}
