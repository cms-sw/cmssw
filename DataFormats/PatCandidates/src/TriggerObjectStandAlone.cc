//
// $Id: TriggerObjectStandAlone.cc,v 1.3 2010/04/20 21:39:46 vadler Exp $
//

#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"


using namespace pat;


/// methods

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

bool TriggerObjectStandAlone::hasFilterLabel( const std::string & filterLabel ) const
{
  for ( unsigned iFilter = 0; iFilter < filterLabels().size(); ++iFilter ) {
    if ( filterLabel == filterLabels().at( iFilter ) ) {
      return true;
    }
  }
  return false;
}

bool TriggerObjectStandAlone::hasPathName( const std::string & pathName, bool pathLastFilterAccepted ) const
{
  for ( unsigned iPath = 0; iPath < pathNames( pathLastFilterAccepted ).size(); ++iPath ) {
    if ( pathName == pathNames( pathLastFilterAccepted ).at( iPath ) ) {
      return true;
    }
  }
  return false;
}

// returns "pure" pat::TriggerObject w/o add-on
TriggerObject TriggerObjectStandAlone::triggerObject()
{
  TriggerObject theObj( p4(), pdgId() );
  theObj.setCollection( collection() );
  for ( size_t i = 0; i < filterIds().size(); ++i ) theObj.addFilterId( filterIds().at( i ) );
  return theObj;
}
