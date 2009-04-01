//
// $Id: TriggerObjectStandAlone.cc,v 1.1.2.1 2009/03/27 21:34:45 vadler Exp $
//

#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"


using namespace pat;

/// methods

bool TriggerObjectStandAlone::hasFilterLabel( const std::string & filterLabel ) const
{
  for ( unsigned iFilter = 0; iFilter < filterLabels().size(); ++iFilter ) {
    if ( filterLabel == filterLabels().at( iFilter ) ) {
      return true;
    }
  }
  return false;
}

bool TriggerObjectStandAlone::hasPathName( const std::string & pathName ) const
{
  for ( unsigned iPath = 0; iPath < pathNames().size(); ++iPath ) {
    if ( pathName == pathNames().at( iPath ) ) {
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
