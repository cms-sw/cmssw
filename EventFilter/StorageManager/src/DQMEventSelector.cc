// $Id: DQMEventSelector.cc,v 1.4 2010/12/16 16:35:29 mommsen Exp $
/// @file: DQMEventSelector.cc

#include "EventFilter/StorageManager/interface/DQMEventSelector.h"

using namespace stor;

bool DQMEventSelector::acceptEvent( const I2OChain& ioc )
{
  if( _stale ) return false;
  if( _registrationInfo.topLevelFolderName() == std::string( "*" ) ) return true;
  if( _registrationInfo.topLevelFolderName() == ioc.topFolderName() ) return true;
  return false;
}


bool DQMEventSelector::operator<(const DQMEventSelector& other) const
{
  if ( queueId() != other.queueId() )
    return ( queueId() < other.queueId() );
  return ( _registrationInfo < other._registrationInfo );
}



/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
