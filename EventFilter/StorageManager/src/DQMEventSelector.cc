// $Id: DQMEventSelector.cc,v 1.3 2009/07/20 13:07:27 mommsen Exp $
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


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
