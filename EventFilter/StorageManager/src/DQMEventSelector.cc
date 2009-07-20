// $Id: DQMEventSelector.cc,v 1.2 2009/06/10 08:15:25 dshpakov Exp $
/// @file: DQMEventSelector.cc

#include "EventFilter/StorageManager/interface/DQMEventSelector.h"

using namespace stor;

bool DQMEventSelector::acceptEvent( const I2OChain& ioc )
{
  if( _stale ) return false;
  if( _topLevelFolderName == std::string( "*" ) ) return true;
  if( ioc.topFolderName() == _topLevelFolderName ) return true;
  return false;
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
