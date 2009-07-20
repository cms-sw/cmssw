// $Id: ErrorStreamSelector.cc,v 1.2 2009/06/10 08:15:26 dshpakov Exp $
/// @file: ErrorStreamSelector.cc

#include "EventFilter/StorageManager/interface/ErrorStreamSelector.h"

using namespace stor;

bool ErrorStreamSelector::acceptEvent( const I2OChain& ioc )
{

  return true;

}
