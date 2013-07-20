// $Id: ErrorStreamSelector.cc,v 1.3 2009/07/20 13:07:27 mommsen Exp $
/// @file: ErrorStreamSelector.cc

#include "EventFilter/StorageManager/interface/ErrorStreamSelector.h"

using namespace stor;

bool ErrorStreamSelector::acceptEvent( const I2OChain& ioc )
{

  return true;

}
