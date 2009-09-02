// $Id: EventConsumerSelector.cc,v 1.3 2009/07/20 13:07:27 mommsen Exp $
/// @file: EventConsumerSelector.cc

#include <vector>

#include "EventFilter/StorageManager/interface/EventConsumerSelector.h"

using namespace stor;

void EventConsumerSelector::initialize( const InitMsgView& imv )
{

  if( _initialized ) return;

  if( _outputModuleLabel != imv.outputModuleLabel() ) return; 

  _outputModuleId = imv.outputModuleId();

  edm::ParameterSet pset;
  pset.addParameter<Strings>( "SelectEvents", _eventSelectionStrings );

  Strings tnames;
  imv.hltTriggerNames( tnames );

  // 18-Aug-2009, KAB: protect against exceptions that can occur
  // when creating the edm::EventSelector (such as an invalid trigger
  // path request).
  try
  {
    _eventSelector.reset( new edm::EventSelector( pset, tnames ) );
  }
  catch ( ... )
  {
    // we should add some logging here!
    return;
  }

  _initialized = true;

}

bool EventConsumerSelector::acceptEvent( const I2OChain& ioc )
{

  if( !_initialized ) return false;
  if( _stale ) return false;

  if( ioc.outputModuleId() != _outputModuleId ) return false;

  std::vector<unsigned char> hlt_out;
  ioc.hltTriggerBits( hlt_out );

  return _eventSelector->wantAll()
    || _eventSelector->acceptEvent( &hlt_out[0], ioc.hltTriggerCount() );

}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
