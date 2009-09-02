// $Id: EventStreamSelector.cc,v 1.3 2009/07/20 13:07:27 mommsen Exp $
/// @file: EventStreamSelector.cc

#include <vector>

#include "EventFilter/StorageManager/interface/EventStreamSelector.h"

using namespace stor;

void EventStreamSelector::initialize( const InitMsgView& imv )
{

  if( _initialized ) return;

  if( _configInfo.outputModuleLabel() != imv.outputModuleLabel() ) return; 

  _outputModuleId = imv.outputModuleId();

  edm::ParameterSet pset;
  pset.addParameter<Strings>( "SelectEvents", _configInfo.selEvents() );

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

bool EventStreamSelector::acceptEvent( const I2OChain& ioc )
{

  if( !_initialized ) return false;

  if( ioc.outputModuleId() != _outputModuleId ) return false;

  std::vector<unsigned char> hlt_out;
  ioc.hltTriggerBits( hlt_out );

  return _eventSelector->wantAll()
    || _eventSelector->acceptEvent( &hlt_out[0], ioc.hltTriggerCount() );

}
