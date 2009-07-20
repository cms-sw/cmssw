// $Id: EventStreamSelector.cc,v 1.2 2009/06/10 08:15:26 dshpakov Exp $
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
  _eventSelector.reset( new edm::EventSelector( pset, tnames ) );

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
