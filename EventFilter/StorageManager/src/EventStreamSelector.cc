// $Id: EventStreamSelector.cc,v 1.12 2010/02/08 11:57:59 mommsen Exp $
/// @file: EventStreamSelector.cc

#include <cstdlib>
#include <ctime>
#include <vector>

#include <boost/lambda/lambda.hpp>

#include "EventFilter/StorageManager/interface/EventStreamSelector.h"
#include "EventFilter/StorageManager/interface/Exception.h"

#include "FWCore/Utilities/interface/EDMException.h"

using namespace stor;


EventStreamSelector::EventStreamSelector( const EventStreamConfigurationInfo& configInfo ):
_initialized( false ),
_outputModuleId(0),
_configInfo( configInfo )
{
  srand( time(0) );
}


void EventStreamSelector::initialize( const InitMsgView& imv )
{

  if( _initialized ) return;

  if( _configInfo.outputModuleLabel() != imv.outputModuleLabel() ) return; 

  _outputModuleId = imv.outputModuleId();

  edm::ParameterSet pset;
  pset.addParameter<std::string>( "TriggerSelector", _configInfo.triggerSelection() );
  pset.addParameter<Strings>( "SelectEvents", _configInfo.eventSelection() );

  Strings tnames;
  imv.hltTriggerNames( tnames );

  std::ostringstream errorMsg;
  errorMsg << "Cannot initialize edm::EventSelector for stream " <<
    _configInfo.streamLabel() << " requesting output module ID" <<
    _outputModuleId << " with label " << _configInfo.outputModuleLabel() <<
    " and HLT trigger names";
  boost::lambda::placeholder1_type arg1;
  std::for_each(tnames.begin(), tnames.end(), errorMsg << boost::lambda::constant(" ") << arg1);
  try
  {
    _eventSelector.reset( new TriggerSelector( pset, tnames ) );
  }
  catch ( edm::Exception& e )
  {
    errorMsg << e.what();
    
    XCEPT_RAISE(stor::exception::InvalidEventSelection, errorMsg.str());
  }
  catch( std::exception &e )
  {
    errorMsg << e.what();

    XCEPT_RAISE(stor::exception::InvalidEventSelection, errorMsg.str());
  }
  catch(...)
  {
    errorMsg << "Unknown exception";

    XCEPT_RAISE(stor::exception::InvalidEventSelection, errorMsg.str());
  }

  _initialized = true;

}

bool EventStreamSelector::acceptEvent( const I2OChain& ioc )
{
  if( !_initialized ) return false;

  if ( _configInfo.fractionToDisk() == 0 ) return false;

  if( ioc.outputModuleId() != _outputModuleId ) return false;

  std::vector<unsigned char> hlt_out;
  ioc.hltTriggerBits( hlt_out );
  
  if ( ! _eventSelector->acceptEvent( &hlt_out[0], ioc.hltTriggerCount() ) )
    return false;

  if ( _configInfo.fractionToDisk() < 1 )
  {
    double rand = static_cast<double>(std::rand())/static_cast<double>(RAND_MAX);
    if ( rand > _configInfo.fractionToDisk() ) return false;
  }

  return true;
}

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
