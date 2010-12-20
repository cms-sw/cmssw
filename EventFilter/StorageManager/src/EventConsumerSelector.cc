// $Id: EventConsumerSelector.cc,v 1.11 2010/12/17 18:21:05 mommsen Exp $
/// @file: EventConsumerSelector.cc

#include <vector>

#include <boost/lambda/lambda.hpp>

#include "EventFilter/StorageManager/interface/EventConsumerSelector.h"
#include "EventFilter/StorageManager/interface/Exception.h"

#include "FWCore/Utilities/interface/EDMException.h"

using namespace stor;

void EventConsumerSelector::initialize( const InitMsgView& imv )
{

  if( _initialized ) return;

  if( _registrationInfo.outputModuleLabel() != imv.outputModuleLabel() ) return; 

  _outputModuleId = imv.outputModuleId();

  edm::ParameterSet pset;
  pset.addParameter<std::string>( "TriggerSelector", _registrationInfo.triggerSelection() );
  pset.addParameter<Strings>( "SelectEvents", _registrationInfo.eventSelection() );

  Strings tnames;
  imv.hltTriggerNames( tnames );

  std::ostringstream errorMsg;
  errorMsg << "Cannot initialize edm::EventSelector for consumer" <<
    _registrationInfo.consumerName() << " running on " << _registrationInfo.remoteHost() <<
    " requesting output module ID" << _outputModuleId <<
    " with label " << _registrationInfo.outputModuleLabel() <<
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

  _acceptedEvents = 0;
  _initialized = true;

}

bool EventConsumerSelector::acceptEvent( const I2OChain& ioc )
{

  if( !_initialized ) return false;
  if( _stale ) return false;

  if( ioc.outputModuleId() != _outputModuleId ) return false;

  std::vector<unsigned char> hlt_out;
  ioc.hltTriggerBits( hlt_out );

  if ( _eventSelector->wantAll()
    || _eventSelector->acceptEvent( &hlt_out[0], ioc.hltTriggerCount() ) )
  {
    if ( (++_acceptedEvents % _registrationInfo.prescale()) == 0 ) return true;
  }
  return false;
}

bool EventConsumerSelector::operator<(const EventConsumerSelector& other) const
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
