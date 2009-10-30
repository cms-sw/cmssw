// $Id: EventStreamSelector.cc,v 1.6 2009/10/13 15:08:34 mommsen Exp $
/// @file: EventStreamSelector.cc

#include <vector>

#include <boost/lambda/lambda.hpp>

#include "EventFilter/StorageManager/interface/EventStreamSelector.h"
#include "EventFilter/StorageManager/interface/Exception.h"

#include "FWCore/Utilities/interface/EDMException.h"

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

  std::ostringstream errorMsg;
  errorMsg << "Cannot initialize edm::EventSelector for stream " <<
    _configInfo.streamLabel() << " requesting output module ID" <<
    _outputModuleId << " with label " << _configInfo.outputModuleLabel() <<
    " and HLT trigger names";
  boost::lambda::placeholder1_type arg1;
  std::for_each(tnames.begin(), tnames.end(), errorMsg << boost::lambda::constant(" ") << arg1);
  try
  {
    _eventSelector.reset( new edm::EventSelector( pset, tnames ) );
  }
<<<<<<< EventStreamSelector.cc
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
=======
  catch ( edm::Exception& e )
>>>>>>> 1.4.4.3
  {
<<<<<<< EventStreamSelector.cc
    errorMsg << "Unknown exception";

    XCEPT_RAISE(stor::exception::InvalidEventSelection, errorMsg.str());
=======
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
>>>>>>> 1.4.4.3
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

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
