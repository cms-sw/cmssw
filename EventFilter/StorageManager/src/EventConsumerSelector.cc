// $Id: EventConsumerSelector.cc,v 1.13 2011/03/07 15:31:32 mommsen Exp $
/// @file: EventConsumerSelector.cc

#include <vector>

#include <boost/lambda/lambda.hpp>

#include "EventFilter/StorageManager/interface/EventConsumerSelector.h"
#include "EventFilter/StorageManager/interface/Exception.h"

#include "FWCore/Utilities/interface/EDMException.h"

using namespace stor;

void EventConsumerSelector::initialize( const InitMsgView& imv )
{

  if( initialized_ ) return;

  if( registrationInfo_->outputModuleLabel() != imv.outputModuleLabel() ) return; 

  outputModuleId_ = imv.outputModuleId();

  edm::ParameterSet pset;
  pset.addParameter<std::string>( "TriggerSelector", registrationInfo_->triggerSelection() );
  pset.addParameter<Strings>( "SelectEvents", registrationInfo_->eventSelection() );

  Strings tnames;
  imv.hltTriggerNames( tnames );

  std::ostringstream errorMsg;
  errorMsg << "Cannot initialize edm::EventSelector for consumer" <<
    registrationInfo_->consumerName() << " running on " << registrationInfo_->remoteHost() <<
    " requesting output module ID" << outputModuleId_ <<
    " with label " << registrationInfo_->outputModuleLabel() <<
    " and HLT trigger names";
  boost::lambda::placeholder1_type arg1;
  std::for_each(tnames.begin(), tnames.end(), errorMsg << boost::lambda::constant(" ") << arg1);
  try
  {
    eventSelector_.reset( new TriggerSelector( pset, tnames ) );
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

  acceptedEvents_ = 0;
  initialized_ = true;

}

bool EventConsumerSelector::acceptEvent( const I2OChain& ioc )
{

  if( !initialized_ ) return false;

  if( ioc.outputModuleId() != outputModuleId_ ) return false;

  std::vector<unsigned char> hlt_out;
  ioc.hltTriggerBits( hlt_out );

  if ( eventSelector_->wantAll()
    || eventSelector_->acceptEvent( &hlt_out[0], ioc.hltTriggerCount() ) )
  {
    if ( (++acceptedEvents_ % registrationInfo_->prescale()) == 0 ) return true;
  }
  return false;
}

bool EventConsumerSelector::operator<(const EventConsumerSelector& other) const
{
  if ( queueId() != other.queueId() )
    return ( queueId() < other.queueId() );
  return ( *(registrationInfo_) < *(other.registrationInfo_) );
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
