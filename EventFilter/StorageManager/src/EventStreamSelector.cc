// $Id: EventStreamSelector.cc,v 1.14 2011/03/07 15:31:32 mommsen Exp $
/// @file: EventStreamSelector.cc

#include <cstdlib>
#include <ctime>
#include <vector>

#include <boost/lambda/lambda.hpp>

#include "EventFilter/StorageManager/interface/EventStreamSelector.h"
#include "EventFilter/StorageManager/interface/Exception.h"

#include "FWCore/Utilities/interface/EDMException.h"


namespace stor {
  
  EventStreamSelector::EventStreamSelector( const EventStreamConfigurationInfo& configInfo ):
  initialized_( false ),
  outputModuleId_(0),
  configInfo_( configInfo )
  {
    srand( time(0) );
  }
  
  
  void EventStreamSelector::initialize( const InitMsgView& imv )
  {
    
    if( initialized_ ) return;
    
    if( configInfo_.outputModuleLabel() != imv.outputModuleLabel() ) return; 
    
    outputModuleId_ = imv.outputModuleId();
    
    edm::ParameterSet pset;
    pset.addParameter<std::string>( "TriggerSelector", configInfo_.triggerSelection() );
    pset.addParameter<Strings>( "SelectEvents", configInfo_.eventSelection() );
    
    Strings tnames;
    imv.hltTriggerNames( tnames );
    
    std::ostringstream errorMsg;
    errorMsg << "Cannot initialize edm::EventSelector for stream " <<
      configInfo_.streamLabel() << " requesting output module ID" <<
      outputModuleId_ << " with label " << configInfo_.outputModuleLabel() <<
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
    
    initialized_ = true;
    
  }
  
  bool EventStreamSelector::acceptEvent( const I2OChain& ioc )
  {
    if( !initialized_ ) return false;
    
    if ( configInfo_.fractionToDisk() == 0 ) return false;
    
    if( ioc.outputModuleId() != outputModuleId_ ) return false;
    
    std::vector<unsigned char> hlt_out;
    ioc.hltTriggerBits( hlt_out );
    
    if ( ! eventSelector_->acceptEvent( &hlt_out[0], ioc.hltTriggerCount() ) )
      return false;
    
    if ( configInfo_.fractionToDisk() < 1 )
    {
      double rand = static_cast<double>(std::rand())/static_cast<double>(RAND_MAX);
      if ( rand > configInfo_.fractionToDisk() ) return false;
    }
    
    return true;
  }
  
} // namespace stor


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
