/**
 * $Id: SharedResources.cc,v 1.9.4.1 2011/03/07 11:33:05 mommsen Exp $
/// @file: SharedResources.cc
 */

#include "EventFilter/StorageManager/interface/Configuration.h"
#include "EventFilter/StorageManager/interface/DiscardManager.h"
#include "EventFilter/StorageManager/interface/DiskWriterResources.h"
#include "EventFilter/StorageManager/interface/DQMEventProcessorResources.h"
#include "EventFilter/StorageManager/interface/InitMsgCollection.h"
#include "EventFilter/StorageManager/interface/RegistrationCollection.h"
#include "EventFilter/StorageManager/interface/SharedResources.h"
#include "EventFilter/StorageManager/interface/StateMachine.h"
#include "EventFilter/StorageManager/interface/StatisticsReporter.h"

#include "xcept/tools.h"

#include "boost/date_time/posix_time/posix_time_types.hpp"

#include <fstream>
#include <iostream>
#include <unistd.h>


namespace stor
{

  void SharedResources::moveToFailedState( xcept::Exception& exception )
  {
    std::string errorMsg = "Failed to process FAIL exception: "
      + xcept::stdformat_exception_history(exception) + " due to ";

    try
    {
      statisticsReporter_->alarmHandler()->notifySentinel(AlarmHandler::FATAL, exception);
      statisticsReporter_->getStateMachineMonitorCollection().setStatusMessage( 
        xcept::stdformat_exception_history(exception)
      );
      EventPtr_t stMachEvent( new Fail() );
      // wait maximum 5 seconds until enqueuing succeeds
      if ( ! commandQueue_->enqTimedWait( stMachEvent, boost::posix_time::seconds(5) ) )
      {
        XCEPT_DECLARE_NESTED( stor::exception::StateTransition,
          sentinelException, "Failed to enqueue FAIL event", exception );
        statisticsReporter_->alarmHandler()->
          notifySentinel(AlarmHandler::FATAL, sentinelException);
      }
    }
    catch(xcept::Exception &e)
    {
      errorMsg += xcept::stdformat_exception_history(e);
      localDebug( errorMsg );
    }
    catch(std::exception &e)
    {
      errorMsg += e.what();
      localDebug( errorMsg );
    }
    catch( ... )
    {
      errorMsg += "an unknown exception.";
      localDebug( errorMsg );
    }
  }


  void SharedResources::localDebug( const std::string& message ) const
  {
    std::ostringstream fname_oss;
    fname_oss << "/tmp/storage_manager_debug_" << 
      configuration_->getDiskWritingParams().smInstanceString_ <<
      "_" << getpid();
    const std::string fname = fname_oss.str();
    std::ofstream f( fname.c_str(), std::ios_base::ate | std::ios_base::out | std::ios_base::app );
    if( f.is_open() )
    {
      try
      {
        f << message << std::endl;
        f.close();
      }
      catch(...)
      {}
    }
  }

    
} // namespace stor

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
