/**
 * $Id: SharedResources.cc,v 1.4 2009/07/20 13:07:28 mommsen Exp $
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


namespace stor
{

  void SharedResources::moveToFailedState( const std::string& reason )
  {
    _statisticsReporter->getStateMachineMonitorCollection().setStatusMessage( reason );
    event_ptr stMachEvent( new Fail() );
    // do we really want enq_wait here?
    // it could cause deadlock if the command queue is full...
    _commandQueue->enq_wait( stMachEvent );
  }  

} // namespace stor

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
