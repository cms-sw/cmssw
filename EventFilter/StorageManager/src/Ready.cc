// $Id: Ready.cc,v 1.18 2011/11/08 10:48:41 mommsen Exp $
/// @file: Ready.cc

#include "EventFilter/StorageManager/interface/AlarmHandler.h"
#include "EventFilter/StorageManager/interface/Configuration.h"
#include "EventFilter/StorageManager/interface/ErrorStreamConfigurationInfo.h"
#include "EventFilter/StorageManager/interface/EventStreamConfigurationInfo.h"
#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/DiscardManager.h"
#include "EventFilter/StorageManager/interface/SharedResources.h"
#include "EventFilter/StorageManager/interface/StateMachine.h"
#include "EventFilter/StorageManager/interface/StatisticsReporter.h"
#include "EventFilter/StorageManager/interface/TransitionRecord.h"

#include "xcept/tools.h"

#include <iostream>

using namespace std;
using namespace stor;

Ready::Ready( my_context c ): my_base(c)
{
  safeEntryAction();
}

void Ready::do_entryActionWork()
{
  TransitionRecord tr( stateName(), true );
  outermost_context().updateHistory( tr );

  SharedResourcesPtr sharedResources =
    outermost_context().getSharedResources();

  // update all configuration parameters
  std::string errorMsg = "Failed to update configuration parameters in Ready state";
  try
  {
    sharedResources->configuration_->updateAllParams();
  }
  catch(xcept::Exception &e)
  {
    XCEPT_DECLARE_NESTED(stor::exception::Configuration,
      sentinelException, errorMsg, e);
    sharedResources->alarmHandler_->moveToFailedState( sentinelException );
    return;
  }
  catch( std::exception &e )
  {
    errorMsg.append(": ");
    errorMsg.append( e.what() );

    XCEPT_DECLARE(stor::exception::Configuration,
      sentinelException, errorMsg);
    sharedResources->alarmHandler_->moveToFailedState( sentinelException );
    return;
  }
  catch(...)
  {
    errorMsg.append(": unknown exception");

    XCEPT_DECLARE(stor::exception::Configuration,
      sentinelException, errorMsg);
    sharedResources->alarmHandler_->moveToFailedState( sentinelException );
    return;
  }

  // configure the various queue sizes
  QueueConfigurationParams queueParams =
    sharedResources->configuration_->getQueueConfigurationParams();
  sharedResources->commandQueue_->
    setCapacity(queueParams.commandQueueSize_);
  sharedResources->fragmentQueue_->
    setCapacity(queueParams.fragmentQueueSize_);
  sharedResources->fragmentQueue_->
    setMemory(queueParams.fragmentQueueMemoryLimitMB_ * 1024*1024);
  sharedResources->registrationQueue_->
    setCapacity(queueParams.registrationQueueSize_);
  sharedResources->streamQueue_->
    setCapacity(queueParams.streamQueueSize_);
  sharedResources->streamQueue_->
    setMemory(queueParams.streamQueueMemoryLimitMB_ * 1024*1024);
  sharedResources->dqmEventQueue_->
    setCapacity(queueParams.dqmEventQueueSize_);
  sharedResources->dqmEventQueue_->
    setMemory(queueParams.dqmEventQueueMemoryLimitMB_ * 1024*1024);

  // convert the SM configuration string into ConfigInfo objects
  // and store them for later use
  DiskWritingParams dwParams =
    sharedResources->configuration_->getDiskWritingParams();
  EvtStrConfigListPtr evtCfgList(new EvtStrConfigList);
  ErrStrConfigListPtr errCfgList(new ErrStrConfigList);

  parseStreamConfiguration(dwParams.streamConfiguration_, evtCfgList,
                           errCfgList);
  sharedResources->configuration_->setCurrentEventStreamConfig(evtCfgList);
  sharedResources->configuration_->setCurrentErrorStreamConfig(errCfgList);

  // reset all alarms
  sharedResources->alarmHandler_->clearAllAlarms();

  // configure the disk monitoring
  ResourceMonitorCollection& rmc =
    sharedResources->statisticsReporter_->getResourceMonitorCollection();
  AlarmParams ap =
    sharedResources->configuration_->getAlarmParams();
  ResourceMonitorParams rmp =
    sharedResources->configuration_->getResourceMonitorParams();
  rmc.configureAlarms(ap);
  rmc.configureResources(rmp);
  rmc.configureDisks(dwParams);
  
  // configure the run monitoring
  RunMonitorCollection& run_mc =
    sharedResources->statisticsReporter_->getRunMonitorCollection();
  run_mc.configureAlarms(ap);

  // configure the discard manager
  sharedResources->discardManager_->configure();
}

Ready::~Ready()
{
  safeExitAction();
}

void Ready::do_exitActionWork()
{
  TransitionRecord tr( stateName(), false );
  outermost_context().updateHistory( tr );
}

string Ready::do_stateName() const
{
  return std::string( "Ready" );
}

void Ready::do_moveToFailedState( xcept::Exception& exception ) const
{
  outermost_context().getSharedResources()->alarmHandler_->moveToFailedState( exception );
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
