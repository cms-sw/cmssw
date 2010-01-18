// $Id: Ready.cc,v 1.11 2009/09/29 07:57:56 mommsen Exp $
/// @file: Ready.cc

#include "EventFilter/StorageManager/interface/Configuration.h"
#include "EventFilter/StorageManager/interface/ErrorStreamConfigurationInfo.h"
#include "EventFilter/StorageManager/interface/EventStreamConfigurationInfo.h"
#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/DiscardManager.h"
#include "EventFilter/StorageManager/interface/SharedResources.h"
#include "EventFilter/StorageManager/interface/StateMachine.h"
#include "EventFilter/StorageManager/interface/StatisticsReporter.h"
#include "EventFilter/StorageManager/interface/TransitionRecord.h"

#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"

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

  if(!edmplugin::PluginManager::isAvailable()) {
    edmplugin::PluginManager::configure(edmplugin::standard::config());
  }

  SharedResourcesPtr sharedResources =
    outermost_context().getSharedResources();

  // update all configuration parameters
  std::string errorMsg = "Failed to update configuration parameters in Ready state";
  try
    {
      sharedResources->_configuration->updateAllParams();
    }
  // we don't have access to the sentinel here. Send an alarm instead.
  catch(xcept::Exception &e)
  {
    XCEPT_DECLARE_NESTED(stor::exception::Configuration,
      sentinelException, errorMsg, e);
    sharedResources->moveToFailedState( sentinelException );
    return;
  }
  catch( std::exception &e )
  {
    errorMsg.append(": ");
    errorMsg.append( e.what() );

    XCEPT_DECLARE(stor::exception::Configuration,
      sentinelException, errorMsg);
    sharedResources->moveToFailedState( sentinelException );
    return;
  }
  catch(...)
  {
    errorMsg.append(": unknown exception");

    XCEPT_DECLARE(stor::exception::Configuration,
      sentinelException, errorMsg);
    sharedResources->moveToFailedState( sentinelException );
    return;
  }

  // configure the various queue sizes
  QueueConfigurationParams queueParams =
    sharedResources->_configuration->getQueueConfigurationParams();
  sharedResources->_commandQueue->
    set_capacity(queueParams._commandQueueSize);
  sharedResources->_fragmentQueue->
    set_capacity(queueParams._fragmentQueueSize);
  sharedResources->_registrationQueue->
    set_capacity(queueParams._registrationQueueSize);
  sharedResources->_streamQueue->
    set_capacity(queueParams._streamQueueSize);

  // convert the SM configuration string into ConfigInfo objects
  // and store them for later use
  DiskWritingParams dwParams =
    sharedResources->_configuration->getDiskWritingParams();
  EvtStrConfigListPtr evtCfgList(new EvtStrConfigList);
  ErrStrConfigListPtr errCfgList(new ErrStrConfigList);

  parseStreamConfiguration(dwParams._streamConfiguration, evtCfgList,
                           errCfgList);
  sharedResources->_configuration->setCurrentEventStreamConfig(evtCfgList);
  sharedResources->_configuration->setCurrentErrorStreamConfig(errCfgList);

  // configure the disk monitoring
  ResourceMonitorCollection& rmc =
    sharedResources->_statisticsReporter->getResourceMonitorCollection();
  rmc.configureDisks(dwParams);
  ResourceMonitorParams rmp =
    sharedResources->_configuration->getResourceMonitorParams();
  rmc.configureResources(rmp);

  // configure the discard manager
  sharedResources->_discardManager->configure();
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
  return string( "Ready" );
}

void Ready::do_moveToFailedState( xcept::Exception& exception ) const
{
  outermost_context().getSharedResources()->moveToFailedState( exception );
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
