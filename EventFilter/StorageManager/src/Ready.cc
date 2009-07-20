// $Id: Ready.cc,v 1.7 2009/07/10 11:41:03 dshpakov Exp $
/// @file: Ready.cc

#include "EventFilter/StorageManager/interface/Configuration.h"
#include "EventFilter/StorageManager/interface/ErrorStreamConfigurationInfo.h"
#include "EventFilter/StorageManager/interface/EventStreamConfigurationInfo.h"
#include "EventFilter/StorageManager/interface/SharedResources.h"
#include "EventFilter/StorageManager/interface/StateMachine.h"
#include "EventFilter/StorageManager/interface/Notifier.h"

#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"

#include <iostream>

using namespace std;
using namespace stor;

Ready::Ready( my_context c ): my_base(c)
{
  safeEntryAction( outermost_context().getNotifier() );
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
  try
    {
      sharedResources->_configuration->updateAllParams();
    }
  catch(...)
    {
      // To do: add logging:
      sharedResources->moveToFailedState( "exception while updating parameters in Ready entry action" );
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

  // configure the discard manager
  sharedResources->_discardManager->configure();
}

Ready::~Ready()
{
  safeExitAction( outermost_context().getNotifier() );
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

void Ready::do_moveToFailedState( const std::string& reason ) const
{
  outermost_context().getSharedResources()->moveToFailedState( reason );
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
