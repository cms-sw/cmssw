// $Id: Ready.cc,v 1.3 2009/06/17 09:40:50 dshpakov Exp $

#include "EventFilter/StorageManager/interface/Configuration.h"
#include "EventFilter/StorageManager/interface/ErrorStreamConfigurationInfo.h"
#include "EventFilter/StorageManager/interface/EventStreamConfigurationInfo.h"
#include "EventFilter/StorageManager/interface/SharedResources.h"
#include "EventFilter/StorageManager/interface/StateMachine.h"
#include "EventFilter/StorageManager/interface/Notifier.h"

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
      sharedResources->moveToFailedState();
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


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
