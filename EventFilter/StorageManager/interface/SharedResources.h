// $Id: SharedResources.h,v 1.2 2009/06/10 08:15:23 dshpakov Exp $

#ifndef StorageManager_SharedResources_h
#define StorageManager_SharedResources_h

#include <string>

#include "boost/shared_ptr.hpp"

#include "EventFilter/StorageManager/interface/CommandQueue.h"
#include "EventFilter/StorageManager/interface/Configuration.h"
#include "EventFilter/StorageManager/interface/DiscardManager.h"
#include "EventFilter/StorageManager/interface/DiskWriterResources.h"
#include "EventFilter/StorageManager/interface/DQMEventProcessorResources.h"
#include "EventFilter/StorageManager/interface/DQMEventQueue.h"
#include "EventFilter/StorageManager/interface/DQMEventQueueCollection.h"
#include "EventFilter/StorageManager/interface/EventQueueCollection.h"
#include "EventFilter/StorageManager/interface/FragmentQueue.h"
#include "EventFilter/StorageManager/interface/InitMsgCollection.h"
#include "EventFilter/StorageManager/interface/RegistrationCollection.h"
#include "EventFilter/StorageManager/interface/RegistrationQueue.h"
#include "EventFilter/StorageManager/interface/StatisticsReporter.h"
#include "EventFilter/StorageManager/interface/StreamQueue.h"

namespace stor {

  /**
   * Container for shared resources.
   *
   * $Author: dshpakov $
   * $Revision: 1.2 $
   * $Date: 2009/06/10 08:15:23 $
   */

  struct SharedResources
  {

    // queues
    boost::shared_ptr<CommandQueue> _commandQueue;
    boost::shared_ptr<DQMEventQueue> _dqmEventQueue;
    boost::shared_ptr<FragmentQueue> _fragmentQueue;
    boost::shared_ptr<StreamQueue> _streamQueue;
    boost::shared_ptr<RegistrationQueue> _registrationQueue;
    boost::shared_ptr<EventQueueCollection> _eventConsumerQueueCollection;
    boost::shared_ptr<DQMEventQueueCollection> _dqmEventConsumerQueueCollection;

    // other
    boost::shared_ptr<Configuration> _configuration;
    boost::shared_ptr<DiscardManager> _discardManager;
    boost::shared_ptr<DiskWriterResources> _diskWriterResources;
    boost::shared_ptr<DQMEventProcessorResources> _dqmEventProcessorResources;
    boost::shared_ptr<InitMsgCollection> _initMsgCollection;
    boost::shared_ptr<StatisticsReporter> _statisticsReporter;
    boost::shared_ptr<RegistrationCollection> _registrationCollection;

    /**
     * Add a Failed state-machine event to the command queue
     */
    void moveToFailedState( const std::string& reason );

  };

  typedef boost::shared_ptr<SharedResources> SharedResourcesPtr;
  
} // namespace stor

#endif // StorageManager_SharedResources_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
