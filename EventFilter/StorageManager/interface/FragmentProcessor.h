// $Id: FragmentProcessor.h,v 1.2 2009/06/10 08:15:23 dshpakov Exp $
/// @file: FragmentProcessor.h 

#ifndef StorageManager_FragmentProcessor_h
#define StorageManager_FragmentProcessor_h

#include "toolbox/lang/Class.h"
#include "toolbox/task/WaitingWorkLoop.h"
#include "xdaq/Application.h"

#include "boost/shared_ptr.hpp"

#include "EventFilter/StorageManager/interface/EventDistributor.h"
#include "EventFilter/StorageManager/interface/FragmentQueue.h"
#include "EventFilter/StorageManager/interface/FragmentStore.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/StorageManager/interface/QueueID.h"
#include "EventFilter/StorageManager/interface/SharedResources.h"
#include "EventFilter/StorageManager/interface/StateMachine.h"
#include "EventFilter/StorageManager/interface/WrapperNotifier.h"

namespace stor {

  /**
   * Processes I2O event fragments
   *
   * It pops the next fragment from the FragmentQueue and adds it to the
   * FragmentStore. If this completes the event, it hands it to the 
   * EventDistributor.
   *
   * $Author: dshpakov $
   * $Revision: 1.2 $
   * $Date: 2009/06/10 08:15:23 $
   */

  class FragmentProcessor : public toolbox::lang::Class
  {
  public:
    
    FragmentProcessor( xdaq::Application *app, SharedResourcesPtr sr );

    ~FragmentProcessor();
    
    /**
     * The workloop action processing state machine commands from the
     * command queue and handling I2O messages retrieved from the
     * FragmentQueue
     */
    bool processMessages(toolbox::task::WorkLoop*);

    /**
     * Create and start the fragment processing workloop
     */
    void startWorkLoop(std::string workloopName);


  private:

    /**
     * Processes all state machine events in the command queue
     */
    void processAllCommands();

    /**
     * Processes all consumer registrations in the registration queue
     */
    void processAllRegistrations();

    /**
       Process a single fragment, if there is  place to put it.
     */
    void processOneFragmentIfPossible();

    /**
       Process a single fragment. This should only be called if it has
       already been determined there is a place to put it.
     */
    void processOneFragment();

    xdaq::Application*                 _app;
    SharedResourcesPtr                 _sharedResources;
    WrapperNotifier                    _wrapperNotifier;
    boost::shared_ptr<StateMachine>    _stateMachine;
    FragmentStore                      _fragmentStore;
    EventDistributor                   _eventDistributor;

    unsigned int                       _timeout; // Waiting time in seconds.
    bool                               _actionIsActive;

    toolbox::task::WorkLoop*           _processWL;      

  };
  
} // namespace stor

#endif // StorageManager_FragmentProcessor_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
