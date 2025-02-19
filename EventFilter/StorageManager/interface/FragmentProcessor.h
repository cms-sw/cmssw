// $Id: FragmentProcessor.h,v 1.6 2011/03/07 15:31:32 mommsen Exp $
/// @file: FragmentProcessor.h 

#ifndef EventFilter_StorageManager_FragmentProcessor_h
#define EventFilter_StorageManager_FragmentProcessor_h

#include "toolbox/lang/Class.h"
#include "toolbox/task/WaitingWorkLoop.h"
#include "xdaq/Application.h"

#include "boost/date_time/posix_time/posix_time_types.hpp"
#include "boost/shared_ptr.hpp"

#include "EventFilter/StorageManager/interface/EventDistributor.h"
#include "EventFilter/StorageManager/interface/FragmentQueue.h"
#include "EventFilter/StorageManager/interface/FragmentStore.h"
#include "EventFilter/StorageManager/interface/SharedResources.h"
#include "EventFilter/StorageManager/interface/StateMachine.h"
#include "EventFilter/StorageManager/interface/WrapperNotifier.h"


namespace stor {

  class I2OChain;
  class QueueID;


  /**
   * Processes I2O event fragments
   *
   * It pops the next fragment from the FragmentQueue and adds it to the
   * FragmentStore. If this completes the event, it hands it to the 
   * EventDistributor.
   *
   * $Author: mommsen $
   * $Revision: 1.6 $
   * $Date: 2011/03/07 15:31:32 $
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

    xdaq::Application*                 app_;
    SharedResourcesPtr                 sharedResources_;
    WrapperNotifier                    wrapperNotifier_;
    StateMachinePtr                    stateMachine_;
    FragmentStore                      fragmentStore_;
    EventDistributor                   eventDistributor_;

    boost::posix_time::time_duration   timeout_; // Waiting time
    bool                               actionIsActive_;

    toolbox::task::WorkLoop*           processWL_;      

  };
  
} // namespace stor

#endif // EventFilter_StorageManager_FragmentProcessor_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
