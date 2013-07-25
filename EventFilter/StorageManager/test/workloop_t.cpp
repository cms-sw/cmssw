// $Id: workloop_t.cpp,v 1.3 2011/03/07 15:31:32 mommsen Exp $

// Test script to demonstrate the use of xdaq workloops
// Documentation at https://twiki.cern.ch/twiki/bin/view/XdaqWiki/WebHome?topic=workloop

#include "toolbox/task/WorkLoopFactory.h"
#include "toolbox/task/WaitingWorkLoop.h"
#include "toolbox/task/Action.h"
#include "toolbox/lang/Class.h"
#include "toolbox/net/URN.h"

#include <iostream>
#include <unistd.h>

class WorkloopTest : public toolbox::lang::Class
{
public:
  
  WorkloopTest()
  {
    // Get 2 work loops
    fragmentCollectorWorkloop_ = 
      toolbox::task::getWorkLoopFactory()->getWorkLoop("FragmentCollectorWorkloop", "polling");

    dqmProcessorWorkloop_ = 
      toolbox::task::getWorkLoopFactory()->getWorkLoop("DqmProcessorWorkloop", "polling");


    // Define actions    
    toolbox::task::ActionSignature* processFragmentQueueAction = 
      toolbox::task::bind(this, &WorkloopTest::processFragmentQueue, "ProcessFragmentQueue");
    
    toolbox::task::ActionSignature* processCommandQueueAction = 
      toolbox::task::bind(this, &WorkloopTest::processCommandQueue, "ProcessCommandQueue");
    
    toolbox::task::ActionSignature* processDQMEventQueueAction = 
      toolbox::task::bind(this, &WorkloopTest::processDQMEventQueue, "ProcessDQMEventQueue");


    // Add actions to workloops
    fragmentCollectorWorkloop_->submit(processFragmentQueueAction);
    fragmentCollectorWorkloop_->submit(processCommandQueueAction);

    dqmProcessorWorkloop_->submit(processDQMEventQueueAction);
    
		
    // Activate the workloops
    // Note: activating an active workloop throws toolbox::task::exception::Exception
    if ( ! fragmentCollectorWorkloop_->isActive() )
    {
      fragmentCollectorWorkloop_->activate();
    }
    if ( ! dqmProcessorWorkloop_->isActive() )
    {
      dqmProcessorWorkloop_->activate();
    }
  }


  ~WorkloopTest()
  {
    fragmentCollectorWorkloop_->cancel();
    dqmProcessorWorkloop_->cancel();

    // Do we need to remove the workloop from the factory, too?
    // This interface is awkward.
    toolbox::net::URN urn1("toolbox-task-workloop", fragmentCollectorWorkloop_->getName());
    toolbox::task::getWorkLoopFactory()->removeWorkLoop(urn1);

    toolbox::net::URN urn2("toolbox-task-workloop", dqmProcessorWorkloop_->getName());
    toolbox::task::getWorkLoopFactory()->removeWorkLoop(urn2);
  }
	

private:
  
  bool processFragmentQueue(toolbox::task::WorkLoop* wl)
  {
    std::cout << "Processing a I2O fragment" << std::endl;
    ::sleep(1);
    return true; // go on
  }

  
  bool processCommandQueue(toolbox::task::WorkLoop* wl)
  {
    std::cout << "Processing a state machine command" << std::endl;
    ::sleep(5);
    return true; // go on
  }

  
  bool processDQMEventQueue(toolbox::task::WorkLoop* wl)
  {
    std::cout << "Processing a DQM event" << std::endl;
    ::sleep(1);
    return true; // go on
  }

	
  toolbox::task::WorkLoop* fragmentCollectorWorkloop_;
  toolbox::task::WorkLoop* dqmProcessorWorkloop_;

	
};

int main ()
{
  WorkloopTest *t = new WorkloopTest();
  ::sleep(30);
  delete t;
  return 0;
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
