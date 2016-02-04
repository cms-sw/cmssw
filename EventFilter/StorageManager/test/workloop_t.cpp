// $Id: workloop_t.cpp,v 1.2 2009/06/10 08:15:32 dshpakov Exp $

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
    _fragmentCollectorWorkloop = 
      toolbox::task::getWorkLoopFactory()->getWorkLoop("FragmentCollectorWorkloop", "polling");

    _dqmProcessorWorkloop = 
      toolbox::task::getWorkLoopFactory()->getWorkLoop("DqmProcessorWorkloop", "polling");


    // Define actions    
    toolbox::task::ActionSignature* processFragmentQueueAction = 
      toolbox::task::bind(this, &WorkloopTest::processFragmentQueue, "ProcessFragmentQueue");
    
    toolbox::task::ActionSignature* processCommandQueueAction = 
      toolbox::task::bind(this, &WorkloopTest::processCommandQueue, "ProcessCommandQueue");
    
    toolbox::task::ActionSignature* processDQMEventQueueAction = 
      toolbox::task::bind(this, &WorkloopTest::processDQMEventQueue, "ProcessDQMEventQueue");


    // Add actions to workloops
    _fragmentCollectorWorkloop->submit(processFragmentQueueAction);
    _fragmentCollectorWorkloop->submit(processCommandQueueAction);

    _dqmProcessorWorkloop->submit(processDQMEventQueueAction);
    
		
    // Activate the workloops
    // Note: activating an active workloop throws toolbox::task::exception::Exception
    if ( ! _fragmentCollectorWorkloop->isActive() )
    {
      _fragmentCollectorWorkloop->activate();
    }
    if ( ! _dqmProcessorWorkloop->isActive() )
    {
      _dqmProcessorWorkloop->activate();
    }
  }


  ~WorkloopTest()
  {
    _fragmentCollectorWorkloop->cancel();
    _dqmProcessorWorkloop->cancel();

    // Do we need to remove the workloop from the factory, too?
    // This interface is awkward.
    toolbox::net::URN urn1("toolbox-task-workloop", _fragmentCollectorWorkloop->getName());
    toolbox::task::getWorkLoopFactory()->removeWorkLoop(urn1);

    toolbox::net::URN urn2("toolbox-task-workloop", _dqmProcessorWorkloop->getName());
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

	
  toolbox::task::WorkLoop* _fragmentCollectorWorkloop;
  toolbox::task::WorkLoop* _dqmProcessorWorkloop;

	
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
