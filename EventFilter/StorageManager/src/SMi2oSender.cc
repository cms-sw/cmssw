/*
   Author: Harry Cheung, FNAL

   Description:
     XDAQ application that is meant to receive locally I2O frames
     from the FU HLT application, and sends them via the network
     to the Storage Manager.
     See CMS EventFilter wiki page for further notes.

   Modification:
     version 1.1 2005/11/23
       Initial implementation, only creates pool and destination.
       It does not relay I2O frames. This is done directly in the
       I2O Output module in this preproduction version.
       Uses a global pointer. Needs changes for production version.

*/

#include "EventFilter/StorageManager/interface/SMi2oSender.h"
#include "EventFilter/StorageManager/interface/i2oStorageManagerMsg.h"

#include "xdaq/Application.h"
#include "xdaq/ApplicationContext.h"
#include "xdaq/ApplicationGroup.h"

#include "toolbox/mem/HeapAllocator.h"
#include "toolbox/mem/Reference.h"

#include "xcept/tools.h"

#include "i2o/Method.h"
#include "i2o/utils/include/i2o/utils/AddressMap.h"

#include <exception>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

struct SMFU_data
{
  SMFU_data();

  xdaq::Application* app;
  toolbox::mem::Pool *pool;
  xdaq::ApplicationDescriptor* destination;
};

SMFU_data::SMFU_data()
{
  app = 0;
  pool = 0;
  destination = 0;
}

// define a global variable and a global function to return pointer
// until we change the EventFilter/Processor/FUEventProcessor and 
// EventFilter/Processor/EventProcessor to provide these via the
// service set.

static SMFU_data SMfudata;

xdaq::Application* getMyXDAQPtr()
{
  return SMfudata.app;
}

toolbox::mem::Pool *getMyXDAQPool()
{
  return SMfudata.pool;
}

xdaq::ApplicationDescriptor* getMyXDAQDest()
{
  return SMfudata.destination;
}

//--------------------------------------------------------------
SMi2oSender::SMi2oSender(xdaq::ApplicationStub * s)
  throw (xdaq::exception::Exception): xdaq::Application(s)
{
  // create memory pool and find destinations
  // to be used for I2O frame relaying
  // currently also used by I2OConsumer output module to avoid
  // changes to EventFilter/Processor/FUEventProcessor, EventProcessor

  LOG4CPLUS_INFO(this->getApplicationLogger(),"Making SMi2oSender");
  // Create a memory pool

  std::string poolName = "SMi2oPool";
  try
  {
    toolbox::mem::HeapAllocator *allocator =
                  new toolbox::mem::HeapAllocator();
    toolbox::net::URN urn("toolbox-mem-pool", poolName);
    toolbox::mem::MemoryPoolFactory *poolFactory =
                  toolbox::mem::getMemoryPoolFactory();

    // Change to CommittedHeapAllocator to limit the pool size
    pool_ = poolFactory->createPool(urn, allocator);

    LOG4CPLUS_INFO(getApplicationLogger(),
                   "Created memory pool: " << poolName);
  }
  catch (toolbox::mem::exception::Exception& e)
  {
    string s = "Failed to create pool: " + poolName;

    LOG4CPLUS_FATAL(getApplicationLogger(), s);
    XCEPT_RETHROW(xcept::Exception, s, e);
  }

  // Get XDAQ application destinations - currently hardwired!
  try{
    destinations_=
    getApplicationContext()->getApplicationGroup()->
    getApplicationDescriptors("testI2OReceiver"); // hardwire here for now
  }
  catch(xdaq::exception::ApplicationDescriptorNotFound e)
  {
    LOG4CPLUS_ERROR(this->getApplicationLogger(),
                    "No testI2OReceiver available in configuration");
  }
  catch(...)
  {
      LOG4CPLUS_FATAL(this->getApplicationLogger(),
                      "Unknown error in looking up connectable testI2OReceiver");
  }
  // set global variable to share with the i2o output module
  SMfudata.app = this;
  SMfudata.pool = pool_;
  if(destinations_.size()>0)
  {
    firstDestination_ = destinations_[0];
    SMfudata.destination = destinations_[0];
  }
  else
  {
    LOG4CPLUS_ERROR(this->getApplicationLogger(),
                    "SMi2oSender::No receiver in configuration");
  }
}

/**
 * Provides factory method for the instantiation of FU applications
 */

extern "C" xdaq::Application * instantiate_SMi2oSender(xdaq::ApplicationStub * stub )
{
        std::cout << "Going to construct a SMi2oSender instance " << endl;
        return new SMi2oSender(stub);
}
