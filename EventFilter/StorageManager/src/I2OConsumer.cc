/*
   Author: Harry Cheung, FNAL

   Description:
     EDM output module that will send the data via I2O frames to 
     a local I2O frame collector that will relay them to the
     Storage Manager.
     In this preproduction version, the I2O frames are directly
     sent to the Storage Manager. The destination is hardwired and
     provided through a global variable.
     See CMS EventFilter wiki page for further notes.

   Modification:
     version 1.1 2005/11/23
       Initial implementation, send directly to Storage Manager
       hardwired to be a xdaq application class testI2OReceiver.
       Uses a global variable instead of service set provided by
       the FU EventProcessor (eventually). 
       Needs changes for production version.
    version 1.2 2005/12/15
       Added passing of the run and event numbers to include in
       the I2O output frames. 
       Added HLT process identification information.
       Added a function to block on memory pool allocation if too full,
         only data I2O frames are blocked, registry and other are not.
         Code assumes the toolbox::mem::CommittedHeapAllocator is used!
       Added code to get performance measurements if used with SMi2oSender,
         should comment out if not using SMi2oSender!
    version 1.3 2006/01/24
       Remake header for leading fragments, and put in header in other
       fragments needed by the Storage Manager

*/

#include "EventFilter/StorageManager/interface/i2oStorageManagerMsg.h"
#include "EventFilter/StorageManager/src/I2OConsumer.h"
#include "IOPool/StreamerData/interface/Messages.h"
#include "EventFilter/StorageManager/interface/SMi2oSender.h"

#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "xdaq/Application.h"
#include "xdaq/ApplicationContext.h"
#include "xdaq/ApplicationGroup.h"

#include "toolbox/mem/MemoryPoolFactory.h"
#include "toolbox/mem/HeapAllocator.h"
#include "toolbox/mem/Reference.h"
#include "toolbox/mem/Pool.h"

// for performance measurements
#include "xdata/UnsignedLong.h"
#include "xdata/Double.h"

#include "xcept/tools.h"

#include "i2o/Method.h"
#include "i2o/utils/include/i2o/utils/AddressMap.h"

#include "xgi/Method.h"

#include <string>
#include <fstream>
#include <iostream>

#include <wait.h>

extern xdaq::Application* getMyXDAQPtr();
extern toolbox::mem::Pool *getMyXDAQPool();
extern xdaq::ApplicationDescriptor* getMyXDAQDest();

// for performance measurements
extern void addMyXDAQMeasurement(unsigned long size);
extern xdata::UnsignedLong getMyXDAQsamples();
extern xdata::Double getMyXDAQdatabw();
extern xdata::Double getMyXDAQdatarate();
extern xdata::Double getMyXDAQdatalatency();
extern xdata::UnsignedLong getMyXDAQtotalsamples();
extern xdata::Double getMyXDAQduration();
extern xdata::Double getMyXDAQmeandatabw();
extern xdata::Double getMyXDAQmeandatarate();
extern xdata::Double getMyXDAQmeandatalatency();
extern xdata::Double getMyXDAQmaxdatabw();
extern xdata::Double getMyXDAQmindatabw();

using namespace edm;
using namespace std;

namespace edmtest
{

  struct I2OWorker
  {
    I2OWorker(const string& s);

    string destinationName_;
    xdaq::Application* app_;
    toolbox::mem::Pool *pool_;
    xdaq::ApplicationDescriptor* destination_;
  };

  //I2OWorker::I2OWorker(const string& s = "testI2OReceiver" ):
  I2OWorker::I2OWorker(const string& s = "testStorageManager" ):
                      destinationName_(s),
                      app_(getMyXDAQPtr()),
                      pool_(getMyXDAQPool()),
                      destination_(getMyXDAQDest())
  {
    //LOG4CPLUS_INFO(app_->getApplicationLogger(),"Making I2OWorker");
    FDEBUG(10) << "I2OConsumer: Making I2OWorker" << std::endl;
  }

  // ----------------------------------

  I2OConsumer::I2OConsumer(edm::ParameterSet const& ps, 
			     edm::EventBuffer* buf):
    worker_(new I2OWorker(ps.getParameter<string>("DestinationName"))),
    bufs_(buf)
  {
    FDEBUG(10) << "I2OConsumer: Constructor" << std::endl;
  }
  
  I2OConsumer::~I2OConsumer()
  {
    delete worker_;
  }

  int I2OConsumer::i2oyield(unsigned int seconds)
  {
    // used to block (should yield to other threads)
    sleep(seconds);
    return 0;
  }
  
  void I2OConsumer::bufferReady()
  {
    EventBuffer::ConsumerBuffer cb(*bufs_);

    FDEBUG(10) << "I2OConsumer: write event to destination" << std::endl;
    int sz = cb.size();
    FDEBUG(10) << "I2OConsumer: event sz = " << sz << std::endl;
    // Should block if memory pool is too full until it has more room
    // (Is this the only thread posting frames to do with this process?)
    // should really check if the high threshold is actually set!
    if (!worker_->pool_->isHighThresholdExceeded())
    {
      edm::EventMsg msg(cb.buffer(),cb.size());
      edm::RunNumber_t runID     = msg.getRunNumber();
      edm::EventNumber_t eventID = msg.getEventNumber();
      // now we can post the data
      writeI2OData((const char*)cb.buffer(),sz,runID,eventID);
      // should test here is there was an error posting the data
    }
    else
    {
      LOG4CPLUS_INFO(worker_->app_->getApplicationLogger(),
        "I2OConsumer: High threshold exceeded in memory pool, blocking on I2O output");
      // instead of really blocking for a set amount of time
      // we yield to other threads until low threshold is reached
      // check the settings at the beginning
      std::cout << " max committed size "
                << worker_->pool_->getMemoryUsage().getCommitted() << std::endl;
      std::cout << " mem size used (bytes) " 
                << worker_->pool_->getMemoryUsage().getUsed() << std::endl;
      std::cout << " max possible mem size " 
                << worker_->pool_->getMemoryUsage().getMax() << std::endl;
      while (worker_->pool_->isLowThresholdExceeded())
      {
        LOG4CPLUS_INFO(worker_->app_->getApplicationLogger(),
          "I2OConsumer: Yield till low threshold reached");
        // Does this work? We need to receive e.g. the halt XOAP message!
        // this is probably not thread safe!
        this->i2oyield(1);
      }
      // get run and event numbers here from the buffer
      edm::EventMsg msg(cb.buffer(),cb.size());
      edm::RunNumber_t runID     = msg.getRunNumber();
      edm::EventNumber_t eventID = msg.getEventNumber();
      // now we can post the data
      writeI2OData((const char*)cb.buffer(),sz,runID,eventID);
    }
  }

  void I2OConsumer::stop()
  {
    EventBuffer::ProducerBuffer pb(*bufs_);
    pb.commit();
    FDEBUG(10) << "I2OConsumer: sending terminate run" << std::endl;
    int sz = 0;
    // The special "other" message is hardwired to send a
    // terminate run message (close file)
    writeI2OOther((const char*)pb.buffer(),sz);
  }

  void I2OConsumer::sendRegistry(void* buf, int len)
  {
    FDEBUG(10) << "I2OConsumer: sending registry" << std::endl;
    FDEBUG(10) << "I2OConsumer: registry len = " << len << std::endl;
    writeI2ORegistry((const char*)buf,len);    
  }

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  void I2OConsumer::writeI2OOther(const char* buffer, unsigned int size)
  {
    FDEBUG(10) << "I2OConsumer: write other message size = " << size << std::endl;
    size_t msgSizeInBytes = sizeof(I2O_SM_OTHER_MESSAGE_FRAME);

    try
    {
      toolbox::mem::Reference* bufRef =
         toolbox::mem::getMemoryPoolFactory()->getFrame(worker_->pool_,msgSizeInBytes);

      I2O_MESSAGE_FRAME *stdMsg =
        (I2O_MESSAGE_FRAME*)bufRef->getDataLocation();
      I2O_PRIVATE_MESSAGE_FRAME*pvtMsg = (I2O_PRIVATE_MESSAGE_FRAME*)stdMsg;
      I2O_SM_OTHER_MESSAGE_FRAME *msg =
        (I2O_SM_OTHER_MESSAGE_FRAME*)stdMsg;

      stdMsg->MessageSize      = msgSizeInBytes >> 2;
      try
      {
        stdMsg->InitiatorAddress =
              i2o::utils::getAddressMap()->getTid(worker_->app_->getApplicationDescriptor());
      }
      catch(xdaq::exception::ApplicationDescriptorNotFound e)
      {
        std::cerr << "I2OConsumer::WriteI2OOther:exception in getting source tid "
                  <<  xcept::stdformat_exception_history(e) << endl;
      }
      try
      {
        stdMsg->TargetAddress    =
                  i2o::utils::getAddressMap()->getTid(worker_->destination_);
      }
      catch(xdaq::exception::ApplicationDescriptorNotFound e)
      {
        std::cerr << "I2OConsumer::WriteI2OOther:exception in getting destination tid "
                  <<  xcept::stdformat_exception_history(e) << endl;
      }

      stdMsg->Function         = I2O_PRIVATE_MESSAGE;
      stdMsg->VersionOffset    = 0;
      stdMsg->MsgFlags         = 0;  // Point-to-point 

      pvtMsg->XFunctionCode    = I2O_SM_OTHER;
      pvtMsg->OrganizationID   = XDAQ_ORGANIZATION_ID;
      // Hardwired to a zero size message for now, meaning terminate run
      // Need to change this
      msg->otherData = size;
      // Fill in the long form of the source (HLT) identifier
      std::string url = worker_->app_->getApplicationDescriptor()->getContextDescriptor()->getURL();
      if(url.size() > MAX_I2O_SM_URLCHARS)
      {
        LOG4CPLUS_INFO(worker_->app_->getApplicationLogger(),
          "I2OConsumer: Error! Source URL truncated");
        for(int i=0; i< MAX_I2O_SM_URLCHARS; i++) msg->hltURL[i] = url[i];
      } else {
        //for(int i=0; i< MAX_I2O_SM_URLCHARS; i++) msg->hltURL[i] = " ";
        for(int i=0; i< (int)url.size(); i++) msg->hltURL[i] = url[i];
      } 
      std::string classname = worker_->app_->getApplicationDescriptor()->getClassName();
      if(classname.size() > MAX_I2O_SM_URLCHARS)
      {
        LOG4CPLUS_INFO(worker_->app_->getApplicationLogger(),
          "I2OConsumer: Error! Source ClassName truncated");
        for(int i=0; i< MAX_I2O_SM_URLCHARS; i++) msg->hltClassName[i] = classname[i];
      } else {
        //for(int i=0; i< MAX_I2O_SM_URLCHARS; i++) msg->hltClassName[i] = " ";
        for(int i=0; i< (int)url.size(); i++) msg->hltClassName[i] = classname[i];
      } 
      msg->hltLocalId = worker_->app_->getApplicationDescriptor()->getLocalId();
      msg->hltInstance = worker_->app_->getApplicationDescriptor()->getInstance();
      msg->hltTid = i2o::utils::getAddressMap()->getTid(worker_->app_->getApplicationDescriptor());

      bufRef->setDataSize(msgSizeInBytes);
      if(worker_->destination_ !=0)
        {
          worker_->app_->getApplicationContext()->postFrame(bufRef,
                         worker_->app_->getApplicationDescriptor(),worker_->destination_);
          // for performance measurements only using global variable!
          addMyXDAQMeasurement((unsigned long)msgSizeInBytes);
        }
      else
        std::cerr << "I2OConsumer:No " << worker_->destinationName_
                  << "destination in configuration" << std::endl;
    }
    catch(toolbox::mem::exception::Exception e)
    {
      std::cout << "I2OConsumer:exception in allocating frame "
           <<  xcept::stdformat_exception_history(e) << endl;
      return;
    }
    catch(...)
    { 
      std::cerr << "I2OConsumer:unknown exception in allocating frame " << endl;
      return;
    }
  }
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  void I2OConsumer::writeI2ORegistry(const char* buffer, unsigned int size)
  {
    FDEBUG(10) << "writeI2ORegistry: size = " << size << std::endl;
    // should really get rid of this
    std::string temp4print(buffer,size);
    FDEBUG(10) << "writeI2ORegistry data = " << temp4print << std::endl;
    size_t msgSizeInBytes = sizeof(I2O_SM_PREAMBLE_MESSAGE_FRAME);
    FDEBUG(10) << "msgSizeInBytes registry frame size = " << msgSizeInBytes << std::endl;
    FDEBUG(10) << "I2O_MESSAGE_FRAME size = " << sizeof(I2O_MESSAGE_FRAME) << std::endl;
    FDEBUG(10) << "I2O_PRIVATE_MESSAGE_FRAME size = " << sizeof(I2O_PRIVATE_MESSAGE_FRAME) << std::endl;

    try
    {
      FDEBUG(10) << "I2OConsumer: getting memory pool frame" << std::endl;
      toolbox::mem::Reference* bufRef =
         toolbox::mem::getMemoryPoolFactory()->getFrame(worker_->pool_,msgSizeInBytes);

      FDEBUG(10) << "I2OConsumer: setting up frame pointers" << std::endl;
      I2O_MESSAGE_FRAME *stdMsg =
        (I2O_MESSAGE_FRAME*)bufRef->getDataLocation();
      I2O_PRIVATE_MESSAGE_FRAME*pvtMsg = (I2O_PRIVATE_MESSAGE_FRAME*)stdMsg;
      I2O_SM_PREAMBLE_MESSAGE_FRAME *msg =
        (I2O_SM_PREAMBLE_MESSAGE_FRAME*)stdMsg;

      stdMsg->MessageSize      = msgSizeInBytes >> 2;
      try
      {
            stdMsg->InitiatorAddress =
                  i2o::utils::getAddressMap()->getTid(worker_->app_->getApplicationDescriptor());
      }
      catch(xdaq::exception::ApplicationDescriptorNotFound e)
      {
          std::cerr << "I2OConsumer::WriteI2ORegistry: exception in getting source tid "
               <<  xcept::stdformat_exception_history(e) << endl;
      }
      try
      {
        stdMsg->TargetAddress    =
                  i2o::utils::getAddressMap()->getTid(worker_->destination_);
      }
      catch(xdaq::exception::ApplicationDescriptorNotFound e)
      {
          std::cerr << "I2OConsumer::WriteI2ORegistry: exception in getting destination tid "
               <<  xcept::stdformat_exception_history(e) << endl;
      }

      stdMsg->Function         = I2O_PRIVATE_MESSAGE;
      stdMsg->VersionOffset    = 0;
      stdMsg->MsgFlags         = 0;  // normal message (not multicast)

      pvtMsg->XFunctionCode    = I2O_SM_PREAMBLE;
      pvtMsg->OrganizationID   = XDAQ_ORGANIZATION_ID;
      msg->dataSize = size;
      // Fill in the long form of the source (HLT) identifier
      std::string url = worker_->app_->getApplicationDescriptor()->getContextDescriptor()->getURL();
      if(url.size() > MAX_I2O_SM_URLCHARS)
      {
        LOG4CPLUS_INFO(worker_->app_->getApplicationLogger(),
          "I2OConsumer: Error! Source URL truncated");
        for(int i=0; i< MAX_I2O_SM_URLCHARS; i++) msg->hltURL[i] = url[i];
      } else {
        //for(int i=0; i< MAX_I2O_SM_URLCHARS; i++) msg->hltURL[i] = " ";
        for(int i=0; i< (int)url.size(); i++) msg->hltURL[i] = url[i];
      } 
      std::string classname = worker_->app_->getApplicationDescriptor()->getClassName();
      if(classname.size() > MAX_I2O_SM_URLCHARS)
      {
        LOG4CPLUS_INFO(worker_->app_->getApplicationLogger(),
          "I2OConsumer: Error! Source ClassName truncated");
        for(int i=0; i< MAX_I2O_SM_URLCHARS; i++) msg->hltClassName[i] = classname[i];
      } else {
        //for(int i=0; i< MAX_I2O_SM_URLCHARS; i++) msg->hltClassName[i] = " ";
        for(int i=0; i< (int)url.size(); i++) msg->hltClassName[i] = classname[i];
      } 
      msg->hltLocalId = worker_->app_->getApplicationDescriptor()->getLocalId();
      msg->hltInstance = worker_->app_->getApplicationDescriptor()->getInstance();
      msg->hltTid = i2o::utils::getAddressMap()->getTid(worker_->app_->getApplicationDescriptor());

      for (unsigned int i=0; i<size; i++){
        msg->data[i] = *(buffer+i);
      }
      // should really get rid of this
      std::string temp4print(msg->data,size);
      FDEBUG(10) << "I2OConsumer::WriteI2ORegistry: string msg_>data = " 
                 << temp4print << std::endl;

      bufRef->setDataSize(msgSizeInBytes);

      FDEBUG(10) << "I2OConsumer::WriteI2ORegistry: checking if destination exist" << std::endl;
      if(worker_->destination_ !=0)
        {
          FDEBUG(10) << "I2OConsumer::WriteI2ORegistry: posting registry frame " << std::endl;
          worker_->app_->getApplicationContext()->postFrame(bufRef,
                         worker_->app_->getApplicationDescriptor(),worker_->destination_);
          // for performance measurements only using global variable!
          addMyXDAQMeasurement((unsigned long)msgSizeInBytes);
        }
      else
        std::cerr << "I2OConsumer::WriteI2ORegistry: No " << worker_->destinationName_
                  << "destination in configuration" << std::endl;
    }
    catch(toolbox::mem::exception::Exception e)
    {
      std::cout << "I2OConsumer::WriteI2ORegistry::exception in allocating frame "
           <<  xcept::stdformat_exception_history(e) << endl;
      return;
    }
    catch(...)
    { 
      std::cout << "I2OConsumer::WriteI2ORegistry: unknown exception in allocating frame " << endl;
      return;
    }
  }
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  void I2OConsumer::writeI2OData(const char* buffer, unsigned int size,
                                 edm::RunNumber_t runid, edm::EventNumber_t eventid)
  //
  // function to write the data buffer in I2O frames. If more than one
  // frame is needed a chain is created and then posted.
  {
    // should really test the size >0
    FDEBUG(10) << "I2OConsumer::writeI2OData: data size (in bytes) = " << size << std::endl;
    // must decide how many frames we need to use to send this message
    unsigned int maxSizeInBytes = MAX_I2O_SM_DATASIZE;
    unsigned int headerNeededSize = sizeof(MsgCode::Codes)+sizeof(EventMsg::Header);
    //std::cout << "headerNeededSize = " << headerNeededSize << std::endl;
    unsigned int maxEvMsgDataFragSizeInBytes = maxSizeInBytes - headerNeededSize;
//    unsigned int numFramesNeeded = size/maxSizeInBytes;
//    unsigned int remainder = size%maxSizeInBytes;
    int size4data = size - headerNeededSize;
    unsigned int numFramesNeeded = size4data/maxEvMsgDataFragSizeInBytes;
    unsigned int remainder = size4data%maxEvMsgDataFragSizeInBytes;

    if (remainder > 0) numFramesNeeded++;
    FDEBUG(10) << "I2OConsumer::writeI2OData: number of frames needed = " << numFramesNeeded 
               << " remainder = " << remainder << std::endl;
    // We need to set up a chain of frames and send this
    // chains are sent together once we post the head frame
    int start = 0;
    int thisCount = 0;
    unsigned int thisSize = 0;
    toolbox::mem::Reference *head = 0;
    toolbox::mem::Reference *tail = 0;

    for (int i=0; i<(int)numFramesNeeded; i++)
    {
      thisCount = i;
      if (size != 0)  // should not be writing anything for size = 0!
      {
        //start = i*maxSizeInBytes;
        start = i*maxEvMsgDataFragSizeInBytes;
        if (i < ((int)numFramesNeeded)-1 || remainder == 0)
          //thisSize = maxSizeInBytes;
          thisSize = maxEvMsgDataFragSizeInBytes;
        else
          thisSize = remainder;
      }
      // should get rid of this later - just used to create a dump for checking
      int minlen = 50;
      if(minlen > (int)thisSize) minlen = thisSize;
      std::string temp4print(buffer+start,minlen);
      FDEBUG(10) << "I2OConsumer::writeI2OData: data = " << temp4print << std::endl;

      size_t msgSizeInBytes = sizeof(I2O_SM_DATA_MESSAGE_FRAME);
      FDEBUG(10) << "I2OConsumer::writeI2OData: msgSizeInBytes data frame size = " 
                 << msgSizeInBytes << std::endl;
      if(thisSize > MAX_I2O_SM_DATASIZE)
      {
        // this should never happen - get rid of this later?
        std::cerr << "I2OConsumer::writeI2OData: unexpected error! "
                  << "Data larger than one frame abort " << std::endl;
        return;
      }
      try
      {
        toolbox::mem::Reference* bufRef =
           toolbox::mem::getMemoryPoolFactory()->getFrame(worker_->pool_,msgSizeInBytes);

        I2O_MESSAGE_FRAME *stdMsg =
          (I2O_MESSAGE_FRAME*)bufRef->getDataLocation();
        I2O_PRIVATE_MESSAGE_FRAME*pvtMsg = (I2O_PRIVATE_MESSAGE_FRAME*)stdMsg;
        I2O_SM_DATA_MESSAGE_FRAME *msg =
          (I2O_SM_DATA_MESSAGE_FRAME*)stdMsg;

        stdMsg->MessageSize      = msgSizeInBytes >> 2;
        try
        {
          stdMsg->InitiatorAddress =
            i2o::utils::getAddressMap()->getTid(worker_->app_->getApplicationDescriptor());
        }
        catch(xdaq::exception::ApplicationDescriptorNotFound e)
        {
          std::cerr << "I2OConsumer::writeI2OData: exception in getting source tid "
                    <<  xcept::stdformat_exception_history(e) << endl;
        }
        try
        {
          stdMsg->TargetAddress    =
                    i2o::utils::getAddressMap()->getTid(worker_->destination_);
        }
        catch(xdaq::exception::ApplicationDescriptorNotFound e)
        {
          std::cerr << "I2OConsumer::writeI2OData: exception in getting destination tid "
                    <<  xcept::stdformat_exception_history(e) << endl;
        }

        stdMsg->Function         = I2O_PRIVATE_MESSAGE;
        stdMsg->VersionOffset    = 0;
        stdMsg->MsgFlags         = 0;  // normal message (not multicast)

        pvtMsg->XFunctionCode    = I2O_SM_DATA;
        pvtMsg->OrganizationID   = XDAQ_ORGANIZATION_ID;
        //msg->dataSize = thisSize;
        msg->dataSize = thisSize+headerNeededSize;
        // Fill in the long form of the source (HLT) identifier
        std::string url = worker_->app_->getApplicationDescriptor()->getContextDescriptor()->getURL();
        if(url.size() > MAX_I2O_SM_URLCHARS)
        {
          LOG4CPLUS_INFO(worker_->app_->getApplicationLogger(),
            "I2OConsumer: Error! Source URL truncated");
          for(int i=0; i< MAX_I2O_SM_URLCHARS; i++) msg->hltURL[i] = url[i];
        } else {
          //for(int i=0; i< MAX_I2O_SM_URLCHARS; i++) msg->hltURL[i] = " "; // needed?
          for(int i=0; i< (int)url.size(); i++) msg->hltURL[i] = url[i];
        } 
        std::string classname = worker_->app_->getApplicationDescriptor()->getClassName();
        if(classname.size() > MAX_I2O_SM_URLCHARS)
        {
          LOG4CPLUS_INFO(worker_->app_->getApplicationLogger(),
            "I2OConsumer: Error! Source ClassName truncated");
          for(int i=0; i< MAX_I2O_SM_URLCHARS; i++) msg->hltClassName[i] = classname[i];
        } else {
          //for(int i=0; i< MAX_I2O_SM_URLCHARS; i++) msg->hltClassName[i] = " "; // needed?
          for(int i=0; i< (int)url.size(); i++) msg->hltClassName[i] = classname[i];
        } 
        msg->hltLocalId = worker_->app_->getApplicationDescriptor()->getLocalId();
        msg->hltInstance = worker_->app_->getApplicationDescriptor()->getInstance();
        msg->hltTid = i2o::utils::getAddressMap()->getTid(worker_->app_->getApplicationDescriptor());

        msg->runID = (unsigned long)runid;     // convert to known size
        msg->eventID = (unsigned long)eventid; // convert to known size
        msg->numFrames = numFramesNeeded;
        msg->frameCount = thisCount;
        msg->originalSize = size;
        // make the chain
        if(thisCount == 0)  // This is the first frame
        {
          head = bufRef;
          tail = bufRef;
        }
        else
        {
          tail->setNextReference(bufRef); // set nextref in last frame to be this one
          tail = bufRef;
        }
        if(thisSize != 0)
        {
          for (unsigned int i=0; i<thisSize; i++)
          {
            msg->data[i+headerNeededSize] = *(buffer+headerNeededSize+i + start);
          }
          // remake header
          char dummyBuffer[MAX_I2O_SM_DATASIZE];
          EventMsg msgFrag(dummyBuffer, thisSize+headerNeededSize, 
            eventid, runid, thisCount+1, numFramesNeeded);
          for (unsigned int i=0; i<headerNeededSize; i++)
          {
            msg->data[i] = dummyBuffer[i];
          }

          // should get rid of this later - just used to create a dump for checking
          minlen = 50;
          if(minlen > (int)thisSize) minlen = thisSize;
          std::string temp4print(msg->data,minlen);
          FDEBUG(10) << "I2OConsumer::writeI2OData: msg data = " << temp4print << std::endl;
        } else {
          std::cout << "I2OConsumer::writeI2OData: Error! Sending zero size data!?" << std::endl;
        }

// need to set the actual buffer size that is being sent
        bufRef->setDataSize(msgSizeInBytes);

      }
      catch(toolbox::mem::exception::Exception e)
      {
        std::cerr << "I2OConsumer::writeI2OData: exception in allocating frame "
                  <<  xcept::stdformat_exception_history(e) << endl;
        return;  // is this the right action?
      }
      catch(...)
      { 
        std::cerr << "I2OConsumer::writeI2OData: unknown exception in allocating frame" << endl;
        return;  // is this the right action?
      } //end try for frame allocation
    } //end loop over frame

    // don't postFrame until all frames in chain are set up and there was no error!
    // need to make a test here later
    if(worker_->destination_ !=0)
      {
        FDEBUG(10) << "I2OConsumer::writeI2OData: posting data chain frame " << std::endl;
        worker_->app_->getApplicationContext()->postFrame(head,
                       worker_->app_->getApplicationDescriptor(),worker_->destination_);
        // for performance measurements only using global variable!
        addMyXDAQMeasurement((unsigned long)(numFramesNeeded * sizeof(I2O_SM_DATA_MESSAGE_FRAME)));
      }
    else
      std::cerr << "I2OConsumer::writeI2OData: No " << worker_->destinationName_
                << "destination in configuration" << std::endl;
    // Do not need to release buffers in the sender as this is done in the transport
    // layer. See tcp::PeerTransportSender::post and tcp::PeerTransportSender::svc
    // in TriDAS_v3.3/daq/pt/tcp/src/common/PeerTransportSender.cc
    //
    // What if there was an error in receiving though? 
    // Shouldn't I want a handshake to release the event data?
  }

}
