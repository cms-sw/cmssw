/*
   Description:
     EDM output module that will send the data via I2O frames to 
     to the Storage Manager. In this version the 
     destination is hardwired and provided through a global variable.
     See the CMS EvF Storage Manager wiki page for further notes.

   $Id$
*/

// why do I need this?
//#include "IOPool/Streamer/interface/EventStreamOutput.h"

#include "EventFilter/StorageManager/interface/i2oStorageManagerMsg.h"
#include "EventFilter/StorageManager/src/StreamerI2OWriter.h"
//-#include "IOPool/Streamer/interface/Messages.h"
#include "DataFormats/Common/interface/EventID.h"
#include "EventFilter/StorageManager/interface/SMi2oSender.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "xdaq/Application.h"
#include "xdaq/ApplicationContext.h"
#include "xdaq/ApplicationGroup.h"

#include "xcept/include/xcept/Exception.h"
#include "toolbox/include/toolbox/fsm/exception/Exception.h"

#include "toolbox/mem/MemoryPoolFactory.h"
#include "toolbox/mem/HeapAllocator.h"
#include "toolbox/mem/Reference.h"
#include "toolbox/mem/Pool.h"

// for performance measurements
#include "xdata/UnsignedInteger32.h"
#include "xdata/Double.h"

#include "xcept/tools.h"

#include "i2o/Method.h"
#include "i2o/utils/include/i2o/utils/AddressMap.h"

#include "xgi/Method.h"

#include <string>
#include <fstream>
#include <iostream>

//#include <wait.h>

extern xdaq::Application* getMyXDAQPtr();
extern toolbox::mem::Pool *getMyXDAQPool();
extern xdaq::ApplicationDescriptor* getMyXDAQDest();
extern xdaq::ApplicationDescriptor* getMyXDAQDest(unsigned int);

// for performance measurements
extern void addMyXDAQMeasurement(unsigned int size);
extern xdata::UnsignedInteger32 getMyXDAQsamples();
extern xdata::Double getMyXDAQdatabw();
extern xdata::Double getMyXDAQdatarate();
extern xdata::Double getMyXDAQdatalatency();
extern xdata::UnsignedInteger32 getMyXDAQtotalsamples();
extern xdata::Double getMyXDAQduration();
extern xdata::Double getMyXDAQmeandatabw();
extern xdata::Double getMyXDAQmeandatarate();
extern xdata::Double getMyXDAQmeandatalatency();
extern xdata::Double getMyXDAQmaxdatabw();
extern xdata::Double getMyXDAQmindatabw();

using namespace edm;
using namespace std;

namespace edm
{

  struct I2OStreamWorker
  {
    I2OStreamWorker(const string& s, const int dsi);

    string destinationName_;
    xdaq::Application* app_;
    toolbox::mem::Pool *pool_;
    xdaq::ApplicationDescriptor* destination_;
  };

  I2OStreamWorker::I2OStreamWorker(const string& s = "testStorageManager", const int dsi = -1):
                      destinationName_(s),
                      app_(getMyXDAQPtr()),
                      pool_(getMyXDAQPool())
  {
    FDEBUG(9) << "I2OStreamWorker: destination name = " << destinationName_;
    FDEBUG(9) << "I2OStreamWorker: I2OStreamWorker application = " << app_;
    if(dsi<0)
      destination_ = getMyXDAQDest();
    else
      destination_ = getMyXDAQDest(dsi);
    FDEBUG(9) << "I2OStreamWorker: app descriptor = " << destination_->getURN();
    if(destination_ == 0)
    {
      ostringstream os;
      os << "Could not find instance " << dsi << " for destination " << s;
      XCEPT_RAISE (toolbox::fsm::exception::Exception, 
		     os.str().c_str());
    }
    FDEBUG(9) << "StreamerI2OWriter: Making I2OStreamWorker" << std::endl;
  }

  // ----------------------------------

  StreamerI2OWriter::StreamerI2OWriter(edm::ParameterSet const& ps):
    worker_(new I2OStreamWorker(ps.getParameter<string>("DestinationName"),
				ps.getUntrackedParameter<int>("smInstance",-1))),
    i2o_max_size_(ps.getUntrackedParameter<int>("i2o_max_size",I2O_MAX_SIZE))
  {
    FDEBUG(10) << "StreamerI2OWriter: Constructor" << std::endl;
    // check the max i20 frame size is not above the value that causes a crash!
    if(i2o_max_size_ > I2O_ABSOLUTE_MAX_SIZE) {
      int old_i2o_max_size = i2o_max_size_;
      i2o_max_size_ = I2O_ABSOLUTE_MAX_SIZE;
      edm::LogWarning("StreamerI2OWriter") <<
        "user defined i2o_max_size too large for xdaq tcp, changed from " 
                     << old_i2o_max_size << " to " << i2o_max_size_;
    }
    // check the total i20 frame size is a multiple of 64 bits (8 bytes)
    if((i2o_max_size_ & 0x7) != 0) {
      int old_i2o_max_size = i2o_max_size_;
      // round it DOWN as this is the maximum size (keep the 0 for illustration!)
      i2o_max_size_ = ((i2o_max_size_ >> 3) + 0) << 3;
      edm::LogWarning("StreamerI2OWriter") <<
        "user defined i2o_max_size not multiple of 64 bits, changed from " 
                     << old_i2o_max_size << " to " << i2o_max_size_;
    }
    // get the actual max data sizes for the each of the types of frames
    max_i2o_sm_datasize_ =  i2o_max_size_ - sizeof(I2O_SM_DATA_MESSAGE_FRAME) ;
    max_i2o_registry_datasize_ = i2o_max_size_ - sizeof(I2O_SM_PREAMBLE_MESSAGE_FRAME); 
    max_i2o_other_datasize_ = i2o_max_size_ - sizeof(I2O_SM_OTHER_MESSAGE_FRAME); 
    max_i2o_DQM_datasize_ = i2o_max_size_ - sizeof(I2O_SM_DQM_MESSAGE_FRAME); 

    FDEBUG(9) <<"StreamerI2OWriter: Ctor i2o_max_size_: "<< i2o_max_size_ << std::endl;
    FDEBUG(9) <<"StreamerI2OWriter: Ctor max_i2o_sm_datasize_: "<< max_i2o_sm_datasize_ << std::endl;
    FDEBUG(9) <<"StreamerI2OWriter: Ctor max_i2o_registry_datasize_: "<< max_i2o_registry_datasize_ << std::endl;
    FDEBUG(9) <<"StreamerI2OWriter: Ctor max_i2o_other_datasize_: "<< max_i2o_other_datasize_ << std::endl;
    FDEBUG(9) <<"StreamerI2OWriter: Ctor max_i2o_DQM_datasize_: "<< max_i2o_DQM_datasize_ << std::endl;
    FDEBUG(9) << "StreamerI2OWriter: constructor" << endl;
  }
  
  StreamerI2OWriter::~StreamerI2OWriter()
  {
    FDEBUG(9) << "StreamerI2OWriter: StreamerI2OWriter destructor" << endl;
    delete worker_;
  }

  void StreamerI2OWriter::doOutputHeader(InitMsgBuilder const& initMessage)
  {
    writeI2ORegistry(initMessage);
    // what happens to memory when initMessage was passed as an auto_ptr?
  }

  void StreamerI2OWriter::doOutputEvent(EventMsgBuilder const& eventMessage)
  {
    // First test if memory pool can hold another event
    FDEBUG(10) << "StreamerI2OWriter: write event to destination" << std::endl;
    int sz = eventMessage.size();
    FDEBUG(10) << "StreamerI2OWriter: event sz = " << sz << std::endl;

    writeI2OData(eventMessage);
  }

  void StreamerI2OWriter::stop()
  {
    FDEBUG(9) << "StreamerI2OWriter: sending terminate run" << std::endl;
    edm::LogInfo("StreamerI2OWriter") << "stop called";
    // make a DONE message
    char* dummyBuffer = new char[16];
    OtherMessageBuilder othermsg(dummyBuffer,Header::DONE);
    writeI2OOther(othermsg);
    delete [] dummyBuffer;
  }

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  void StreamerI2OWriter::writeI2ORegistry(InitMsgBuilder const& initMessage)
  {
    char* buffer = (char*) initMessage.startAddress();
    unsigned int size = initMessage.size();

    // Use the common chain creation routine
    toolbox::mem::Reference *head = 0;
    unsigned int numBytesSent = 0;
    // need to catch rethrows? Or leave it to framework?
    head = SMi2oSender::createI2OFragmentChain(buffer, size,
                                               worker_->pool_,
                                               max_i2o_registry_datasize_,
                                               sizeof(I2O_SM_PREAMBLE_MESSAGE_FRAME),
                                               (unsigned short)I2O_SM_PREAMBLE,
                                               worker_->app_,
                                               worker_->destination_,
                                               numBytesSent);

    if(head != NULL) {
      FDEBUG(10) << "StreamerI2OWriter::WriteI2ORegistry: checking if destination exist" << std::endl;
      if(worker_->destination_ !=0)
      {
        FDEBUG(10) << "StreamerI2OWriter::WriteI2ORegistry: posting registry frame " << std::endl;
        try{
          worker_->app_->getApplicationContext()->postFrame(head,
                   worker_->app_->getApplicationDescriptor(),worker_->destination_);
        }
        catch(xcept::Exception &e)
        {
          edm::LogError("writeI2ORegistry") << "Exception writeI2ORegistry postFrame" 
                        << xcept::stdformat_exception_history(e);
          throw cms::Exception("CommunicationError",e.message());
        }
        // for performance measurements only using global variable!
        addMyXDAQMeasurement(numBytesSent);
      }
      else
        edm::LogError("writeI2ORegistry") << "StreamerI2OWriter::WriteI2ORegistry: No " 
                      << worker_->destinationName_
                      << "destination in configuration";
    }
  }

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

  void StreamerI2OWriter::writeI2OData(EventMsgBuilder const& eventMessage)
  {
    char* buffer = (char*) eventMessage.startAddress();
    unsigned int size = eventMessage.size();
    EventMsgView eventView(eventMessage.startAddress());
    edm::RunNumber_t runid = eventView.run();
    edm::EventNumber_t eventid = eventView.event();

    FDEBUG(10) << "StreamerI2OWriter::writeI2OData: data size (in bytes) = " << size << std::endl;
    FDEBUG(10) << "StreamerI2OWriter::writeI2OData: run, event = " 
               << runid << " " << eventid << std::endl;
    // Use common chain creation routine
    toolbox::mem::Reference *head = 0;
    unsigned int numBytesSent = 0;
    head = SMi2oSender::createI2OFragmentChain(buffer, size,
                                               worker_->pool_,
                                               max_i2o_sm_datasize_,
                                               sizeof(I2O_SM_DATA_MESSAGE_FRAME),
                                               (unsigned short)I2O_SM_DATA,
                                               worker_->app_,
                                               worker_->destination_,
                                               numBytesSent);

    // set the run and event numbers in each I2O fragment
    toolbox::mem::Reference *currentElement = head;
    while(currentElement != NULL)
    {
      I2O_SM_DATA_MESSAGE_FRAME *dataMsg =
        (I2O_SM_DATA_MESSAGE_FRAME *) currentElement->getDataLocation();
      dataMsg->runID = (unsigned int) runid;     // convert to known size
      dataMsg->eventID = (unsigned int) eventid; // convert to known size
      currentElement = currentElement->getNextReference();
    }

    if(head != NULL) {
      if(worker_->destination_ !=0)
      {
        FDEBUG(10) << "StreamerI2OWriter::writeI2OData: posting data chain frame " << std::endl;
        try{
          worker_->app_->getApplicationContext()->postFrame(head,
                   worker_->app_->getApplicationDescriptor(),worker_->destination_);
        }
        catch(xcept::Exception &e)
        {
          edm::LogError("writeI2ORegistry") << "Exception writeI2OData postFrame" 
                        << xcept::stdformat_exception_history(e);
          throw cms::Exception("CommunicationError",e.message());
        }
        // for performance measurements only using global variable!
        addMyXDAQMeasurement(numBytesSent);
      }
      else
        edm::LogError("writeI2ORegistry") 
                      << "StreamerI2OWriter::writeI2OData: No " << worker_->destinationName_
                      << "destination in configuration";
    }
    // Keep the note below!
    // Do not need to release buffers in the sender as this is done in the transport
    // layer. See tcp::PeerTransportSender::post and tcp::PeerTransportSender::svc
    // in TriDAS_v3.3/daq/pt/tcp/src/common/PeerTransportSender.cc
    //
    // What if there was an error in receiving though? 
    // Do we need some handshaking to release the event data?
  }

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  void StreamerI2OWriter::writeI2OOther(OtherMessageBuilder othermsg)
  {
    char* buffer = (char*) othermsg.startAddress();
    unsigned int size = othermsg.size();
    FDEBUG(9) << "StreamerI2OWriter: write other message size = " << size << std::endl;
    // Use common chain creation routine
    toolbox::mem::Reference *head = 0;
    unsigned int numBytesSent = 0;
    head = SMi2oSender::createI2OFragmentChain(buffer, size,
                                               worker_->pool_,
                                               max_i2o_other_datasize_,
                                               sizeof(I2O_SM_OTHER_MESSAGE_FRAME),
                                               (unsigned short)I2O_SM_OTHER,
                                               worker_->app_,
                                               worker_->destination_,
                                               numBytesSent);

    // set the run and other data in each I2O fragment
    toolbox::mem::Reference *currentElement = head;
    while(currentElement != NULL)
    {
      I2O_SM_OTHER_MESSAGE_FRAME *otherMsg =
        (I2O_SM_OTHER_MESSAGE_FRAME *) currentElement->getDataLocation();
      otherMsg->otherData = 0;
      currentElement = currentElement->getNextReference();
    }

    if(head != NULL) {
      if(worker_->destination_ !=0)
      {
        FDEBUG(10) << "StreamerI2OWriter::writeI2OOther: posting data chain frame " << std::endl;
        try{
          worker_->app_->getApplicationContext()->postFrame(head,
                   worker_->app_->getApplicationDescriptor(),worker_->destination_);
        }
        catch(xcept::Exception &e)
        {
          edm::LogError("writeI2OOther") << "Exception writeI2OOther postFrame" 
                        << xcept::stdformat_exception_history(e);
          throw cms::Exception("CommunicationError",e.message());
        }
        // for performance measurements only using global variable!
        addMyXDAQMeasurement(numBytesSent);
      }
      else
        edm::LogError("writeI2OOther") 
                      << "StreamerI2OWriter::writeI2OOther: No " << worker_->destinationName_
                      << "destination in configuration";
    }
  }

}
