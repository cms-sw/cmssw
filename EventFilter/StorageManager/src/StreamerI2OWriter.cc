/*
   Author: Harry Cheung, Kurt Biery, FNAL  (taken from I2OConsumer.cc)

   Description:
     EDM output module that will send the data via I2O frames to 
     to the Storage Manager. In this preproduction version the 
     destination is hardwired and provided through a global variable.
     See CMS EventFilter wiki page for further notes.

   Modification:
     version 1.1 2006/7/26
       Initial implementation, starting with I2OConsumer.cc
       but with new message classes.

*/

// why do I need this?
#include "IOPool/Streamer/interface/EventStreamOutput.h"

#include "EventFilter/StorageManager/interface/i2oStorageManagerMsg.h"
#include "EventFilter/StorageManager/src/StreamerI2OWriter.h"
//-#include "IOPool/Streamer/interface/Messages.h"
#include "DataFormats/Common/interface/EventID.h"
#include "EventFilter/StorageManager/interface/SMi2oSender.h"

#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "xdaq/Application.h"
#include "xdaq/ApplicationContext.h"
#include "xdaq/ApplicationGroup.h"

#include "xcept/include/xcept/Exception.h"
#include "xcept/include/xcept/tools.h"

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

#include <wait.h>

extern xdaq::Application* getMyXDAQPtr();
extern toolbox::mem::Pool *getMyXDAQPool();
extern xdaq::ApplicationDescriptor* getMyXDAQDest();
extern xdaq::ApplicationDescriptor* getMyXDAQDest(unsigned int);

// for performance measurements
extern void addMyXDAQMeasurement(unsigned long size);
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

  //I2OStreamWorker::I2OStreamWorker(const string& s = "testI2OReceiver" ):
  I2OStreamWorker::I2OStreamWorker(const string& s = "simpleI2OReceiver", const int dsi = -1):
                      destinationName_(s),
                      app_(getMyXDAQPtr()),
                      pool_(getMyXDAQPool())
  {
    if(dsi<0)
      destination_ = getMyXDAQDest();
    else
      destination_ = getMyXDAQDest(dsi);
    if(destination_ == 0)
      {
	ostringstream os;
	os << "Could not find instance " << dsi << " for destination " << s;
	XCEPT_RAISE (toolbox::fsm::exception::Exception, 
		     os.str().c_str());
      }
    //LOG4CPLUS_INFO(app_->getApplicationLogger(),"Making I2OStreamWorker");
    FDEBUG(10) << "StreamerI2OWriter: Making I2OStreamWorker" << std::endl;
  }

  // ----------------------------------

  StreamerI2OWriter::StreamerI2OWriter(edm::ParameterSet const& ps):
    worker_(new I2OStreamWorker(ps.getParameter<string>("DestinationName"),
				ps.getUntrackedParameter<int>("smInstance",-1))),
    i2o_max_size_(ps.getUntrackedParameter<int>("i2o_max_size",I2O_MAX_SIZE))
  {
    FDEBUG(10) << "StreamerI2OWriter: Constructor" << std::endl;
    // max i2o frame size must be less than 262140 or less we get
    // a crash in the xdaq synchronous tcp transport layer
    // (also needs to be a multiple of 64 bits)
    if(i2o_max_size_ > I2O_ABSOLUTE_MAX_SIZE) {
      int old_i2o_max_size = i2o_max_size_;
      i2o_max_size_ = I2O_ABSOLUTE_MAX_SIZE;
      LOG4CPLUS_INFO(worker_->app_->getApplicationLogger(),
        "StreamerI2OWriter: user defined i2o_max_size too large for xdaq tcp, changed from " 
                     << old_i2o_max_size << " to " << i2o_max_size_);
    }
    // the total i20 frame size must be a multiple of 64 bits (8 bytes)
    if((i2o_max_size_ & 0x7) != 0) {
      int old_i2o_max_size = i2o_max_size_;
      // round it DOWN as this is the maximum size
      i2o_max_size_ = ((i2o_max_size_ >> 3) + 0) << 3;
      LOG4CPLUS_INFO(worker_->app_->getApplicationLogger(),
        "StreamerI2OWriter: user defined i2o_max_size not multiple of 64 bits, changed from " 
                     << old_i2o_max_size << " to " << i2o_max_size_);
    }
    // now don't have to hardwire use sizeof(frame) now gives header
    //max_i2o_sm_datasize_ =  i2o_max_size_ - 28 - 136 ;
    //max_i2o_registry_datasize_ = i2o_max_size_ - 28 - 116; 
    max_i2o_sm_datasize_ =  i2o_max_size_ - sizeof(I2O_SM_DATA_MESSAGE_FRAME) ;
    max_i2o_registry_datasize_ = i2o_max_size_ - sizeof(I2O_SM_PREAMBLE_MESSAGE_FRAME); 

    FDEBUG(10) <<"StreamerI2OWriter: Ctor i2o_max_size_: "<< i2o_max_size_ << std::endl;
    FDEBUG(10) <<"StreamerI2OWriter: Ctor max_i2o_sm_datasize_: "<< max_i2o_sm_datasize_ << std::endl;
    FDEBUG(10) <<"StreamerI2OWriter: Ctor max_i2o_registry_datasize_: "<< max_i2o_registry_datasize_ << std::endl;
  }
  
  StreamerI2OWriter::~StreamerI2OWriter()
  {
    delete worker_;
  }

  int StreamerI2OWriter::i2oyield(unsigned int microseconds)
  {
    // used to block (should yield to other threads)
    usleep(microseconds);
    return 0;
  }

  void StreamerI2OWriter::doOutputHeader(InitMsgBuilder const& initMessage)
  {
    writeI2ORegistry(initMessage);
  }

  void StreamerI2OWriter::doOutputEvent(EventMsgBuilder const& eventMessage)
  {
    // First test if memory pool can hold another event
    FDEBUG(10) << "StreamerI2OWriter: write event to destination" << std::endl;
    int sz = eventMessage.size();
    FDEBUG(10) << "StreamerI2OWriter: event sz = " << sz << std::endl;
    // Should block if memory pool is too full until it has more room
    // (Is this the only thread posting frames to do with this process?)
    // should really check if the high threshold is actually set!
    if (!worker_->pool_->isHighThresholdExceeded())
    {
      // now we can post the data
      writeI2OData(eventMessage);
      // should test here is there was an error posting the data
    }
    else
    {
      LOG4CPLUS_INFO(worker_->app_->getApplicationLogger(),
        "StreamerI2OWriter: High threshold exceeded in memory pool, blocking on I2O output"
                     << " max committed size "
                     << worker_->pool_->getMemoryUsage().getCommitted()
                     << " mem size used (bytes) "
                     << worker_->pool_->getMemoryUsage().getUsed());
      unsigned int yc = 0;
      while (worker_->pool_->isLowThresholdExceeded())
      {
        // Does this work? We need to receive e.g. the halt XOAP message!
        // Is uwait thread safe!?
        this->i2oyield(50000);
        yc++;
      }
      LOG4CPLUS_INFO(worker_->app_->getApplicationLogger(),
                     "StreamerI2OWriter: Yielded " << yc << " times before low threshold reached");
      // now we can post the data
      writeI2OData(eventMessage);
    }
  }

  void StreamerI2OWriter::stop()
  {
    FDEBUG(9) << "StreamerI2OWriter: sending terminate run" << std::endl;
  //  int sz = 0;
    // The special "other" message is hardwired to send a
    // terminate run message (close file)
    std::cout << "stop called" << std::endl;
    // make a DONE message
    //int sz = 16;
// somethign wwrong here!
    char* dummyBuffer = new char[16];
    //std::auto_ptr<OtherMessageBuilder> othermsg(
    //                    new OtherMessageBuilder(dummyBuffer,Header::DONE));
    OtherMessageBuilder othermsg(dummyBuffer,Header::DONE);
    //std::cout << "making other message code = " << othermsg->code()
    //          << " and size = " << othermsg->size() << std::endl;
    //std::cout << "making other message code = " << othermsg.code()
    //          << " and size = " << othermsg.size() << std::endl;
  //  writeI2OOther((const char*)pb.buffer(),sz);
  //  writeI2OOther((const char*)othermsg,sz);
    //writeI2OOther(*othermsg);
    writeI2OOther(othermsg);
    delete [] dummyBuffer;
  }

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  void StreamerI2OWriter::writeI2ORegistry(InitMsgBuilder const& initMessage)
  {
    char* buffer = (char*) initMessage.startAddress();
    unsigned int size = initMessage.size();

    FDEBUG(9) << "writeI2ORegistry: size = " << size << std::endl;
    // should really get rid of this
    //std::string temp4print(buffer,size);
    //FDEBUG(10) << "writeI2ORegistry data = " << temp4print << std::endl;
    // work out the maximum bytes per frame for data
    unsigned int maxSizePerFrame = max_i2o_registry_datasize_;
    //unsigned int headerNeededSize = sizeof(MsgCode::Codes)+sizeof(InitMsg::Header);
    //FDEBUG(10) << "headerNeededSize = " << headerNeededSize << std::endl;
    //unsigned int maxInMsgDataFragSizeInBytes = maxSizePerFrame - headerNeededSize;
    //int size4data = size - headerNeededSize;
    // for the registry we do not need a header for each message fragment
    // as only the Storage Manager takes care of reassembling frames without this
    unsigned int maxInMsgDataFragSizeInBytes = maxSizePerFrame;
    int size4data = size;
    unsigned int numFramesNeeded = size4data/maxInMsgDataFragSizeInBytes;
    unsigned int remainder = size4data%maxInMsgDataFragSizeInBytes;

    if (remainder > 0) numFramesNeeded++;
    FDEBUG(9) << "StreamerI2OWriter::writeI2ORegistry: number of frames needed = " << numFramesNeeded 
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
        start = i*maxInMsgDataFragSizeInBytes;
        if (i < ((int)numFramesNeeded)-1 || remainder == 0)
          thisSize = maxInMsgDataFragSizeInBytes;
        else
          thisSize = remainder;
      }
      // This is the actual size to be posted to I2O
      size_t msgSizeInBytes = i2o_max_size_ ;
      FDEBUG(10) << "msgSizeInBytes registry frame size = " << msgSizeInBytes << std::endl;
      FDEBUG(10) << "I2O_MESSAGE_FRAME size = " << sizeof(I2O_MESSAGE_FRAME) << std::endl;
      FDEBUG(10) << "I2O_PRIVATE_MESSAGE_FRAME size = " << sizeof(I2O_PRIVATE_MESSAGE_FRAME) 
                 << std::endl;
      if(thisSize > max_i2o_registry_datasize_) 
      {
        // this should never happen!
        std::cerr << "StreamerI2OWriter::writeI2ORegistry: unexpected error! "
                  << "Data larger than one frame: abort " << std::endl;
        return;
      }
      try
      {
        FDEBUG(10) << "StreamerI2OWriter: getting memory pool frame" << std::endl;
        toolbox::mem::Reference* bufRef =
           toolbox::mem::getMemoryPoolFactory()->getFrame(worker_->pool_,i2o_max_size_);

        FDEBUG(10) << "StreamerI2OWriter: setting up frame pointers" << std::endl;
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
          std::cerr << "StreamerI2OWriter::WriteI2ORegistry: exception in getting source tid "
               <<  xcept::stdformat_exception_history(e) << endl;
        }
        try
        {
          stdMsg->TargetAddress    =
                  i2o::utils::getAddressMap()->getTid(worker_->destination_);
        }
        catch(xdaq::exception::ApplicationDescriptorNotFound e)
        {
          std::cerr << "StreamerI2OWriter::WriteI2ORegistry: exception in getting destination tid "
               <<  xcept::stdformat_exception_history(e) << endl;
        }

        stdMsg->Function         = I2O_PRIVATE_MESSAGE;
        stdMsg->VersionOffset    = 0;
        stdMsg->MsgFlags         = 0;  // normal message (not multicast)

        pvtMsg->XFunctionCode    = I2O_SM_PREAMBLE;
        pvtMsg->OrganizationID   = XDAQ_ORGANIZATION_ID;
        //msg->dataSize = size;
        //msg->dataSize = thisSize+headerNeededSize;
        msg->dataSize = thisSize;
        // Fill in the long form of the source (HLT) identifier
        std::string url = worker_->app_->getApplicationDescriptor()->getContextDescriptor()->getURL();
        if(url.size() > MAX_I2O_SM_URLCHARS)
        {
          LOG4CPLUS_INFO(worker_->app_->getApplicationLogger(),
            "StreamerI2OWriter: Error! Source URL truncated");
          for(int i=0; i< MAX_I2O_SM_URLCHARS; i++) msg->hltURL[i] = url[i];
        } else {
          //for(int i=0; i< MAX_I2O_SM_URLCHARS; i++) msg->hltURL[i] = " ";
          for(int i=0; i< (int)url.size(); i++) msg->hltURL[i] = url[i];
        } 
        std::string classname = worker_->app_->getApplicationDescriptor()->getClassName();
        if(classname.size() > MAX_I2O_SM_URLCHARS)
        {
          LOG4CPLUS_INFO(worker_->app_->getApplicationLogger(),
            "StreamerI2OWriter: Error! Source ClassName truncated");
          for(int i=0; i< MAX_I2O_SM_URLCHARS; i++) msg->hltClassName[i] = classname[i];
        } else {
          //for(int i=0; i< MAX_I2O_SM_URLCHARS; i++) msg->hltClassName[i] = " ";
          for(int i=0; i< (int)url.size(); i++) msg->hltClassName[i] = classname[i];
        } 
        msg->hltLocalId = worker_->app_->getApplicationDescriptor()->getLocalId();
        msg->hltInstance = worker_->app_->getApplicationDescriptor()->getInstance();
        msg->hltTid = i2o::utils::getAddressMap()->getTid(worker_->app_->getApplicationDescriptor());
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
        // fill in the data for this fragment
        if(thisSize != 0)
        {
          for (unsigned int i=0; i<thisSize; i++){
            msg->dataPtr()[i] = *(buffer+i + start);
          }
          // should really get rid of this
          std::string temp4print(msg->dataPtr(),size);
          FDEBUG(9) << "StreamerI2OWriter::WriteI2ORegistry: string msg_>data = " 
                     << temp4print << std::endl;
        } else {
          std::cout << "StreamerI2OWriter::writeI2ORegistry: Error! Sending zero size data!?" 
                    << std::endl;
        }
        bufRef->setDataSize(msgSizeInBytes);
        }
      catch(toolbox::mem::exception::Exception e)
      {
        std::cout << "StreamerI2OWriter::WriteI2ORegistry::exception in allocating frame "
             <<  xcept::stdformat_exception_history(e) << endl;
        return;
      }
      catch(...)
      { 
        std::cout << "StreamerI2OWriter::WriteI2ORegistry: unknown exception in allocating frame " << endl;
        return;
      } // end try
    } // end loop over frame fragments

    // don't postFrame until all frames in chain are set up and there was no error!
    // need to make a test here later
    FDEBUG(10) << "StreamerI2OWriter::WriteI2ORegistry: checking if destination exist" << std::endl;
    if(worker_->destination_ !=0)
    {
      FDEBUG(10) << "StreamerI2OWriter::WriteI2ORegistry: posting registry frame " << std::endl;
      worker_->app_->getApplicationContext()->postFrame(head,
                   worker_->app_->getApplicationDescriptor(),worker_->destination_);
      // for performance measurements only using global variable!
      //addMyXDAQMeasurement((unsigned long)msgSizeInBytes);
      addMyXDAQMeasurement((unsigned long)(numFramesNeeded * i2o_max_size_));
    }
    else
      std::cerr << "StreamerI2OWriter::WriteI2ORegistry: No " << worker_->destinationName_
                << "destination in configuration" << std::endl;
  }

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

  void StreamerI2OWriter::writeI2OData(EventMsgBuilder const& eventMessage)
  //
  // function to write the data buffer in I2O frames. If more than one
  // frame is needed a chain is created and then posted.
  {
    char* buffer = (char*) eventMessage.startAddress();
    unsigned int size = eventMessage.size();
    // don't have hlt and l1 trigger bit counts
    //EventMsgView eventView(eventMessage.startAddress(), 0, 0);
    EventMsgView eventView(eventMessage.startAddress());
    edm::RunNumber_t runid = eventView.run();
    edm::EventNumber_t eventid = eventView.event();

    // should really test the size >0
    FDEBUG(10) << "StreamerI2OWriter::writeI2OData: data size (in bytes) = " << size << std::endl;
    FDEBUG(10) << "StreamerI2OWriter::writeI2OData: run, event = " 
               << runid << " " << eventid << std::endl;
    // must decide how many frames we need to use to send this message
    // HEREHERE need to say should have used a different name here
    // here I used the same name maxSizeInBytes when I should not have!
    // This maxSizeInBytes here is for size for available data
    // the maxSizeInBytes in the loop below is for the total size as
    // posted to I2O

    unsigned int maxSizeInBytes = max_i2o_sm_datasize_;
    ///unsigned int maxSizeInBytes = MAX_I2O_SM_DATASIZE;
  //-  unsigned int headerNeededSize = sizeof(MsgCode::Codes)+sizeof(EventMsg::Header);
    //std::cout << "headerNeededSize = " << headerNeededSize << std::endl;
  //-  unsigned int maxEvMsgDataFragSizeInBytes = maxSizeInBytes - headerNeededSize;
//    unsigned int numFramesNeeded = size/maxSizeInBytes;
//    unsigned int remainder = size%maxSizeInBytes;
    unsigned int maxEvMsgDataFragSizeInBytes = maxSizeInBytes;
  //-  int size4data = size - headerNeededSize;
    int size4data = size;
    unsigned int numFramesNeeded = size4data/maxEvMsgDataFragSizeInBytes;
    unsigned int remainder = size4data%maxEvMsgDataFragSizeInBytes;

    if (remainder > 0) numFramesNeeded++;
    FDEBUG(10) << "StreamerI2OWriter::writeI2OData: number of frames needed = " << numFramesNeeded 
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
      //int minlen = 50;
      //if(minlen > (int)thisSize) minlen = thisSize;
      //std::string temp4print(buffer+start,minlen);
      //FDEBUG(10) << "StreamerI2OWriter::writeI2OData: data = " << temp4print << std::endl;

      //size_t msgSizeInBytes = max_i2o_sm_datasize_;
      //size_t msgSizeInBytes = thisSize+headerNeededSize;
  //-    size_t msgSizeInBytes = sizeof(I2O_SM_DATA_MESSAGE_FRAME)+thisSize+headerNeededSize;
      size_t msgSizeInBytes = sizeof(I2O_SM_DATA_MESSAGE_FRAME)+thisSize;
      // round up size to multiple of 8 bytes (i2o uses 32-bit words, but use 64 for future)
      if((msgSizeInBytes & 0x7) != 0)
	msgSizeInBytes = ((msgSizeInBytes >> 3) + 1) << 3;
      FDEBUG(10) << "StreamerI2OWriter::writeI2OData: msgSizeInBytes data frame size = " 
                 << msgSizeInBytes << std::endl;
      if(thisSize > max_i2o_sm_datasize_) //MAX_I2O_SM_DATASIZE+MAX_I2O_SM_DATASIZE)
      {
        // this should never happen - get rid of this later?
        std::cerr << "StreamerI2OWriter::writeI2OData: unexpected error! "
                  << "Data larger than one frame abort " << std::endl;
        return;
      }
      try
      {
        FDEBUG(10) << "StreamerI2OWriter::writeI2OData: getting frame" << std::endl;
        toolbox::mem::Reference* bufRef =
           toolbox::mem::getMemoryPoolFactory()->getFrame(worker_->pool_,i2o_max_size_);
        I2O_MESSAGE_FRAME *stdMsg =
          (I2O_MESSAGE_FRAME*)bufRef->getDataLocation();
        I2O_PRIVATE_MESSAGE_FRAME*pvtMsg = (I2O_PRIVATE_MESSAGE_FRAME*)stdMsg;
        I2O_SM_DATA_MESSAGE_FRAME *msg =
          (I2O_SM_DATA_MESSAGE_FRAME*)stdMsg;

        FDEBUG(10) << "StreamerI2OWriter::writeI2OData: setting MessageSize"
                   << std::endl;
        stdMsg->MessageSize      = msgSizeInBytes >> 2;

        FDEBUG(10) << "StreamerI2OWriter::writeI2OData: getting xdaq stuff"
                   << std::endl;
        try
        {
          stdMsg->InitiatorAddress =
            i2o::utils::getAddressMap()->getTid(worker_->app_->getApplicationDescriptor());
        }
        catch(xdaq::exception::ApplicationDescriptorNotFound e)
        {
          std::cerr << "StreamerI2OWriter::writeI2OData: exception in getting source tid "
                    <<  xcept::stdformat_exception_history(e) << endl;
        }
        try
        {
          stdMsg->TargetAddress    =
                    i2o::utils::getAddressMap()->getTid(worker_->destination_);
        }
        catch(xdaq::exception::ApplicationDescriptorNotFound e)
        {
          std::cerr << "StreamerI2OWriter::writeI2OData: exception in getting destination tid "
                    <<  xcept::stdformat_exception_history(e) << endl;
        }

        FDEBUG(10) << "StreamerI2OWriter::writeI2OData: setting std frame"
                   << std::endl;
        stdMsg->Function         = I2O_PRIVATE_MESSAGE;
        stdMsg->VersionOffset    = 0;
        stdMsg->MsgFlags         = 0;  // normal message (not multicast)

        pvtMsg->XFunctionCode    = I2O_SM_DATA;
        pvtMsg->OrganizationID   = XDAQ_ORGANIZATION_ID;
        //msg->dataSize = thisSize;
    //-    msg->dataSize = thisSize+headerNeededSize;
        msg->dataSize = thisSize;
        // Fill in the long form of the source (HLT) identifier
        FDEBUG(10) << "StreamerI2OWriter::writeI2OData: setting url/class"
                   << std::endl;
        std::string url = worker_->app_->getApplicationDescriptor()->getContextDescriptor()->getURL();
        if(url.size() > MAX_I2O_SM_URLCHARS)
        {
          LOG4CPLUS_INFO(worker_->app_->getApplicationLogger(),
            "StreamerI2OWriter: Error! Source URL truncated");
          for(int i=0; i< MAX_I2O_SM_URLCHARS; i++) msg->hltURL[i] = url[i];
        } else {
          //for(int i=0; i< MAX_I2O_SM_URLCHARS; i++) msg->hltURL[i] = " "; // needed?
          for(int i=0; i< (int)url.size(); i++) msg->hltURL[i] = url[i];
        } 
        std::string classname = worker_->app_->getApplicationDescriptor()->getClassName();
        if(classname.size() > MAX_I2O_SM_URLCHARS)
        {
          LOG4CPLUS_INFO(worker_->app_->getApplicationLogger(),
            "StreamerI2OWriter: Error! Source ClassName truncated");
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
        FDEBUG(10) << "StreamerI2OWriter::writeI2OData: making the chain"
                   << std::endl;
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
        // fill in the data for this fragment
        FDEBUG(10) << "StreamerI2OWriter::writeI2OData: filling data"
                   << std::endl;
        if(thisSize != 0)
        {
        FDEBUG(10) << "StreamerI2OWriter::writeI2OData: loop over data"
                   << std::endl;
          for (unsigned int j=0; j<thisSize; j++)
          {
  //      FDEBUG(10) << "StreamerI2OWriter::writeI2OData: loop " << j << " start " << start << endl;
  //-          msg->dataPtr()[i+headerNeededSize] = *(buffer+headerNeededSize+i + start);
            msg->dataPtr()[j] = *(buffer+j + start);
            //msg->data[i+headerNeededSize] = *(buffer+headerNeededSize+i + start);
          }
        FDEBUG(10) << "StreamerI2OWriter::writeI2OData: finish loop"
                   << std::endl;
          // remake header -- no need with new message frags and new split
  //-        char* dummyBuffer = new char[max_i2o_sm_datasize_];
          //char dummyBuffer[MAX_I2O_SM_DATASIZE];
  //-        EventMsg msgFrag(dummyBuffer, thisSize+headerNeededSize, 
  //-          eventid, runid, thisCount+1, numFramesNeeded);
  //-        for (unsigned int i=0; i<headerNeededSize; i++)
  //-        {
  //-          msg->dataPtr()[i] = dummyBuffer[i];
            //msg->data[i] = dummyBuffer[i];
  //-        }
  //-        delete [] dummyBuffer;
          // should get rid of this later - just used to create a dump for checking
  //-        minlen = 50;
  //-        if(minlen > (int)thisSize) minlen = thisSize;
  //-        std::string temp4print(msg->dataPtr(),minlen);
  //-        FDEBUG(10) << "StreamerI2OWriter::writeI2OData: msg data = " << temp4print << std::endl;
        } else {
          std::cout << "StreamerI2OWriter::writeI2OData: Error! Sending zero size data!?" << std::endl;
        }

     
        FDEBUG(10) << "StreamerI2OWriter::writeI2OData: setDataSize right?"
                   << std::endl;
// need to set the actual buffer size that is being sent
        bufRef->setDataSize(msgSizeInBytes);
      }
      catch(toolbox::mem::exception::Exception e)
      {
        std::cerr << "StreamerI2OWriter::writeI2OData: exception in allocating frame "
                  <<  xcept::stdformat_exception_history(e) << endl;
        return;  // is this the right action?
      }
      catch(...)
      { 
        std::cerr << "StreamerI2OWriter::writeI2OData: unknown exception in allocating frame" << endl;
        return;  // is this the right action?
      } //end try for frame allocation
    } //end loop over frame

    // don't postFrame until all frames in chain are set up and there was no error!
    // need to make a test here later
    if(worker_->destination_ !=0)
      {
        FDEBUG(10) << "StreamerI2OWriter::writeI2OData: posting data chain frame " << std::endl;
        try{
	  worker_->app_->getApplicationContext()->postFrame(head,
							    worker_->app_->getApplicationDescriptor(),worker_->destination_);
	}
	catch(xcept::Exception &e)
	  {
	    LOG4CPLUS_ERROR(worker_->app_->getApplicationLogger(),
			    "Exception writeI2OData postFrame" 
			    << xcept::stdformat_exception_history(e));
	    throw cms::Exception("CommunicationError",e.message());
	  }
        // for performance measurements only using global variable!
        addMyXDAQMeasurement((unsigned long)(numFramesNeeded * i2o_max_size_));
      }
    else
      std::cerr << "StreamerI2OWriter::writeI2OData: No " << worker_->destinationName_
                << "destination in configuration" << std::endl;
    // Do not need to release buffers in the sender as this is done in the transport
    // layer. See tcp::PeerTransportSender::post and tcp::PeerTransportSender::svc
    // in TriDAS_v3.3/daq/pt/tcp/src/common/PeerTransportSender.cc
    //
    // What if there was an error in receiving though? 
    // Shouldn't I want a handshake to release the event data?
  }

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  //void StreamerI2OWriter::writeI2OOther(OtherMessageBuilder& othermsg)
  void StreamerI2OWriter::writeI2OOther(OtherMessageBuilder othermsg)
  {
    // char* buffer = (char*) othermsg.startAddress();
    unsigned int size = othermsg.size();
    FDEBUG(9) << "StreamerI2OWriter: write other message size = " << size << std::endl;
    size_t msgSizeInBytes = sizeof(I2O_SM_OTHER_MESSAGE_FRAME);
    // will assume msgSizeInBytes is smaller than I2O_MAX_SIZE

    try
    {
      // should test that msgSizeInBytes is smaller than I2O_MAX_SIZE and
      // use msgSizeInBytes instead of I2O_MAX_SIZE in mem pool frame size
      toolbox::mem::Reference* bufRef =
         toolbox::mem::getMemoryPoolFactory()->getFrame(worker_->pool_,i2o_max_size_);

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
        std::cerr << "StreamerI2OWriter::WriteI2OOther:exception in getting source tid "
                  <<  xcept::stdformat_exception_history(e) << endl;
      }
      try
      {
        stdMsg->TargetAddress    =
                  i2o::utils::getAddressMap()->getTid(worker_->destination_);
      }
      catch(xdaq::exception::ApplicationDescriptorNotFound e)
      {
        std::cerr << "StreamerI2OWriter::WriteI2OOther:exception in getting destination tid "
                  <<  xcept::stdformat_exception_history(e) << endl;
      }

      stdMsg->Function         = I2O_PRIVATE_MESSAGE;
      stdMsg->VersionOffset    = 0;
      stdMsg->MsgFlags         = 0;  // Point-to-point 

      pvtMsg->XFunctionCode    = I2O_SM_OTHER;
      pvtMsg->OrganizationID   = XDAQ_ORGANIZATION_ID;
      // Sending DONE (other) message meaning terminate run
      ///*msg->dataSize = size;
      msg->dataSize = 0;
      //msg->dataSize = size;
      msg->otherData = 0;  // should put the run number here and change its name
      // Fill in the long form of the source (HLT) identifier
      std::string url = worker_->app_->getApplicationDescriptor()->getContextDescriptor()->getURL();
      if(url.size() > MAX_I2O_SM_URLCHARS)
      {
        LOG4CPLUS_INFO(worker_->app_->getApplicationLogger(),
          "StreamerI2OWriter: Error! Source URL truncated");
        for(int i=0; i< MAX_I2O_SM_URLCHARS; i++) msg->hltURL[i] = url[i];
      } else {
        //for(int i=0; i< MAX_I2O_SM_URLCHARS; i++) msg->hltURL[i] = " ";
        for(int i=0; i< (int)url.size(); i++) msg->hltURL[i] = url[i];
      } 
      std::string classname = worker_->app_->getApplicationDescriptor()->getClassName();
      if(classname.size() > MAX_I2O_SM_URLCHARS)
      {
        LOG4CPLUS_INFO(worker_->app_->getApplicationLogger(),
          "StreamerI2OWriter: Error! Source ClassName truncated");
        for(int i=0; i< MAX_I2O_SM_URLCHARS; i++) msg->hltClassName[i] = classname[i];
      } else {
        //for(int i=0; i< MAX_I2O_SM_URLCHARS; i++) msg->hltClassName[i] = " ";
        for(int i=0; i< (int)url.size(); i++) msg->hltClassName[i] = classname[i];
      } 
      msg->hltLocalId = worker_->app_->getApplicationDescriptor()->getLocalId();
      msg->hltInstance = worker_->app_->getApplicationDescriptor()->getInstance();
      msg->hltTid = i2o::utils::getAddressMap()->getTid(worker_->app_->getApplicationDescriptor());

      //for (unsigned int i=0; i<size; i++){
      //   msg->data[i] = *(buffer+i);
      //}

      bufRef->setDataSize(msgSizeInBytes);
      if(worker_->destination_ !=0)
        {
          FDEBUG(9) << "StreamerI2OWriter: posting other message" << std::endl;
          worker_->app_->getApplicationContext()->postFrame(bufRef,
                         worker_->app_->getApplicationDescriptor(),worker_->destination_);
          // for performance measurements only using global variable!
          addMyXDAQMeasurement((unsigned long)msgSizeInBytes);
        }
      else
        std::cerr << "StreamerI2OWriter:No " << worker_->destinationName_
                  << "destination in configuration" << std::endl;
    }
    catch(toolbox::mem::exception::Exception e)
    {
      std::cout << "StreamerI2OWriter:exception in allocating frame "
           <<  xcept::stdformat_exception_history(e) << endl;
      return;
    }
    catch(...)
    { 
      std::cerr << "StreamerI2OWriter:unknown exception in allocating frame " << endl;
      return;
    }
  }

}
