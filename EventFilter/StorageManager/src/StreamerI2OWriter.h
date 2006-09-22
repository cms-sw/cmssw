#ifndef _StreamerI2OWriter_h
#define _StreamerI2OWriter_h 

/*
   Author: Harry Cheung, Kurt Biery, FNAL

   Description:
     Header file for I2O class to be used with StreamerOutputModule.
     See CMS EventFilter wiki page for further notes.

   Modification:
     version 1.0 2006/07/25
       Initial implementation starting from I2OConsumer.h but
       changed to use new message classes.

*/

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "IOPool/Streamer/interface/InitMsgBuilder.h"
#include "IOPool/Streamer/interface/EventMsgBuilder.h"
#include "IOPool/Streamer/interface/OtherMessage.h"

namespace edm
{
  struct I2OStreamWorker;

  class StreamerI2OWriter
  {
  public:

    StreamerI2OWriter(edm::ParameterSet const& ps);
    ~StreamerI2OWriter();

    void doOutputHeader(InitMsgBuilder const& initMessage);
    void doOutputEvent(EventMsgBuilder const& eventMessage);
    void stop();

  private:

    I2OStreamWorker* worker_;
    void writeI2ORegistry(InitMsgBuilder const& initMessage);
    void writeI2OData(EventMsgBuilder const& eventMessage);
    // and for sending the Storage Manager other commands
    //void writeI2OOther(OtherMessageBuilder& otherMsg);
    void writeI2OOther(OtherMessageBuilder otherMsg);

    // function used to block until memory pool is not full
    int i2oyield(unsigned int microseconds);  // should yield to other threads instead of blocking

    unsigned long i2o_max_size_;
    unsigned long max_i2o_sm_datasize_;
    unsigned long max_i2o_registry_datasize_; 
  };
}

#endif
