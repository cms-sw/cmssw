#ifndef _StreamerI2OWriter_h
#define _StreamerI2OWriter_h 

/*
   Description:
     Header file for I2O class to be used with StreamerOutputModule.
     See CMS EvF Storage Manager wiki page for further notes.

   $Id$
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
    void writeI2OOther(OtherMessageBuilder otherMsg);

    unsigned int i2o_max_size_;
    unsigned int max_i2o_sm_datasize_;
    unsigned int max_i2o_registry_datasize_; 
    unsigned int max_i2o_other_datasize_; 
    unsigned int max_i2o_DQM_datasize_; 
  };
}

#endif
