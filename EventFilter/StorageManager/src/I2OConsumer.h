#ifndef IOP_I2OCONSUMER_H
#define IOP_I2OCONSUMER_H

/*
   Author: Harry Cheung, FNAL

   Description:
     Header file for I2O output module
     See CMS EventFilter wiki page for further notes.

   Modification:
     version 1.1 2005/11/23
       Initial implementation. Using a temporary event counter
       for the event ID. Need this provided by the EventProcessor
       in the service set, or directly from basic streamer code.

*/

#include "IOPool/Streamer/interface/EventBuffer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edmtest
{
  struct I2OWorker;

  class I2OConsumer
  {
  public:
    I2OConsumer(edm::ParameterSet const& ps, 
		 edm::EventBuffer* buf);

    ~I2OConsumer();

    void bufferReady();
    void stop();
    void sendRegistry(void* buf, int len);

  private:
    I2OWorker* worker_;
    edm::EventBuffer* bufs_;
    // temporary event counter until we can get the real event id
    int SMEventCounter_;
    void writeI2OOther(const char* buffer, unsigned int size);
    void writeI2OData(const char* buffer, unsigned int size);
    void writeI2ORegistry(const char* buffer, unsigned int size);

  };
}

#endif

