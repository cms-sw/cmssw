#ifndef _StreamerOutputService_h
#define _StreamerOutputService_h 

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "IOPool/Streamer/interface/StreamerFileIO.h"
#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/InitMsgBuilder.h"
#include "IOPool/Streamer/interface/EventMessage.h"

#include <iostream>
#include <vector>
#include <memory>
#include <string>

namespace edm
{
  class StreamerOutputService 
  {
  public:

    //explicit StreamerOutputService(edm::ParameterSet const& ps);
    explicit StreamerOutputService();
    ~StreamerOutputService();

    void init(std::string fileName, InitMsgView& init_message) ;
    void writeEvent(EventMsgView& msg, uint32 hlt_trig_count);

    void stop(); // shouldn't be called from destructor.

  private:
    void writeHeader(InitMsgBuilder& init_message);
 
     unsigned long maxFileSize_;
     unsigned long maxFileEventCount_;
     unsigned long currentFileSize_;
     unsigned long totalEventCount_;
     unsigned long eventsInFile_;
     unsigned long fileNameCounter_;

     std::string fileName_;
     std::string indexFileName_;

     std::auto_ptr<StreamerOutputFile> stream_writer_;
     std::auto_ptr<StreamerOutputIndexFile> index_writer_;

  };
}
#endif

