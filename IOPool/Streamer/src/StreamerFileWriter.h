#ifndef IOPool_Streamer_StreamerFileWriter_h
#define IOPool_Streamer_StreamerFileWriter_h 

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "IOPool/Streamer/interface/StreamerOutputFile.h"
#include "IOPool/Streamer/interface/InitMsgBuilder.h"
#include "IOPool/Streamer/interface/EventMsgBuilder.h"
#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/EventMessage.h"
#include "IOPool/Streamer/interface/MsgTools.h"

#include <iostream>
#include <vector>
#include <memory>
#include <string>

namespace edm
{
  class ParameterSetDescription;
  class StreamerFileWriter 
  {
  public:

    explicit StreamerFileWriter(edm::ParameterSet const& ps);
    explicit StreamerFileWriter(std::string const& fileName);
    ~StreamerFileWriter();

    static void fillDescription(ParameterSetDescription& desc);

    void doOutputHeader(InitMsgBuilder const& init_message);    
    void doOutputHeader(InitMsgView const& init_message);    

    void doOutputEvent(EventMsgBuilder const& msg);
    void doOutputEvent(EventMsgView const& msg);

    void start(){}
    void stop(){};

    uint32 get_adler32() const { return stream_writer_->adler32();}

  private:

    std::auto_ptr<StreamerOutputFile> stream_writer_;

  };
}
#endif
