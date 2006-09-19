#ifndef _StreamerFileWriter_h
#define _StreamerFileWriter_h 

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "IOPool/Streamer/interface/StreamerOutputFile.h"
#include "IOPool/Streamer/interface/StreamerOutputIndexFile.h"
#include "IOPool/Streamer/interface/InitMsgBuilder.h"
#include "IOPool/Streamer/interface/EventMsgBuilder.h"
#include "IOPool/Streamer/interface/MsgTools.h"

#include <iostream>
#include <vector>
#include <memory>
#include <string>

namespace edm
{
  class StreamerFileWriter 
  {
  public:

    explicit StreamerFileWriter(edm::ParameterSet const& ps);
    ~StreamerFileWriter();

    void doOutputHeader(std::auto_ptr<InitMsgBuilder> init_message);    
    void doOutputEvent(std::auto_ptr<EventMsgBuilder> msg);
    void stop();
 
  private:
    void updateHLTStats(std::vector<uint8> const& packedHlt);

    std::auto_ptr<StreamerOutputFile> stream_writer_;
    std::auto_ptr<StreamerOutputIndexFile> index_writer_; 
    uint32 hltCount_;
    std::vector<uint32> hltStats_;
  };
}
#endif

