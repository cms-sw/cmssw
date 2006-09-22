#ifndef _StreamerFileWriter_h
#define _StreamerFileWriter_h 

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "IOPool/Streamer/interface/StreamerOutputFile.h"
#include "IOPool/Streamer/interface/StreamerOutputIndexFile.h"
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
  class StreamerFileWriter 
  {
  public:

    explicit StreamerFileWriter(edm::ParameterSet const& ps);
    explicit StreamerFileWriter(string const& fileName, string const& indexFileName);
    ~StreamerFileWriter();

    //void doOutputHeader(std::auto_ptr<InitMsgBuilder> init_message);    
    void doOutputHeader(InitMsgBuilder const& init_message);    
    void doOutputHeader(InitMsgView const& init_message);    

    //void doOutputEvent(std::auto_ptr<EventMsgBuilder> msg);
    void doOutputEvent(EventMsgBuilder const& msg);
    void doOutputEvent(EventMsgView const& msg);

    void stop();
    // Returns the sizes of EOF records, call them after 
    // u called stop, just before destruction
    uint32 getStreamEOFSize() const {return stream_eof_size_;}
    uint32 getIndexEOFSize() const {return index_eof_size_;}
 
  private:
    void updateHLTStats(std::vector<uint8> const& packedHlt);

    std::auto_ptr<StreamerOutputFile> stream_writer_;
    std::auto_ptr<StreamerOutputIndexFile> index_writer_; 
    uint32 hltCount_;
    std::vector<uint32> hltStats_;
    uint32 index_eof_size_;
    uint32 stream_eof_size_;

  };
}
#endif

