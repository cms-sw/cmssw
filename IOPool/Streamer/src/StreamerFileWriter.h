#ifndef IOPool_Streamer_StreamerFileWriter_h
#define IOPool_Streamer_StreamerFileWriter_h 

// $Id: StreamerFileWriter.h,v 1.11.6.1 2009/03/24 21:21:46 biery Exp $

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
  struct StreamerFileWriterHeaderParams
  {
    uint32 runNumber;
    uint32 hltCount;
    const char* headerPtr;
    uint32 headerSize;

    uint32 fragmentIndex;
    uint32 fragmentCount;
    const char* dataPtr;
    uint32 dataSize;
  };

  struct StreamerFileWriterEventParams
  {
    std::vector<unsigned char> hltBits;
    const char* headerPtr;
    uint32 headerSize;

    uint32 fragmentIndex;
    uint32 fragmentCount;
    const char* dataPtr;
    uint32 dataSize;
  };

  class StreamerFileWriter 
  {
  public:

    explicit StreamerFileWriter(edm::ParameterSet const& ps);
    explicit StreamerFileWriter(std::string const& fileName, std::string const& indexFileName);
    ~StreamerFileWriter();

    void doOutputHeader(InitMsgBuilder const& init_message);    
    void doOutputHeader(InitMsgView const& init_message);    
    void doOutputHeaderFragment(StreamerFileWriterHeaderParams const&);

    void doOutputEvent(EventMsgBuilder const& msg);
    void doOutputEvent(EventMsgView const& msg);
    void doOutputEventFragment(StreamerFileWriterEventParams const&);

    void start(){}
    void stop();
    // Returns the sizes of EOF records, call them after 
    // u called stop, just before destruction
    uint32 getStreamEOFSize() const {return stream_eof_size_;}
    uint32 getIndexEOFSize() const {return index_eof_size_;}

    uint32 get_adler32_stream() const { return stream_writer_->adler32();}
    uint32 get_adler32_index()  const { return index_writer_->adler32();}

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
