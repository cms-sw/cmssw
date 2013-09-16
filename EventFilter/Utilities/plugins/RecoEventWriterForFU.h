#ifndef IOPool_Streamer_RecoEventWriterForFU_h
#define IOPool_Streamer_RecoEventWriterForFU_h 

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
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

namespace evf
{
  struct RecoEventWriterForFUHeaderParams
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

  struct RecoEventWriterForFUEventParams
  {
    std::vector<unsigned char> hltBits;
    const char* headerPtr;
    uint32 headerSize;

    uint32 fragmentIndex;
    uint32 fragmentCount;
    const char* dataPtr;
    uint32 dataSize;
  };

  class ParameterSetDescription;
  class RecoEventWriterForFU 
  {
  public:

    explicit RecoEventWriterForFU(edm::ParameterSet const& ps);
    ~RecoEventWriterForFU();

    static void fillDescription(edm::ParameterSetDescription& desc);

    void setOutputFiles(std::string &, std::string &);
    void setOutputFile(std::string &);
    void doOutputHeader(InitMsgBuilder const& init_message);    
    void doOutputHeader(InitMsgView const& init_message);    
    void doOutputHeaderFragment(RecoEventWriterForFUHeaderParams const&);

    void doOutputEvent(EventMsgBuilder const& msg);
    void doOutputEvent(EventMsgView const& msg);
    void doOutputEventFragment(RecoEventWriterForFUEventParams const&);

    void start(){}
    void stop();
    // Returns the sizes of EOF records, call them after 
    // u called stop, just before destruction
    uint32 getStreamEOFSize() const {return stream_eof_size_;}

    uint32 get_adler32() const { return stream_writer_events_->adler32();}

  private:
    void updateHLTStats(std::vector<uint8> const& packedHlt);

    std::auto_ptr<StreamerOutputFile> stream_writer_preamble_;
    std::auto_ptr<StreamerOutputFile> stream_writer_postamble_;
    std::auto_ptr<StreamerOutputFile> stream_writer_events_;
    uint32 hltCount_;
    std::vector<uint32> hltStats_;
    uint32 stream_eof_size_;

  };
}
#endif
