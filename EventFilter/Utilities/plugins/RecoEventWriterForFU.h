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

#include "boost/shared_ptr.hpp"

#include <iostream>
#include <vector>
#include <memory>
#include <string>

namespace evf
{
  class ParameterSetDescription;
  class RecoEventWriterForFU 
  {
  public:

    explicit RecoEventWriterForFU(edm::ParameterSet const& ps);
    ~RecoEventWriterForFU();

    static void fillDescription(edm::ParameterSetDescription& desc);

    void setInitMessageFile(std::string const&);
    void setOutputFile(std::string const&);
    void closeOutputFile();

    void doOutputHeader(InitMsgBuilder const& init_message);    
    void doOutputHeader(InitMsgView const& init_message);    

    void doOutputEvent(EventMsgBuilder const& msg);
    void doOutputEvent(EventMsgView const& msg);

    void start(){}
    void stop(){}

    uint32 get_adler32_ini() const { return preamble_adler32_;}
    uint32 get_adler32() const { return stream_writer_events_->adler32();}

  private:

    boost::shared_ptr<StreamerOutputFile> stream_writer_preamble_;
    boost::shared_ptr<StreamerOutputFile> stream_writer_events_;
    uint32 preamble_adler32_=1;

  };
}
#endif
