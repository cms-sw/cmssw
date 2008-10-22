#include "IOPool/Streamer/interface/StreamDQMOutputFile.h"

  StreamDQMOutputFile::~StreamDQMOutputFile() 
  {
    delete dqmstreamfile_;
  }

  StreamDQMOutputFile::StreamDQMOutputFile(const std::string& name):
    dqmstreamfile_(new OutputFile(name))
  {
  }

  uint64 StreamDQMOutputFile::write(const DQMEventMsgBuilder& inmsg)
  {
    /** Offset where current event starts */
    uint64 offset_to_return = dqmstreamfile_->current_offset();

    /** Offset of last written event */
    dqmstreamfile_->set_last_event_offset(dqmstreamfile_->current_offset());

    writeDQMEventHeader(inmsg);
    dqmstreamfile_->write((const char*) inmsg.eventAddress(),
                          inmsg.size() - inmsg.headerSize());

    dqmstreamfile_->inc_events();

    return offset_to_return;
  }

  uint64 StreamDQMOutputFile::write(const DQMEventMsgView& ineview)
  {
    /** Offset where current event starts */
    uint64 offset_to_return = dqmstreamfile_->current_offset(); 

    /** Offset of last written event */
    dqmstreamfile_->set_last_event_offset(dqmstreamfile_->current_offset()); 

    writeDQMEventHeader(ineview);
    dqmstreamfile_->write((const char*) ineview.eventAddress(), 
                          ineview.size() - ineview.headerSize());

    dqmstreamfile_->inc_events();

    return offset_to_return;
  }

  void StreamDQMOutputFile::writeDQMEventHeader(const DQMEventMsgBuilder& inmsg)
  {
    dqmstreamfile_->write((const char*) 
                          inmsg.startAddress(), 
                          inmsg.headerSize()) ;
  }

  void StreamDQMOutputFile::writeDQMEventHeader(const DQMEventMsgView& ineview)
  {
    dqmstreamfile_->write((const char*)
                          ineview.startAddress(),
                          ineview.headerSize()) ;
  }
