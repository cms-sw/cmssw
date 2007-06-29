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
    uint64 offset_to_return = dqmstreamfile_->current_offset_;

    /** Offset of last written event */
    dqmstreamfile_->last_event_offset_ = dqmstreamfile_->current_offset_;

    writeDQMEventHeader(inmsg);
    dqmstreamfile_->ost()->write((const char*) inmsg.eventAddress(),
                              inmsg.size() - inmsg.headerSize());
    dqmstreamfile_->current_offset_ += (uint64)
                                     (inmsg.size()
                                      - inmsg.headerSize()) ;
    ++dqmstreamfile_->events_;
    if (!(dqmstreamfile_->events_ % 100)) {
        dqmstreamfile_->ost()->flush();
    }
    return offset_to_return;
  }

  uint64 StreamDQMOutputFile::write(const DQMEventMsgView& ineview)
  {
    /** Offset where current event starts */
    uint64 offset_to_return = dqmstreamfile_->current_offset_; 

    /** Offset of last written event */
    dqmstreamfile_->last_event_offset_ = dqmstreamfile_->current_offset_; 

    writeDQMEventHeader(ineview);
    dqmstreamfile_->ost()->write((const char*) ineview.eventAddress(), 
                              ineview.size() - ineview.headerSize());
    dqmstreamfile_->current_offset_ += (uint64)
                                     (ineview.size() 
                                      - ineview.headerSize()) ;
    ++dqmstreamfile_->events_;
    if (!(dqmstreamfile_->events_ % 100)) {
	dqmstreamfile_->ost()->flush();
    }
    return offset_to_return;
  }

  void StreamDQMOutputFile::writeDQMEventHeader(const DQMEventMsgBuilder& inmsg)
  {
    dqmstreamfile_->ost()->write((const char*) 
                                inmsg.startAddress(), 
                                inmsg.headerSize()) ;
    dqmstreamfile_->current_offset_ += (uint64) inmsg.headerSize(); 
  }

  void StreamDQMOutputFile::writeDQMEventHeader(const DQMEventMsgView& ineview)
  {
    dqmstreamfile_->ost()->write((const char*)
                                ineview.startAddress(),
                                ineview.headerSize()) ;
    dqmstreamfile_->current_offset_ += (uint64) ineview.headerSize();
  }

