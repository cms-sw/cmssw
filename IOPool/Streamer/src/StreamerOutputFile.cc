#include "IOPool/Streamer/interface/StreamerOutputFile.h"
#include "IOPool/Streamer/interface/EOFRecordBuilder.h"
#include "FWCore/Utilities/interface/Exception.h"

  StreamerOutputFile::~StreamerOutputFile() {
  }

  StreamerOutputFile::StreamerOutputFile(const std::string& name):
  streamerfile_(new OutputFile(name))
  {
    streamerfile_->set_do_adler(1);
  }

  uint64 StreamerOutputFile::write(const EventMsgBuilder& ineview)
  {
    EventMsgView tmpView(ineview.startAddress());
    return write(tmpView);
  }

  uint64 StreamerOutputFile::write(const EventMsgView& ineview)
  {
    /** Offset where current event starts */
    uint64 offset_to_return = streamerfile_->current_offset();

    /** Offset of last written event */
    streamerfile_->set_last_event_offset(streamerfile_->current_offset());

    writeEventHeader(ineview);
    bool ret = streamerfile_->write((const char*) ineview.eventData(),
                                    ineview.size() - ineview.headerSize());
    if (ret) {
      throw cms::Exception("OutputFile", "write(EventMsgView)")
        << "Error writing streamer event data to "
        << streamerfile_->fileName() << ".  Possibly the output disk "
        << "is full?" << std::endl;
    }

    streamerfile_->inc_events();

    return offset_to_return;
  }

  void StreamerOutputFile::writeEventHeader(const EventMsgView& ineview)
  {
    bool ret = streamerfile_->write((const char*)
                         ineview.startAddress(),
                         ineview.headerSize()) ;
    if (ret) {
      throw cms::Exception("OutputFile", "writeEventHeader")
        << "Error writing streamer event data to "
        << streamerfile_->fileName() << ".  Possibly the output disk "
        << "is full?" << std::endl;
    }
  }

  void StreamerOutputFile::write(const InitMsgBuilder& inview)
  {
    InitMsgView tmpView(inview.startAddress());
    return write(tmpView);
  }

  void StreamerOutputFile::write(const InitMsgView& inview)
  {
    writeStart(inview);
    bool ret = streamerfile_->write((const char*) inview.descData(),
                                    inview.size() - inview.headerSize());
    if (ret) {
      throw cms::Exception("OutputFile", "write(InitMsgView)")
        << "Error writing streamer header data to "
        << streamerfile_->fileName() << ".  Possibly the output disk "
        << "is full?" << std::endl;
    }

    /** Offset of first event to be written */
    streamerfile_->set_first_event_offset(streamerfile_->current_offset());
  }

  void StreamerOutputFile::writeStart(const InitMsgView& inview)
  {
    bool ret = streamerfile_->write((const char*)
                                    inview.startAddress(),
                                    inview.headerSize()) ;
    if (ret) {
      throw cms::Exception("OutputFile", "writeStart")
        << "Error writing streamer header data to "
        << streamerfile_->fileName() << ".  Possibly the output disk "
        << "is full?" << std::endl;
    }
    streamerfile_->set_run(inview.run()); 
  }
 
  uint32 StreamerOutputFile::writeEOF(uint32 statusCode, 
                                      const std::vector<uint32>& hltStats) 
  {
    EOFRecordBuilder eof(streamerfile_->run(), 
                         streamerfile_->events(),
                         statusCode,
                         hltStats,
                         streamerfile_->first_event_offset(),
                         streamerfile_->last_event_offset());

    bool ret = streamerfile_->write((const char*) 
                                    eof.recAddress(), 
                                    eof.size());  
    if (ret) {
      throw cms::Exception("OutputFile", "writeEOF")
        << "Error writing streamer end-of-file to "
        << streamerfile_->fileName() << ".  Possibly the output disk "
        << "is full?" << std::endl;
    }
    return eof.size();  
  }
