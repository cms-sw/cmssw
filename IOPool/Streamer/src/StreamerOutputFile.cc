#include "IOPool/Streamer/interface/StreamerOutputFile.h"
#include "IOPool/Streamer/interface/EOFRecordBuilder.h"
#include "FWCore/Utilities/interface/Exception.h"

  OutputFile::OutputFile(const std::string& name):
  current_offset_(1), 
  events_(0),
  ost_(new std::ofstream(name.c_str(), std::ios_base::binary | std::ios_base::out)),
  filename_(name)
  {
   if(!ost_->is_open()) {
          throw cms::Exception("OutputFile","OutputFile")
          << "Error Opening Output File: "<<name<<"\n";
   }
  }

  OutputFile::~OutputFile() 
  {
    ost_->close();
    delete ost_;
  }

  StreamerOutputFile::~StreamerOutputFile() 
  {
    delete streamerfile_;
    //streamerfile_.ost()->close();
  }

  StreamerOutputFile::StreamerOutputFile(const std::string& name):
  streamerfile_(new OutputFile(name))
  {
  }

  uint64 StreamerOutputFile::write(const EventMsgBuilder& ineview)
  {
    EventMsgView tmpView(ineview.startAddress());
    return write(tmpView);
  }

  uint64 StreamerOutputFile::write(const EventMsgView& ineview)
  {
    /** Offset where current event starts */
    uint64 offset_to_return = streamerfile_->current_offset_;

    /** Offset of last written event */
    streamerfile_->last_event_offset_ = streamerfile_->current_offset_;

    writeEventHeader(ineview);
    streamerfile_->ost()->write((const char*) ineview.eventData(),
                                ineview.size() - ineview.headerSize());
    if (streamerfile_->ost()->fail()) {
      throw cms::Exception("OutputFile", "write(EventMsgView)")
        << "Error writing streamer event data to "
        << streamerfile_->fileName() << ".  Possibly the output disk "
        << "is full?" << std::endl;
    }
    streamerfile_->current_offset_ += (uint64) (ineview.size()
                                                - ineview.headerSize());
    ++streamerfile_->events_;
    if (!(streamerfile_->events_ % 100)) {
      streamerfile_->ost()->flush();
      if (streamerfile_->ost()->fail()) {
        throw cms::Exception("OutputFile", "write(EventMsgView)")
          << "Error writing streamer event data to "
          << streamerfile_->fileName() << ".  Possibly the output disk "
          << "is full?" << std::endl;
      }
    }
    return offset_to_return;
  }

  void StreamerOutputFile::writeEventHeader(const EventMsgView& ineview)
  {
    streamerfile_->ost()->write((const char*)
                                ineview.startAddress(),
                                ineview.headerSize()) ;
    if (streamerfile_->ost()->fail()) {
      throw cms::Exception("OutputFile", "writeEventHeader")
        << "Error writing streamer event data to "
        << streamerfile_->fileName() << ".  Possibly the output disk "
        << "is full?" << std::endl;
    }
    streamerfile_->current_offset_ += (uint64) ineview.headerSize();
  }

  void StreamerOutputFile::write(const InitMsgBuilder& inview)
  {
    InitMsgView tmpView(inview.startAddress());
    return write(tmpView);
  }

  void StreamerOutputFile::write(const InitMsgView& inview)
  {
    writeStart(inview);
    streamerfile_->ost()->write((const char*) inview.descData(),
                                inview.size() - inview.headerSize());
    if (streamerfile_->ost()->fail()) {
      throw cms::Exception("OutputFile", "write(InitMsgView)")
        << "Error writing streamer header data to "
        << streamerfile_->fileName() << ".  Possibly the output disk "
        << "is full?" << std::endl;
    }
    streamerfile_->current_offset_ += (uint64)(inview.size()
                                               - inview.headerSize());

    /** Offset of first event to be written */
    streamerfile_->first_event_offset_ = streamerfile_->current_offset_;
  }

  void StreamerOutputFile::writeStart(const InitMsgView& inview)
  {
    streamerfile_->ost()->write((const char*)
                                inview.startAddress(),
                                inview.headerSize()) ;
    if (streamerfile_->ost()->fail()) {
      throw cms::Exception("OutputFile", "writeStart")
        << "Error writing streamer header data to "
        << streamerfile_->fileName() << ".  Possibly the output disk "
        << "is full?" << std::endl;
    }
    streamerfile_->current_offset_ += (uint64) inview.headerSize();
    streamerfile_->run_ = inview.run(); 
  }
 
  uint32 StreamerOutputFile::writeEOF(uint32 statusCode, 
                                      const std::vector<uint32>& hltStats) 
  {
    EOFRecordBuilder eof(streamerfile_->run_, 
                         streamerfile_->events_,
                         statusCode,
                         hltStats,
                         streamerfile_->first_event_offset_,
                         streamerfile_->last_event_offset_);

    streamerfile_->ost()->write((const char*) 
                               eof.recAddress(), 
                               eof.size());  
    if (streamerfile_->ost()->fail()) {
      throw cms::Exception("OutputFile", "writeEOF")
        << "Error writing streamer end-of-file to "
        << streamerfile_->fileName() << ".  Possibly the output disk "
        << "is full?" << std::endl;
    }
    return eof.size();  
  }

