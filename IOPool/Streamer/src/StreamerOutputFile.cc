#include "IOPool/Streamer/interface/StreamerFileIO.h"
#include "IOPool/Streamer/interface/EOFRecordBuilder.h"

  StreamerOutputFile::~StreamerOutputFile() 
  {
    /** For the time being writeEOF is invoked for user here
        if its a "user operation, then we canb expose it to 
        user and call it explicitly 
    writeEOF();
    */   

    ost_->close();
    delete ost_;
  }

  StreamerOutputFile::StreamerOutputFile(const string& name):
  filename_(name),
  ost_(new ofstream(name.c_str(), ios_base::binary | ios_base::out)),
  //ost_(makeOutputFile(filename_)),
  current_offset_(1),  //0 or 1 ???????
  events_(0)
  {
  }

  uint64 StreamerOutputFile::write(EventMsgBuilder& ineview)
  {
    uint64 offset_to_return = current_offset_; /** Offset where current event starts */
    last_event_offset_ = current_offset_; /** Offset of last written event */

    writeEventHeader(ineview);
    ost_->write((const char*) ineview.eventAddr(), ineview.size() - ineview.headerSize() );
    current_offset_ += (uint64)(ineview.size() - ineview.headerSize()) ;
    events_++;
    if ( ! (events_ % 100) )
        {
	ost_->flush();
        }
    return offset_to_return;
  }

  void StreamerOutputFile::writeEventHeader(EventMsgBuilder& ineview)
  {
    ost_->write((const char*) ineview.startAddress(), ineview.headerSize() ) ;
    current_offset_ += (uint64) ineview.headerSize(); 
  }

  void StreamerOutputFile::write(InitMsgBuilder& inview)
  {
    writeStart(inview);
    ost_->write((const char*) inview.dataAddress(), inview.size() - inview.headerSize() );
    current_offset_ += (uint64)(inview.size() - inview.headerSize() );
    first_event_offset_ = current_offset_; /** Offset of first event to be written */
  }

  void StreamerOutputFile::writeStart(InitMsgBuilder& inview)
  {
    ost_->write((const char*) inview.startAddress(), inview.headerSize() ) ;
    current_offset_ += (uint64) inview.headerSize();
    run_ = inview.run();
  }

  void StreamerOutputFile::writeEOF(uint32 statusCode, 
                                    std::vector<uint32>& hltStats) 
  {
    /*uint32 dummyStatusCode = 1234;
    std::vector<uint32> hltStats;

    hltStats.push_back(32);
    hltStats.push_back(32);
    hltStats.push_back(32);

    */

    EOFRecordBuilder eof(run_, events_,
                 statusCode,
                 hltStats,
                 first_event_offset_,
                 last_event_offset_);
    ost_->write((const char*) eof.recAddress(), eof.size() );  
  }



