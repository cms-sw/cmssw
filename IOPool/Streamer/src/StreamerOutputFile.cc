#include "IOPool/Streamer/interface/StreamerOutputFile.h"
#include "IOPool/Streamer/interface/EOFRecordBuilder.h"
#include "FWCore/Utilities/interface/Exception.h"

  OutputFile::OutputFile(const std::string& name):
  current_offset_(1), 
  events_(0),
  ost_(new ofstream(name.c_str(), ios_base::binary | ios_base::out)),
  filename_(name)
  {
   if(!ost_->is_open()) {
          throw cms::Exception("OutputFile","OutputFile")
          << "Error Opening Output File: "<<name<<"\n";
   }
  }

  OutputFile::~OutputFile() 
  {
    delete ost_;
  }

  StreamerOutputFile::~StreamerOutputFile() 
  {
    streamerfile_.ost()->close();
  }

  StreamerOutputFile::StreamerOutputFile(const std::string& name):
  streamerfile_(name)
  {
  }

  uint64 StreamerOutputFile::write(EventMsgBuilder& ineview)
  {
    /** Offset where current event starts */
    uint64 offset_to_return = streamerfile_.current_offset_; 

    /** Offset of last written event */
    streamerfile_.last_event_offset_ = streamerfile_.current_offset_; 

    writeEventHeader(ineview);
    streamerfile_.ost()->write((const char*) ineview.eventAddr(), 
                              ineview.size() - ineview.headerSize() );
    streamerfile_.current_offset_ += (uint64)
                                     (ineview.size() 
                                      - ineview.headerSize()) ;
    streamerfile_.events_++;
    if ( ! (streamerfile_.events_ % 100) )
        {
	streamerfile_.ost()->flush();
        }
    return offset_to_return;
  }

  void StreamerOutputFile::writeEventHeader(EventMsgBuilder& ineview)
  {
    streamerfile_.ost()->write((const char*) 
                                ineview.startAddress(), 
                                ineview.headerSize() ) ;
    streamerfile_.current_offset_ += (uint64) ineview.headerSize(); 
  }

  void StreamerOutputFile::write(InitMsgBuilder& inview)
  {
    writeStart(inview);
    streamerfile_.ost()->write((const char*) inview.dataAddress(), 
                              inview.size() - inview.headerSize() );
    streamerfile_.current_offset_ += (uint64)(inview.size() 
                                              - inview.headerSize() );

    /** Offset of first event to be written */
    streamerfile_.first_event_offset_ = streamerfile_.current_offset_; 
  }

  void StreamerOutputFile::writeStart(InitMsgBuilder& inview)
  {
    streamerfile_.ost()->write((const char*) 
                                inview.startAddress(), 
                                inview.headerSize() ) ;
    streamerfile_.current_offset_ += (uint64) inview.headerSize();
    streamerfile_.run_ = inview.run();
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

    EOFRecordBuilder eof(streamerfile_.run_, 
                         streamerfile_.events_,
                         statusCode,
                         hltStats,
                         streamerfile_.first_event_offset_,
                         streamerfile_.last_event_offset_);
    streamerfile_.ost()->write((const char*) 
                               eof.recAddress(), 
                               eof.size() );  
  }



