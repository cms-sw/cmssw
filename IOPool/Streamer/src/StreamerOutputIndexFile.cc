#include "IOPool/Streamer/interface/StreamerFileIO.h"
#include "IOPool/Streamer/interface/EOFRecordBuilder.h"


 StreamerOutputIndexFile::StreamerOutputIndexFile(const string& name): 
 indexfile_(name)
  {
  
  }

  StreamerOutputIndexFile::~StreamerOutputIndexFile() 
  {
   indexfile_.ost()->close();
  }

  void StreamerOutputIndexFile::write(InitMsgBuilder& inview)
  {
    indexfile_.ost()->write((const char*)
                                inview.startAddress(),
                                inview.headerSize() ) ;
    indexfile_.current_offset_ += (uint64) inview.headerSize();
    indexfile_.run_ = inview.run();
    
  }

  void StreamerOutputIndexFile::write(EventMsgBuilder& ineview, 
                                                   long long offset)
  {
    /** Write the Event Header */
    indexfile_.ost()->write((const char*)
                             ineview.startAddress(),
                             ineview.headerSize() ) ;
    indexfile_.current_offset_ += (uint64) ineview.headerSize();    

    /** Write the Event Offset */
    uint64 offsetstr;
    convert((uint64)offset, (unsigned char*)&offsetstr);
    indexfile_.ost()->write((const char*) &offsetstr, sizeof(uint64) );
    if (indexfile_.events_ == 0) 
       indexfile_.first_event_offset_ = (uint64) offset;
    indexfile_.events_++; 
    /** Offset of last written event */
    indexfile_.last_event_offset_ = (uint64) offset; 
  }


  void StreamerOutputIndexFile::writeIndexFileHeader(uint32 magicNumber, 
                                                     uint64 reserved) 
  {
    //04 Bytes Magic Number
    uint32 magicstr;
    convert((uint32)magicNumber, (unsigned char*)&magicstr);
    indexfile_.ost()->write((const char*) &magicstr, sizeof(uint32) );

    //08 Bytes Reserved Value
    uint64 reservedstr;
    convert((uint64)reserved, (unsigned char*)&reservedstr);
    indexfile_.ost()->write((const char*) &reservedstr, sizeof(uint64) );
  }


  void StreamerOutputIndexFile::writeEOF(uint32 statusCode,
                                    std::vector<uint32>& hltStats)
  {
    /*uint32 dummyStatusCode = 1234;
    std::vector<uint32> hltStats;

    hltStats.push_back(32);
    hltStats.push_back(32);
    hltStats.push_back(32);

    */

    EOFRecordBuilder eof(indexfile_.run_,
                         indexfile_.events_,
                         statusCode,
                         hltStats,
                         indexfile_.first_event_offset_,
                         indexfile_.last_event_offset_);
    indexfile_.ost()->write((const char*)
                               eof.recAddress(),
                               eof.size() );
  }

