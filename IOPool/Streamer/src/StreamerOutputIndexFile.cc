#include "IOPool/Streamer/interface/StreamerFileIO.h"

 StreamerOutputIndexFile::StreamerOutputIndexFile(const string& name): 
 StreamerOutputFile(name)
 
  {
  
  }

  void StreamerOutputIndexFile::write(InitMsgBuilder& inview)
  {
    writeStart(inview);
  }

  void StreamerOutputIndexFile::write(EventMsgBuilder& ineview, 
                                                   long long offset)
  {
    
    writeEventHeader(ineview);
    uint64 offsetstr;
    convert((uint64)offset, (unsigned char*)&offsetstr);
    ost_->write((const char*) &offsetstr, sizeof(uint64) );
    if (events_ == 0) first_event_offset_ = (uint64) offset;
    events_++; 
    last_event_offset_ = (uint64) offset; /** Offset of last written event */
  }


  void StreamerOutputIndexFile::writeIndexFileHeader(uint32 magicNumber, 
                                                     uint64 reserved) 
  {
    //04 Bytes Magic Number
    uint32 magicstr;
    convert((uint32)magicNumber, (unsigned char*)&magicstr);
    ost_->write((const char*) &magicstr, sizeof(uint32) );

    //08 Bytes Reserved Value
    uint64 reservedstr;
    convert((uint64)reserved, (unsigned char*)&reservedstr);
    ost_->write((const char*) &reservedstr, sizeof(uint64) );
  }

