#include "IOPool/Streamer/interface/StreamerOutputIndexFile.h"
#include "IOPool/Streamer/interface/EOFRecordBuilder.h"

#include <fstream>

 StreamerOutputIndexFile::StreamerOutputIndexFile(const std::string& name): 
 indexfile_(new OutputFile(name))
  {
   
  }

  StreamerOutputIndexFile::~StreamerOutputIndexFile() 
  {
   //indexfile_.ost()->close();
   delete indexfile_; 
  }

  void StreamerOutputIndexFile::write(const InitMsgBuilder& inview)
  {
    indexfile_->ost()->write((const char*)
                                inview.startAddress(),
                                inview.headerSize() ) ;
    indexfile_->current_offset_ += (uint64) inview.headerSize();
    indexfile_->run_ = inview.run();
    
  }

  void StreamerOutputIndexFile::write(const InitMsgView& inview)
  {
    indexfile_->ost()->write((const char*)
                                inview.startAddress(),
                                inview.headerSize() ) ;
    indexfile_->current_offset_ += (uint64) inview.headerSize();
    indexfile_->run_ = inview.run();

  }

  void StreamerOutputIndexFile::write(const EventMsgBuilder& ineview, 
                                                   uint64 offset)
  {
    /** Write the Event Header */
    indexfile_->ost()->write((const char*)
                             ineview.startAddress(),
                             ineview.headerSize() ) ;
    indexfile_->current_offset_ += (uint64) ineview.headerSize();    

    /** Write the Event Offset */
    uint64 offsetstr;
    convert(offset, (unsigned char*)&offsetstr);
    indexfile_->ost()->write((const char*) &offsetstr, sizeof(uint64) );
    if (indexfile_->events_ == 0) 
       indexfile_->first_event_offset_ = (uint64) offset;
    ++indexfile_->events_; 
    /** Offset of last written event */
    indexfile_->last_event_offset_ = (uint64) offset; 
  }

  void StreamerOutputIndexFile::write(const EventMsgView& ineview,
                                                   uint64 offset)
  {
    /** Write the Event Header */
    indexfile_->ost()->write((const char*)
                             ineview.startAddress(),
                             ineview.headerSize() ) ;
    indexfile_->current_offset_ += (uint64) ineview.headerSize();

    /** Write the Event Offset */
    uint64 offsetstr;
    convert(offset, (unsigned char*)&offsetstr);
    indexfile_->ost()->write((const char*) &offsetstr, sizeof(uint64) );
    if (indexfile_->events_ == 0)
       indexfile_->first_event_offset_ = (uint64) offset;
    ++indexfile_->events_;
    /** Offset of last written event */
    indexfile_->last_event_offset_ = (uint64) offset;
  }


  void StreamerOutputIndexFile::writeIndexFileHeader(uint32 magicNumber, 
                                                     uint64 reserved) 
  {
    //04 Bytes Magic Number
    uint32 magicstr;
    convert(magicNumber, (unsigned char*)&magicstr);
    indexfile_->ost()->write((const char*) &magicstr, sizeof(uint32) );

    //08 Bytes Reserved Value
    uint64 reservedstr;
    convert(reserved, (unsigned char*)&reservedstr);
    indexfile_->ost()->write((const char*) &reservedstr, sizeof(uint64) );
  }


  uint32 StreamerOutputIndexFile::writeEOF(uint32 statusCode,
                                    const std::vector<uint32>& hltStats)
  {

    EOFRecordBuilder eof(indexfile_->run_,
                         indexfile_->events_,
                         statusCode,
                         hltStats,
                         indexfile_->first_event_offset_,
                         indexfile_->last_event_offset_);
    indexfile_->ost()->write((const char*)
                               eof.recAddress(),
                               eof.size() );
    return eof.size();
  }

