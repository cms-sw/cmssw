#include "IOPool/Streamer/interface/StreamerOutputIndexFile.h"
#include "IOPool/Streamer/interface/EOFRecordBuilder.h"

#include <fstream>

 StreamerOutputIndexFile::StreamerOutputIndexFile(const std::string& name): 
 indexfile_(new OutputFile(name))
  {
    indexfile_->set_do_adler(1);
  }

  StreamerOutputIndexFile::~StreamerOutputIndexFile() 
  {
  }

  void StreamerOutputIndexFile::write(const InitMsgBuilder& inview)
  {
    indexfile_->write((const char*)
                      inview.startAddress(),
                      inview.headerSize() ) ;
    indexfile_->set_run(inview.run());
  }

  void StreamerOutputIndexFile::write(const InitMsgView& inview)
  {
    indexfile_->write((const char*)
                      inview.startAddress(),
                      inview.headerSize() ) ;
    indexfile_->set_run(inview.run());
  }

  void StreamerOutputIndexFile::
  writeInit(uint32 runNumber, const char *headerPtr, uint32 headerSize)
  {
    indexfile_->write(headerPtr, headerSize);
    indexfile_->set_run(runNumber);
  }

  void StreamerOutputIndexFile::write(const EventMsgBuilder& ineview, 
                                      uint64 offset)
  {
    /** Write the Event Header */
    indexfile_->write((const char*)
                      ineview.startAddress(),
                      ineview.headerSize() ) ;

    /** Write the Event Offset */
    uint64 offsetstr;
    convert(offset, (unsigned char*)&offsetstr);
    indexfile_->write((const char*) &offsetstr, sizeof(uint64) );
    if (indexfile_->events() == 0) 
      indexfile_->set_first_event_offset(offset);
    indexfile_->inc_events(); 
    /** Offset of last written event */
    indexfile_->set_last_event_offset(offset);
  }

  void StreamerOutputIndexFile::write(const EventMsgView& ineview,
                                      uint64 offset)
  {
    /** Write the Event Header */
    indexfile_->write((const char*)
                      ineview.startAddress(),
                      ineview.headerSize() ) ;

    /** Write the Event Offset */
    uint64 offsetstr;
    convert(offset, (unsigned char*)&offsetstr);
    indexfile_->write((const char*) &offsetstr, sizeof(uint64) );
    if (indexfile_->events() == 0)
      indexfile_->set_first_event_offset(offset);
    indexfile_->inc_events();
    /** Offset of last written event */
    indexfile_->set_last_event_offset(offset);
  }

  void StreamerOutputIndexFile::
  writeEvent(const char *headerPtr, uint32 headerSize, uint64 offset)
  {
    /** Write the Event Header */
    indexfile_->write(headerPtr, headerSize) ;

    /** Write the Event Offset */
    uint64 offsetstr;
    convert(offset, (unsigned char*)&offsetstr);
    indexfile_->write((const char*) &offsetstr, sizeof(uint64) );
    if (indexfile_->events() == 0)
      indexfile_->set_first_event_offset(offset);
    indexfile_->inc_events();
    /** Offset of last written event */
    indexfile_->set_last_event_offset(offset);
  }

  void StreamerOutputIndexFile::writeIndexFileHeader(uint32 magicNumber, 
                                                     uint64 reserved) 
  {
    //04 Bytes Magic Number
    uint32 magicstr;
    convert(magicNumber, (unsigned char*)&magicstr);
    indexfile_->write((const char*) &magicstr, sizeof(uint32) );

    //08 Bytes Reserved Value
    uint64 reservedstr;
    convert(reserved, (unsigned char*)&reservedstr);
    indexfile_->write((const char*) &reservedstr, sizeof(uint64) );
  }


  uint32 StreamerOutputIndexFile::writeEOF(uint32 statusCode,
                                           const std::vector<uint32>& hltStats)
  {

    EOFRecordBuilder eof(indexfile_->run(),
                         indexfile_->events(),
                         statusCode,
                         hltStats,
                         indexfile_->first_event_offset(),
                         indexfile_->last_event_offset());
    indexfile_->write((const char*)
                      eof.recAddress(),
                      eof.size() );
    return eof.size();
  }

