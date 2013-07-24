// $Id: FRDFileHandler.cc,v 1.19 2012/04/04 12:17:02 mommsen Exp $
/// @file: FRDFileHandler.cc

#include <EventFilter/StorageManager/interface/FRDFileHandler.h>
#include <EventFilter/StorageManager/interface/Exception.h>
#include <EventFilter/StorageManager/interface/I2OChain.h>
#include <IOPool/Streamer/interface/FRDEventMessage.h>

#include <iostream>


namespace stor {
  
  FRDFileHandler::FRDFileHandler
  (
    FilesMonitorCollection::FileRecordPtr fileRecord,
    const DbFileHandlerPtr dbFileHandler,
    const uint64_t& maxFileSize
  ) :
  FileHandler(fileRecord, dbFileHandler, maxFileSize),
  writer_(new FRDEventFileWriter(fileRecord->completeFileName()))
  {}
  
  
  void FRDFileHandler::do_writeEvent(const I2OChain& chain)
  {
    unsigned int fragCount = chain.fragmentCount();
    for (unsigned int idx = 0; idx < fragCount; ++idx)
    {
      writer_->doOutputEventFragment(
        chain.dataLocation(idx),
        chain.dataSize(idx));
    }
  }
  
  
  void FRDFileHandler::closeFile(const FilesMonitorCollection::FileRecord::ClosingReason& reason)
  {
    if ( ! fileRecord_->isOpen ) return;
    
    if (writer_)
    {
      // if writer was reset, we already closed the stream but failed to move the file to the closed position
      writer_->stop();
      fileRecord_->adler32 = writer_->adler32();
      writer_.reset(); // Destruct the writer to flush the file stream
    }
    moveFileToClosed(reason);
    updateDatabase();
  }
  
} // namespace stor

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
