// $Id: EventFileHandler.cc,v 1.19 2012/04/04 12:17:00 mommsen Exp $
/// @file: EventFileHandler.cc

#include <EventFilter/StorageManager/interface/EventFileHandler.h>
#include <EventFilter/StorageManager/interface/Exception.h>
#include <EventFilter/StorageManager/interface/I2OChain.h>
#include <IOPool/Streamer/interface/EventMessage.h>

#include <iostream>

 
namespace stor {
  
  EventFileHandler::EventFileHandler
  (
    InitMsgSharedPtr view,
    FilesMonitorCollection::FileRecordPtr fileRecord,
    const DbFileHandlerPtr dbFileHandler,
    const uint64_t& maxFileSize
  ) :
  FileHandler(fileRecord, dbFileHandler, maxFileSize),
  writer_(new edm::StreamerFileWriter(fileRecord->completeFileName()))
  {
    writeHeader(view);
  }
  
  
  void EventFileHandler::writeHeader(InitMsgSharedPtr view)
  {
    InitMsgView initView(&(*view)[0]);
    writer_->doOutputHeader(initView);
    fileRecord_->fileSize += view->size();
    lastEntry_ = utils::getCurrentTime();
  }
  
  
  void EventFileHandler::do_writeEvent(const I2OChain& event)
  {
    edm::StreamerFileWriterEventParams evtParams;
    
    event.hltTriggerBits(evtParams.hltBits);
    evtParams.headerPtr = (char*) event.headerLocation();
    evtParams.headerSize = event.headerSize();
    
    unsigned int fragCount = event.fragmentCount();
    evtParams.fragmentCount = fragCount;
    
    for (unsigned int idx = 0; idx < fragCount; ++idx)
    {
      evtParams.fragmentIndex = idx;
      evtParams.dataPtr = (char*) event.dataLocation(idx);
      evtParams.dataSize = event.dataSize(idx);
      
      writer_->doOutputEventFragment(evtParams);
    }
  }
  
  
  void EventFileHandler::closeFile(const FilesMonitorCollection::FileRecord::ClosingReason& reason)
  {
    if ( ! fileRecord_->isOpen ) return;
    
    if (writer_)
    {
      // if writer was reset, we already closed the stream but failed to move the file to the closed position
      writer_->stop();
      fileRecord_->fileSize += writer_->getStreamEOFSize();
      fileRecord_->adler32 = writer_->get_adler32();
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
