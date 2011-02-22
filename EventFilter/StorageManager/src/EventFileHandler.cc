// $Id: EventFileHandler.cc,v 1.12 2010/03/19 13:24:05 mommsen Exp $
/// @file: EventFileHandler.cc

#include <EventFilter/StorageManager/interface/EventFileHandler.h>
#include <EventFilter/StorageManager/interface/Exception.h>
#include <EventFilter/StorageManager/interface/I2OChain.h>
#include <IOPool/Streamer/interface/EventMessage.h>

#include <iostream>
 
using namespace stor;


EventFileHandler::EventFileHandler
(
  InitMsgSharedPtr view,
  FilesMonitorCollection::FileRecordPtr fileRecord,
  const DbFileHandlerPtr dbFileHandler,
  const DiskWritingParams& dwParams,
  const unsigned long long& maxFileSize
) :
FileHandler(fileRecord, dbFileHandler, dwParams, maxFileSize),
_writer(new edm::StreamerFileWriter(
  fileRecord->completeFileName()+".dat",
  fileRecord->completeFileName()+".ind"
  ))
{
  writeHeader(view);
}


void EventFileHandler::writeHeader(InitMsgSharedPtr view)
{
  InitMsgView initView(&(*view)[0]);
  _writer->doOutputHeader(initView);
  _fileRecord->fileSize += view->size();
  _lastEntry = utils::getCurrentTime();
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

      _writer->doOutputEventFragment(evtParams);
    }
}


void EventFileHandler::closeFile(const FilesMonitorCollection::FileRecord::ClosingReason& reason)
{
  if ( ! _fileRecord->isOpen ) return;

  if (_writer)
  {
    // if writer was reset, we already closed the stream but failed to move the file to the closed position
    _writer->stop();
    _fileRecord->fileSize += _writer->getStreamEOFSize();
    setAdler(_writer->get_adler32_stream(), _writer->get_adler32_index());
    _writer.reset(); // Destruct the writer to flush the file stream
  }
  moveFileToClosed(true, reason);
  writeToSummaryCatalog();
  updateDatabase();
}



/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
