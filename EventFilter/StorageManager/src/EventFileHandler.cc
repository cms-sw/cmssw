// $Id: EventFileHandler.cc,v 1.7 2009/09/17 11:03:19 mommsen Exp $
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
  const DiskWritingParams& dwParams,
  const unsigned long long& maxFileSize
) :
FileHandler(fileRecord, dwParams, maxFileSize),
_writer(
  fileRecord->completeFileName()+".dat",
  fileRecord->completeFileName()+".ind"
)
{
  writeHeader(view);
}


void EventFileHandler::writeHeader(InitMsgSharedPtr view)
{
  InitMsgView initView(&(*view)[0]);
  _writer.doOutputHeader(initView);
  _fileRecord->fileSize += view->size();
  _lastEntry = utils::getCurrentTime();
}


void EventFileHandler::writeEvent(const I2OChain& event)
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

      _writer.doOutputEventFragment(evtParams);
    }

  _fileRecord->fileSize += event.totalDataSize();
  ++_fileRecord->eventCount;
  _lastEntry = utils::getCurrentTime();
}


bool EventFileHandler::tooOld(const utils::time_point_t currentTime)
{
  if (_diskWritingParams._lumiSectionTimeOut > 0 && 
    (currentTime - _lastEntry) > _diskWritingParams._lumiSectionTimeOut)
  {
    closeFile(FilesMonitorCollection::FileRecord::timeout);
    return true;
  }
  else
  {
    return false;
  }
}


bool EventFileHandler::isFromLumiSection(const uint32_t lumiSection)
{
  if (lumiSection == _fileRecord->lumiSection)
  {
    closeFile(FilesMonitorCollection::FileRecord::endOfLS);
    return true;
  }
  else
  {
    return false;
  }
}


void EventFileHandler::closeFile(const FilesMonitorCollection::FileRecord::ClosingReason& reason)
{
  _writer.stop();
  _fileRecord->fileSize += _writer.getStreamEOFSize();
  setAdler(_writer.get_adler32_stream(), _writer.get_adler32_index());
  try
  {
    moveFileToClosed(true, reason);
  }
  catch (stor::exception::DiskWriting& e)
  {
    _fileRecord->fileSize -= _writer.getStreamEOFSize();

    XCEPT_RETHROW(stor::exception::DiskWriting, 
      "Failed to close " + _fileRecord->completeFileName(), e);
  }
  writeToSummaryCatalog();
  updateDatabase();
}



/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
