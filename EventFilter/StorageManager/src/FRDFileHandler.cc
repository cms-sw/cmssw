// $Id: FRDFileHandler.cc,v 1.7 2009/09/17 11:03:19 mommsen Exp $
/// @file: FRDFileHandler.cc

#include <EventFilter/StorageManager/interface/FRDFileHandler.h>
#include <EventFilter/StorageManager/interface/Exception.h>
#include <EventFilter/StorageManager/interface/I2OChain.h>
#include <IOPool/Streamer/interface/FRDEventMessage.h>

#include <iostream>
 
using namespace stor;


FRDFileHandler::FRDFileHandler
(
  FilesMonitorCollection::FileRecordPtr fileRecord,
  const DiskWritingParams& dwParams,
  const unsigned long long& maxFileSize
) :
FileHandler(fileRecord, dwParams, maxFileSize),
_writer(fileRecord->completeFileName()+".dat")
{}


void FRDFileHandler::writeEvent(const I2OChain& chain)
{
  unsigned int fragCount = chain.fragmentCount();
  for (unsigned int idx = 0; idx < fragCount; ++idx)
    {
      _writer.doOutputEventFragment(chain.dataLocation(idx),
                                    chain.dataSize(idx));
    }

  _fileRecord->fileSize += chain.totalDataSize();
  ++_fileRecord->eventCount;
  _lastEntry = utils::getCurrentTime();
}


bool FRDFileHandler::tooOld(const utils::time_point_t currentTime)
{
  if (_diskWritingParams._errorEventsTimeOut > 0 &&
    (currentTime - _lastEntry) > _diskWritingParams._errorEventsTimeOut)
  {
    closeFile(FilesMonitorCollection::FileRecord::timeout);
    return true;
  }
  else
  {
    return false;
  }
}


void FRDFileHandler::closeFile(const FilesMonitorCollection::FileRecord::ClosingReason& reason)
{
  _writer.stop();
  moveFileToClosed(false, reason);
  writeToSummaryCatalog();
  updateDatabase();
}



/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
