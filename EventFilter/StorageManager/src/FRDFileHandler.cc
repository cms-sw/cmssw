// $Id: FRDFileHandler.cc,v 1.3 2009/07/03 11:08:46 mommsen Exp $
/// @file: FRDFileHandler.cc

#include <EventFilter/StorageManager/interface/FRDFileHandler.h>
#include <IOPool/Streamer/interface/FRDEventMessage.h>

#include <iostream>
 
using namespace stor;


FRDFileHandler::FRDFileHandler
(
  FilesMonitorCollection::FileRecordPtr fileRecord,
  const DiskWritingParams& dwParams,
  const long long& maxFileSize
) :
FileHandler(fileRecord, dwParams, maxFileSize),
_writer(fileRecord->completeFileName()+".dat")
{}


FRDFileHandler::~FRDFileHandler()
{
  closeFile();
}


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


const bool FRDFileHandler::tooOld(utils::time_point_t currentTime)
{
  if (_diskWritingParams._errorEventsTimeOut > 0 &&
    (currentTime - _lastEntry) > _diskWritingParams._errorEventsTimeOut)
  {
    _closingReason = FilesMonitorCollection::FileRecord::timeout;
    return true;
  }
  else
  {
    return false;
  }
}


void FRDFileHandler::closeFile()
{
  _writer.stop();
  moveFileToClosed(false);
  writeToSummaryCatalog();
  updateDatabase();
}



/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
