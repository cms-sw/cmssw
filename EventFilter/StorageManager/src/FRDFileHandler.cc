// $Id: FRDFileHandler.cc,v 1.14 2010/09/28 16:25:29 mommsen Exp $
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
  const DbFileHandlerPtr dbFileHandler,
  const DiskWritingParams& dwParams,
  const uint64_t& maxFileSize
) :
FileHandler(fileRecord, dbFileHandler, dwParams, maxFileSize),
_writer(new FRDEventFileWriter(fileRecord->completeFileName()))
{}


void FRDFileHandler::do_writeEvent(const I2OChain& chain)
{
  unsigned int fragCount = chain.fragmentCount();
  for (unsigned int idx = 0; idx < fragCount; ++idx)
    {
      _writer->doOutputEventFragment(chain.dataLocation(idx),
                                     chain.dataSize(idx));
    }
}


void FRDFileHandler::closeFile(const FilesMonitorCollection::FileRecord::ClosingReason& reason)
{
  if ( ! _fileRecord->isOpen ) return;

  if (_writer)
  {
    // if writer was reset, we already closed the stream but failed to move the file to the closed position
    _writer->stop();
    setAdler(_writer->adler32());
    _writer.reset(); // Destruct the writer to flush the file stream
  }
  moveFileToClosed(reason);
  writeToSummaryCatalog();
  updateDatabase();
}



/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
