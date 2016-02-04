// $Id: FRDStreamHandler.cc,v 1.6 2010/03/19 13:24:05 mommsen Exp $
/// @file: FRDStreamHandler.cc

#include "EventFilter/StorageManager/interface/ErrorStreamConfigurationInfo.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/StorageManager/interface/FRDFileHandler.h"
#include "EventFilter/StorageManager/interface/FRDStreamHandler.h"


using namespace stor;


FRDStreamHandler::FRDStreamHandler
(
  const ErrorStreamConfigurationInfo& streamConfig,
  const SharedResourcesPtr sharedResources,
  const DbFileHandlerPtr dbFileHandler
):
StreamHandler(sharedResources, dbFileHandler),
_streamConfig(streamConfig)
{
  _streamRecord->streamName = streamLabel();
  _streamRecord->fractionToDisk = fractionToDisk();
}


FRDStreamHandler::FileHandlerPtr
FRDStreamHandler::newFileHandler(const I2OChain& event)
{
  FilesMonitorCollection::FileRecordPtr fileRecord = getNewFileRecord(event);

  FileHandlerPtr newFileHandler(
    new FRDFileHandler(fileRecord, _dbFileHandler, _diskWritingParams, getMaxFileSize())
  );
  _fileHandlers.push_back(newFileHandler);

  return newFileHandler;
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
