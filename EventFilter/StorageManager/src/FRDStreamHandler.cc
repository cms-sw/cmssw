// $Id: FRDStreamHandler.cc,v 1.4 2009/08/28 16:41:26 mommsen Exp $
/// @file: FRDStreamHandler.cc

#include "EventFilter/StorageManager/interface/ErrorStreamConfigurationInfo.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/StorageManager/interface/FRDFileHandler.h"
#include "EventFilter/StorageManager/interface/FRDStreamHandler.h"


using namespace stor;


FRDStreamHandler::FRDStreamHandler
(
  const ErrorStreamConfigurationInfo& streamConfig,
  SharedResourcesPtr sharedResources
):
StreamHandler(sharedResources),
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
    new FRDFileHandler(fileRecord, _diskWritingParams, getMaxFileSize())
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
