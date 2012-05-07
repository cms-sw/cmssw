// $Id: FRDStreamHandler.cc,v 1.6.10.1 2011/03/07 11:33:05 mommsen Exp $
/// @file: FRDStreamHandler.cc

#include "EventFilter/StorageManager/interface/ErrorStreamConfigurationInfo.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/StorageManager/interface/FRDFileHandler.h"
#include "EventFilter/StorageManager/interface/FRDStreamHandler.h"


namespace stor {
  
  FRDStreamHandler::FRDStreamHandler
  (
    const ErrorStreamConfigurationInfo& streamConfig,
    const SharedResourcesPtr sharedResources,
    const DbFileHandlerPtr dbFileHandler
  ):
  StreamHandler(sharedResources, dbFileHandler),
  streamConfig_(streamConfig)
  {
    streamRecord_->streamName = streamLabel();
    streamRecord_->fractionToDisk = fractionToDisk();
  }
  
  
  FRDStreamHandler::FileHandlerPtr
  FRDStreamHandler::newFileHandler(const I2OChain& event)
  {
    FilesMonitorCollection::FileRecordPtr fileRecord = getNewFileRecord(event);
    
    FileHandlerPtr newFileHandler(
      new FRDFileHandler(fileRecord, dbFileHandler_, diskWritingParams_, getMaxFileSize())
    );
    fileHandlers_.push_back(newFileHandler);
    
    return newFileHandler;
  }

} // namespace stor

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
