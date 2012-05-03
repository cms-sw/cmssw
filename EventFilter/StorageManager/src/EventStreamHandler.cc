// $Id: EventStreamHandler.cc,v 1.10 2012/04/04 12:17:01 mommsen Exp $
/// @file: EventStreamHandler.cc

#include "EventFilter/StorageManager/interface/Configuration.h"
#include "EventFilter/StorageManager/interface/EventFileHandler.h"
#include "EventFilter/StorageManager/interface/EventStreamConfigurationInfo.h"
#include "EventFilter/StorageManager/interface/EventStreamHandler.h"


namespace stor {

  EventStreamHandler::EventStreamHandler
  (
    const EventStreamConfigurationInfo& streamConfig,
    const SharedResourcesPtr sharedResources,
    const DbFileHandlerPtr dbFileHandler
  ):
  StreamHandler(sharedResources, dbFileHandler),
  streamConfig_(streamConfig),
  initMsgCollection_(sharedResources->initMsgCollection_)
  {
    streamRecord_->streamName = streamLabel();
    streamRecord_->outputModuleLabel = streamConfig_.outputModuleLabel();
    streamRecord_->fractionToDisk = fractionToDisk();
  }
  
  
  StreamHandler::FileHandlerPtr
  EventStreamHandler::newFileHandler(const I2OChain& event)
  {
    // the INIT message is not available when the EventStreamHandler is
    // constructed, so we need to fetch it when we first need a new file
    // handler (when the first event is received, which is after the 
    // INIT messages have been received)
    if (initMsgView_.get() == 0)
    {
      initMsgView_ = initMsgCollection_->getElementForOutputModuleId( event.outputModuleId() );
    }
    
    FilesMonitorCollection::FileRecordPtr fileRecord = getNewFileRecord(event);
    
    FileHandlerPtr newFileHandler(
      new EventFileHandler(initMsgView_, fileRecord, dbFileHandler_, getMaxFileSize())
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
