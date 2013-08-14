// $Id: FaultyEventStreamHandler.cc,v 1.6 2012/10/17 10:13:25 mommsen Exp $
/// @file: FaultyEventStreamHandler.cc

#include "FWCore/Utilities/interface/Exception.h"

#include "EventFilter/StorageManager/interface/Configuration.h"
#include "EventFilter/StorageManager/interface/EventFileHandler.h"
#include "EventFilter/StorageManager/interface/EventStreamConfigurationInfo.h"
#include "EventFilter/StorageManager/interface/FaultyEventStreamHandler.h"
#include "EventFilter/StorageManager/interface/FRDFileHandler.h"


namespace stor {
  
  FaultyEventStreamHandler::FaultyEventStreamHandler
  (
    const SharedResourcesPtr sharedResources,
    const DbFileHandlerPtr dbFileHandler,
    const std::string& streamName
  ):
  StreamHandler(sharedResources, dbFileHandler),
  initMsgCollection_(sharedResources->initMsgCollection_)
  {
    streamRecord_->streamName = streamName;
    streamRecord_->fractionToDisk = 1;
  }
  
  
  StreamHandler::FileHandlerPtr
  FaultyEventStreamHandler::getFileHandler(const I2OChain& event)
  {
    // In the faulty event stream a new file is opened for each event.
    closeAllFiles();
    return newFileHandler(event);
  }
  
  
  StreamHandler::FileHandlerPtr
  FaultyEventStreamHandler::newFileHandler(const I2OChain& event)
  {
    FileHandlerPtr newFileHandler;
    FilesMonitorCollection::FileRecordPtr fileRecord = getNewFileRecord(event);
    
    // As each event can have a different outputModuleId, we need to 
    // determine the init msg for each event.
    // If this is not possible, use the FRD data format.
    try
    {
      InitMsgSharedPtr initMsgView =
        initMsgCollection_->getElementForOutputModuleId( event.outputModuleId() );
      
      newFileHandler.reset(
        new EventFileHandler(initMsgView, fileRecord, dbFileHandler_, 0)
      );
    }
    catch (stor::exception::IncompleteEventMessage& e) //faulty data event
    {
      newFileHandler.reset(
        new FRDFileHandler(fileRecord, dbFileHandler_, 0)
      );
    }
    catch (stor::exception::WrongI2OMessageType& e) //faulty error event
    {
      newFileHandler.reset(
        new FRDFileHandler(fileRecord, dbFileHandler_, 0)
      );
    }
    
    fileHandlers_.push_back(newFileHandler);
    
    streamRecord_->incrementFileCount(fileRecord->lumiSection);
    
    return newFileHandler;
  }
  
} // namespace stor

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
