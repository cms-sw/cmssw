// $Id: FaultyEventStreamHandler.cc,v 1.1 2010/05/11 18:02:30 mommsen Exp $
/// @file: FaultyEventStreamHandler.cc

#include "FWCore/Utilities/interface/Exception.h"

#include "EventFilter/StorageManager/interface/Configuration.h"
#include "EventFilter/StorageManager/interface/EventFileHandler.h"
#include "EventFilter/StorageManager/interface/EventStreamConfigurationInfo.h"
#include "EventFilter/StorageManager/interface/FaultyEventStreamHandler.h"
#include "EventFilter/StorageManager/interface/FRDFileHandler.h"


using namespace stor;


FaultyEventStreamHandler::FaultyEventStreamHandler
(
  const SharedResourcesPtr sharedResources,
  const DbFileHandlerPtr dbFileHandler,
  const std::string& streamName
):
StreamHandler(sharedResources, dbFileHandler),
_initMsgCollection(sharedResources->_initMsgCollection)
{
  _streamRecord->streamName = streamName;
  _streamRecord->fractionToDisk = 1;
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
      _initMsgCollection->getElementForOutputModule(
        _initMsgCollection->getOutputModuleName( event.outputModuleId() )
      );
    
    newFileHandler.reset(
      new EventFileHandler(initMsgView, fileRecord, _dbFileHandler,
        _diskWritingParams, 0)
    );
  }
  catch (stor::exception::IncompleteEventMessage& e) //faulty data event
  {
    newFileHandler.reset(
      new FRDFileHandler(fileRecord, _dbFileHandler, _diskWritingParams, 0)
    );
  }
  catch (stor::exception::WrongI2OMessageType& e) //faulty error event
  {
    newFileHandler.reset(
      new FRDFileHandler(fileRecord, _dbFileHandler, _diskWritingParams, 0)
    );
  }

  _fileHandlers.push_back(newFileHandler);

  return newFileHandler;
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
