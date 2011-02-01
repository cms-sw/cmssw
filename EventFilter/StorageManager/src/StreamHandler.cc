// $Id: StreamHandler.cc,v 1.19 2010/05/17 15:59:10 mommsen Exp $
/// @file: StreamHandler.cc

#include <sstream>
#include <iomanip>

#include "EventFilter/StorageManager/interface/FileHandler.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/StorageManager/interface/StreamHandler.h"

#include "boost/bind.hpp"


using namespace stor;


StreamHandler::StreamHandler(
  const SharedResourcesPtr sharedResources,
  const DbFileHandlerPtr dbFileHandler
) :
_sharedResources(sharedResources),
_statReporter(sharedResources->_statisticsReporter),
_streamRecord(_statReporter->getStreamsMonitorCollection().getNewStreamRecord()),
_diskWritingParams(sharedResources->_configuration->getDiskWritingParams()),
_dbFileHandler(dbFileHandler)
{}


void StreamHandler::closeAllFiles()
{
  std::string errorMsg = "Failed to close all files for stream " + streamLabel() + ": ";
  
  for (FileHandlers::const_iterator it = _fileHandlers.begin(),
         itEnd = _fileHandlers.end(); it != itEnd; ++it)
  {
    try
    {
      (*it)->closeFile(FilesMonitorCollection::FileRecord::runEnded);
    }
    catch(xcept::Exception& e)
    {
      XCEPT_DECLARE_NESTED( stor::exception::DiskWriting,
        sentinelException, errorMsg, e );
      _statReporter->alarmHandler()->
        notifySentinel(AlarmHandler::ERROR, sentinelException);
    }
    catch(std::exception &e)
    {
      errorMsg += e.what();
      XCEPT_DECLARE( stor::exception::DiskWriting,
        sentinelException, errorMsg );
      _statReporter->alarmHandler()->
        notifySentinel(AlarmHandler::ERROR, sentinelException);
    }
    catch(...)
    {
      errorMsg += "Unknown exception";
      XCEPT_DECLARE( stor::exception::DiskWriting,
        sentinelException, errorMsg );
      _statReporter->alarmHandler()->
        notifySentinel(AlarmHandler::ERROR, sentinelException);
    }
  }
  _fileHandlers.clear();
}


void StreamHandler::closeTimedOutFiles(utils::time_point_t currentTime)
{
  _fileHandlers.erase(std::remove_if(_fileHandlers.begin(),
                                     _fileHandlers.end(),
                                     boost::bind(&FileHandler::tooOld,
                                                 _1, currentTime)),
                      _fileHandlers.end());
}

void StreamHandler::closeFilesForLumiSection
(
  const uint32_t& lumiSection,
  std::string& str
)
{
  _streamRecord->reportLumiSectionInfo(lumiSection, str);
  closeFilesForLumiSection(lumiSection);
}


void StreamHandler::closeFilesForLumiSection(const uint32_t lumiSection)
{
  _fileHandlers.erase(std::remove_if(_fileHandlers.begin(),
                                     _fileHandlers.end(),
                                     boost::bind(&FileHandler::isFromLumiSection,
                                                 _1, lumiSection)),
                      _fileHandlers.end());
}


void StreamHandler::writeEvent(const I2OChain& event)
{
  FileHandlerPtr handler = getFileHandler(event);
  handler->writeEvent(event);
  _streamRecord->addSizeInBytes(event.totalDataSize());
  _statReporter->getThroughputMonitorCollection().addDiskWriteSample(event.totalDataSize());
}


StreamHandler::FileHandlerPtr StreamHandler::getFileHandler(const I2OChain& event)
{
  for (
    FileHandlers::iterator it = _fileHandlers.begin(), itEnd = _fileHandlers.end();
    it != itEnd;
    ++it
  ) 
  {
    if ( (*it)->lumiSection() == event.lumiSection() )
    {
      if ( (*it)->tooLarge(event.totalDataSize()) )
      { 
        _fileHandlers.erase(it);
        break;
      }
      else
      {
        return (*it);
      }
    }
  }    
  return newFileHandler(event);
}


FilesMonitorCollection::FileRecordPtr
StreamHandler::getNewFileRecord(const I2OChain& event)
{
  FilesMonitorCollection::FileRecordPtr fileRecord =
    _statReporter->getFilesMonitorCollection().getNewFileRecord();
  
  try
  {
    fileRecord->runNumber = event.runNumber();
    fileRecord->lumiSection = event.lumiSection();
  }
  catch(stor::exception::IncompleteEventMessage &e)
  {
    fileRecord->runNumber = _sharedResources->_configuration->getRunNumber();
    fileRecord->lumiSection = 0;
  }
  fileRecord->streamLabel = streamLabel();
  fileRecord->baseFilePath = getBaseFilePath(fileRecord->runNumber, fileRecord->entryCounter);
  fileRecord->coreFileName = getCoreFileName(fileRecord->runNumber, fileRecord->lumiSection);
  fileRecord->fileCounter = getFileCounter(fileRecord->coreFileName);
  fileRecord->whyClosed = FilesMonitorCollection::FileRecord::notClosed;
  fileRecord->isOpen = true;

  _streamRecord->incrementFileCount(fileRecord->lumiSection);

  return fileRecord;
}


std::string StreamHandler::getBaseFilePath(const uint32_t& runNumber, uint32_t fileCount) const
{
  return _diskWritingParams._filePath + getFileSystem(runNumber, fileCount);
}


std::string StreamHandler::getFileSystem(const uint32_t& runNumber, uint32_t fileCount) const
{
  // if the number of logical disks is not specified, don't
  // add a file system subdir to the path
  if (_diskWritingParams._nLogicalDisk <= 0) {return "";}

  unsigned int fileSystemNumber =
    (runNumber
     + atoi(_diskWritingParams._smInstanceString.c_str())
     + fileCount)
    % _diskWritingParams._nLogicalDisk;

  std::ostringstream fileSystem;
  fileSystem << "/" << std::setfill('0') << std::setw(2) << fileSystemNumber; 

  return fileSystem.str();
}


std::string StreamHandler::getCoreFileName
(
  const uint32_t& runNumber,
  const uint32_t& lumiSection
) const
{
  std::ostringstream coreFileName;
  coreFileName << _diskWritingParams._setupLabel
    << "." << std::setfill('0') << std::setw(8) << runNumber
    << "." << std::setfill('0') << std::setw(4) << lumiSection
    << "." << streamLabel()
    << "." << _diskWritingParams._fileName
    << "." << std::setfill('0') << std::setw(2) << _diskWritingParams._smInstanceString;

  return coreFileName.str();
}

 

unsigned int StreamHandler::getFileCounter(const std::string& coreFileName)
{
  CoreFileNamesMap::iterator pos = _usedCoreFileNames.find(coreFileName);
  if (pos == _usedCoreFileNames.end())
  {
    _usedCoreFileNames.insert(pos, std::make_pair(coreFileName, 0));
    return 0;
  }
  else
  {
    ++(pos->second);
    return pos->second;
  }
}


unsigned long long StreamHandler::getMaxFileSize() const
{
  const unsigned long long maxFileSizeMB =
    _diskWritingParams._maxFileSizeMB > 0 ? 
    _diskWritingParams._maxFileSizeMB : getStreamMaxFileSize();

  return ( maxFileSizeMB * 1024 * 1024 );
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
