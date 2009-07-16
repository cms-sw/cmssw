// $Id: StreamHandler.cc,v 1.4 2009/06/18 16:02:01 biery Exp $

#include <sstream>
#include <iomanip>

#include "EventFilter/StorageManager/interface/StreamHandler.h"
#include "EventFilter/StorageManager/interface/FilesMonitorCollection.h"
#include "EventFilter/StorageManager/interface/StreamsMonitorCollection.h"

#include "boost/bind.hpp"


using namespace stor;


StreamHandler::StreamHandler(SharedResourcesPtr sharedResources) :
_statReporter(sharedResources->_statisticsReporter),
_streamRecord(_statReporter->getStreamsMonitorCollection().getNewStreamRecord()),
_diskWritingParams(sharedResources->_configuration->getDiskWritingParams())
{}


StreamHandler::~StreamHandler()
{
  closeAllFiles();
}


void StreamHandler::closeAllFiles()
{
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


void StreamHandler::writeEvent(const I2OChain& event)
{
  FileHandlerPtr handler = getFileHandler(event);
  handler->writeEvent(event);
  _streamRecord->addSizeInBytes(event.totalDataSize());
}


const StreamHandler::FileHandlerPtr StreamHandler::getFileHandler(const I2OChain& event)
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


const FilesMonitorCollection::FileRecordPtr
StreamHandler::getNewFileRecord(const I2OChain& event)
{
  FilesMonitorCollection::FileRecordPtr fileRecord =
    _statReporter->getFilesMonitorCollection().getNewFileRecord();
  
  fileRecord->runNumber = event.runNumber();
  fileRecord->lumiSection = event.lumiSection();
  fileRecord->streamLabel = streamLabel();
  fileRecord->baseFilePath = getBaseFilePath(event.runNumber(), fileRecord->entryCounter);
  fileRecord->coreFileName = getCoreFileName(event.runNumber(), event.lumiSection());
  fileRecord->fileCounter = getFileCounter(fileRecord->coreFileName);
  fileRecord->whyClosed = FilesMonitorCollection::FileRecord::notClosed;

  _streamRecord->incrementFileCount();

  return fileRecord;
}


const std::string StreamHandler::getBaseFilePath(const uint32& runNumber, uint32_t fileCount) const
{
  return _diskWritingParams._filePath + getFileSystem(runNumber, fileCount);
}


const std::string StreamHandler::getFileSystem(const uint32& runNumber, uint32_t fileCount) const
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


const std::string StreamHandler::getCoreFileName
(
  const uint32& runNumber,
  const uint32& lumiSection
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

 

const unsigned int StreamHandler::getFileCounter(const std::string& coreFileName)
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



const long long StreamHandler::getMaxFileSize() const
{
  int maxFileSizeMB = _diskWritingParams._maxFileSizeMB > 0 ? 
    _diskWritingParams._maxFileSizeMB : getStreamMaxFileSize();

  return ( maxFileSizeMB * 1024 * 1024 );
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
