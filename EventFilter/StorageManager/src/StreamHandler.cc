// $Id: StreamHandler.cc,v 1.25 2012/10/17 10:13:25 mommsen Exp $
/// @file: StreamHandler.cc

#include <sstream>
#include <iomanip>

#include "EventFilter/StorageManager/interface/FileHandler.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/StorageManager/interface/StreamHandler.h"

#include "boost/bind.hpp"


namespace stor {
  
  StreamHandler::StreamHandler(
    const SharedResourcesPtr sharedResources,
    const DbFileHandlerPtr dbFileHandler
  ) :
  sharedResources_(sharedResources),
  statReporter_(sharedResources->statisticsReporter_),
  streamRecord_(statReporter_->getStreamsMonitorCollection().getNewStreamRecord()),
  diskWritingParams_(dbFileHandler->getDiskWritingParams()),
  dbFileHandler_(dbFileHandler)
  {}
  
  
  void StreamHandler::closeAllFiles()
  {
    std::string errorMsg = "Failed to close all files for stream " + streamLabel() + ": ";
    
    for (FileHandlers::const_iterator it = fileHandlers_.begin(),
           itEnd = fileHandlers_.end(); it != itEnd; ++it)
    {
      try
      {
        (*it)->closeFile(FilesMonitorCollection::FileRecord::runEnded);
      }
      catch(xcept::Exception& e)
      {
        XCEPT_DECLARE_NESTED( stor::exception::DiskWriting,
          sentinelException, errorMsg, e );
        sharedResources_->alarmHandler_->
          notifySentinel(AlarmHandler::ERROR, sentinelException);
      }
      catch(std::exception &e)
      {
        errorMsg += e.what();
        XCEPT_DECLARE( stor::exception::DiskWriting,
          sentinelException, errorMsg );
        sharedResources_->alarmHandler_->
          notifySentinel(AlarmHandler::ERROR, sentinelException);
      }
      catch(...)
      {
        errorMsg += "Unknown exception";
        XCEPT_DECLARE( stor::exception::DiskWriting,
          sentinelException, errorMsg );
        sharedResources_->alarmHandler_->
          notifySentinel(AlarmHandler::ERROR, sentinelException);
      }
    }
    fileHandlers_.clear();
  }
  
  
  void StreamHandler::closeTimedOutFiles(utils::TimePoint_t currentTime)
  {
    fileHandlers_.erase(
      std::remove_if(fileHandlers_.begin(),
        fileHandlers_.end(),
        boost::bind(&FileHandler::tooOld,
          _1, currentTime)),
      fileHandlers_.end());
  }
  
  
  bool StreamHandler::closeFilesForLumiSection
  (
    const uint32_t& lumiSection,
    std::string& str
  )
  {
    fileHandlers_.erase(
      std::remove_if(fileHandlers_.begin(),
        fileHandlers_.end(),
        boost::bind(&FileHandler::isFromLumiSection,
          _1, lumiSection)),
      fileHandlers_.end());
    
    return streamRecord_->reportLumiSectionInfo(lumiSection, str);
  }
  
  
  void StreamHandler::writeEvent(const I2OChain& event)
  {
    FileHandlerPtr handler = getFileHandler(event);
    handler->writeEvent(event);
    streamRecord_->addSizeInBytes(event.totalDataSize());
    statReporter_->getThroughputMonitorCollection().
      addDiskWriteSample(event.totalDataSize());
  }
  
  
  StreamHandler::FileHandlerPtr StreamHandler::getFileHandler(const I2OChain& event)
  {
    for (
      FileHandlers::iterator it = fileHandlers_.begin(), itEnd = fileHandlers_.end();
      it != itEnd;
      ++it
    ) 
    {
      if ( (*it)->lumiSection() == event.lumiSection() )
      {
        if ( (*it)->tooLarge(event.totalDataSize()) )
        { 
          fileHandlers_.erase(it);
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
      statReporter_->getFilesMonitorCollection().getNewFileRecord();
    
    try
    {
      fileRecord->runNumber = event.runNumber();
      fileRecord->lumiSection = event.lumiSection();
    }
    catch(stor::exception::IncompleteEventMessage &e)
    {
      fileRecord->runNumber = sharedResources_->configuration_->getRunNumber();
      fileRecord->lumiSection = 0;
    }
    fileRecord->streamLabel = streamLabel();
    fileRecord->baseFilePath =
      getBaseFilePath(fileRecord->runNumber, fileRecord->entryCounter);
    fileRecord->coreFileName =
      getCoreFileName(fileRecord->runNumber, fileRecord->lumiSection);
    fileRecord->fileCounter = getFileCounter(fileRecord->coreFileName);
    fileRecord->whyClosed = FilesMonitorCollection::FileRecord::notClosed;
    fileRecord->isOpen = true;
    
    return fileRecord;
  }
  
  
  std::string StreamHandler::getBaseFilePath
  (
    const uint32_t& runNumber,
    uint32_t fileCount
  ) const
  {
    return diskWritingParams_.filePath_ + getFileSystem(runNumber, fileCount);
  }
  
  
  std::string StreamHandler::getFileSystem
  (
    const uint32_t& runNumber,
    uint32_t fileCount
  ) const
  {
    // if the number of logical disks is not specified, don't
    // add a file system subdir to the path
    if (diskWritingParams_.nLogicalDisk_ <= 0) {return "";}
    
    unsigned int fileSystemNumber =
      (runNumber
        + atoi(diskWritingParams_.smInstanceString_.c_str())
        + fileCount)
      % diskWritingParams_.nLogicalDisk_;
    
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
    coreFileName << diskWritingParams_.setupLabel_
      << "." << std::setfill('0') << std::setw(8) << runNumber
      << "." << std::setfill('0') << std::setw(4) << lumiSection
      << "." << streamLabel()
      << "." << diskWritingParams_.fileName_
      << "." << std::setfill('0') << std::setw(2) << diskWritingParams_.smInstanceString_;
    
    return coreFileName.str();
  }
  
  
  unsigned int StreamHandler::getFileCounter(const std::string& coreFileName)
  {
    CoreFileNamesMap::iterator pos = usedCoreFileNames_.find(coreFileName);
    if (pos == usedCoreFileNames_.end())
    {
      usedCoreFileNames_.insert(pos, std::make_pair(coreFileName, 0));
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
      diskWritingParams_.maxFileSizeMB_ > 0 ? 
      diskWritingParams_.maxFileSizeMB_ : getStreamMaxFileSize();
    
    return ( maxFileSizeMB * 1024 * 1024 );
  }
  
} // namespace stor

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
