// $Id: FilesMonitorCollection.cc,v 1.15 2011/07/07 09:22:45 mommsen Exp $
/// @file: FilesMonitorCollection.cc

#include <string>
#include <sstream>
#include <iomanip>

#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/FilesMonitorCollection.h"


namespace stor {
  
  FilesMonitorCollection::FilesMonitorCollection(const utils::Duration_t& updateInterval) :
  MonitorCollection(updateInterval),
  maxFileEntries_(250),
  entryCounter_(0)
  {
    boost::mutex::scoped_lock sl(fileRecordsMutex_);
    fileRecords_.set_capacity(maxFileEntries_);
  }
  
  
  const FilesMonitorCollection::FileRecordPtr
  FilesMonitorCollection::getNewFileRecord()
  {
    boost::mutex::scoped_lock sl(fileRecordsMutex_);
    
    boost::shared_ptr<FileRecord> fileRecord(new FilesMonitorCollection::FileRecord());
    fileRecord->entryCounter = entryCounter_++;
    fileRecord->fileSize = 0;
    fileRecord->eventCount = 0;
    fileRecord->adler32 = 0;
    fileRecords_.push_back(fileRecord);
    return fileRecord;
  }
  
  void FilesMonitorCollection::getFileRecords(FileRecordList& fileRecords) const
  {
    boost::mutex::scoped_lock sl(fileRecordsMutex_);
    fileRecords = fileRecords_;
  }
  
  
  void FilesMonitorCollection::do_calculateStatistics()
  {
    // nothing to do
  }
  
  
  void FilesMonitorCollection::do_reset()
  {
    boost::mutex::scoped_lock sl(fileRecordsMutex_);
    fileRecords_.clear();
    entryCounter_ = 0;
  }
  
  
  void FilesMonitorCollection::do_appendInfoSpaceItems(InfoSpaceItems& infoSpaceItems)
  {
    infoSpaceItems.push_back(std::make_pair("openFiles", &openFiles_));
    infoSpaceItems.push_back(std::make_pair("closedFiles", &closedFiles_));
  }
  
  
  void FilesMonitorCollection::do_updateInfoSpaceItems()
  {
    boost::mutex::scoped_lock sl(fileRecordsMutex_);
    
    openFiles_ = 0;
    
    for (
      FileRecordList::const_iterator it = fileRecords_.begin(),
        itEnd = fileRecords_.end();
      it != itEnd;
      ++it
    )
    {
      if ( (*it)->isOpen )
        ++openFiles_;
    }
    
    closedFiles_ = entryCounter_ - openFiles_;
  }
  
  
  std::string FilesMonitorCollection::FileRecord::closingReason()
  {
    switch (whyClosed)
    {
      case notClosed:   return "open";
      case runEnded:    return "run ended";
      case LSended:     return "LS ended";
      case timeout:     return "timeout";
      case size:        return "file size";
      case truncated:   return "TRUNCATED";
      case inaccessible:return "INACCESSIBLE";
      default:          return "unknown";
    }
  }
  
  
  std::string FilesMonitorCollection::FileRecord::filePath(FileStatus status)
  {
    switch (status)
    {
      case open:    return ( baseFilePath + "/open/" );
      case closed:  return ( baseFilePath + "/closed/" );
      case current: return ( baseFilePath + (isOpen ? "/open/" : "/closed/") );
    }
    return "";
  }
  
  
  std::string FilesMonitorCollection::FileRecord::fileName()
  {
    std::ostringstream fileName;
    fileName << coreFileName 
      << "." << std::setfill('0') << std::setw(4) << fileCounter
      << ".dat";
    return fileName.str();
  }
  
} // namespace stor

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
