// $Id: FilesMonitorCollection.cc,v 1.11 2010/01/29 15:45:47 mommsen Exp $
/// @file: FilesMonitorCollection.cc

#include <string>
#include <sstream>
#include <iomanip>

#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/FilesMonitorCollection.h"

using namespace stor;

FilesMonitorCollection::FilesMonitorCollection(const utils::duration_t& updateInterval) :
MonitorCollection(updateInterval),
_maxFileEntries(250),
_entryCounter(0)
{
  boost::mutex::scoped_lock sl(_fileRecordsMutex);
  _fileRecords.set_capacity(_maxFileEntries);
}


const FilesMonitorCollection::FileRecordPtr
FilesMonitorCollection::getNewFileRecord()
{
  boost::mutex::scoped_lock sl(_fileRecordsMutex);

  boost::shared_ptr<FileRecord> fileRecord(new FilesMonitorCollection::FileRecord());
  fileRecord->entryCounter = _entryCounter++;
  fileRecord->fileSize = 0;
  fileRecord->eventCount = 0;
  _fileRecords.push_back(fileRecord);
  return fileRecord;
}

void FilesMonitorCollection::getFileRecords(FileRecordList& fileRecords) const
{
  boost::mutex::scoped_lock sl(_fileRecordsMutex);
  fileRecords = _fileRecords;
}


void FilesMonitorCollection::do_calculateStatistics()
{
  // nothing to do
}


void FilesMonitorCollection::do_reset()
{
  boost::mutex::scoped_lock sl(_fileRecordsMutex);
  _fileRecords.clear();
  _entryCounter = 0;
}


void FilesMonitorCollection::do_appendInfoSpaceItems(InfoSpaceItems& infoSpaceItems)
{
  infoSpaceItems.push_back(std::make_pair("openFiles", &_openFiles));
  infoSpaceItems.push_back(std::make_pair("closedFiles", &_closedFiles));
}


void FilesMonitorCollection::do_updateInfoSpaceItems()
{
  boost::mutex::scoped_lock sl(_fileRecordsMutex);

  _openFiles = 0;
  
  for (
    FileRecordList::const_iterator it = _fileRecords.begin(),
      itEnd = _fileRecords.end();
    it != itEnd;
    ++it
  )
  {
    if ( (*it)->isOpen )
      ++_openFiles;
  }

  _closedFiles = _entryCounter - _openFiles;
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
    << "." << std::setfill('0') << std::setw(4) << fileCounter;
  return fileName.str();
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
