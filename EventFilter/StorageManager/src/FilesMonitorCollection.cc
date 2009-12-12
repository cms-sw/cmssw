// $Id: FilesMonitorCollection.cc,v 1.9 2009/09/18 15:37:44 mommsen Exp $
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
    if ( (*it)->whyClosed == FileRecord::notClosed )
      ++_openFiles;
  }

  _closedFiles = _entryCounter - _openFiles;
}


std::string FilesMonitorCollection::FileRecord::closingReason()
{
  switch (whyClosed)
  {
    case notClosed:   return "open";
    case stop:        return "run stopped";
    case endOfLS:     return "LS ended";
    case timeout:     return "timeout";
    case size:        return "file size";
    case truncated:   return "TRUNCATED";
    default:          return "unknown";
  }
}


std::string FilesMonitorCollection::FileRecord::filePath()
{
  return ( baseFilePath + (whyClosed == notClosed ? "/open/" : "/closed/") );
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
