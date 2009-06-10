// $Id$

#include <string>
#include <sstream>
#include <iomanip>

#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/FilesMonitorCollection.h"

using namespace stor;

FilesMonitorCollection::FilesMonitorCollection(xdaq::Application *app) :
MonitorCollection(app),
_maxFileEntries(250),
_entryCounter(0),
_numberOfErasedRecords(0)
{
  boost::mutex::scoped_lock sl(_fileRecordsMutex);
  _fileRecords.reserve(_maxFileEntries);

  _infoSpaceItems.push_back(std::make_pair("openFiles", &_openFiles));
  _infoSpaceItems.push_back(std::make_pair("closedFiles", &_closedFiles));

  // These infospace items were defined in the old SM
  // _infoSpaceItems.push_back(std::make_pair("fileList", &_fileList));
  // _infoSpaceItems.push_back(std::make_pair("eventsInFile", &_eventsInFile));
  // _infoSpaceItems.push_back(std::make_pair("fileSize", &_fileSize));

  putItemsIntoInfoSpace();
}


const FilesMonitorCollection::FileRecordPtr
FilesMonitorCollection::getNewFileRecord()
{
  boost::mutex::scoped_lock sl(_fileRecordsMutex);

  if (_fileRecords.size() >= _maxFileEntries)
  {
    ++_numberOfErasedRecords;
    _fileRecords.erase(_fileRecords.begin());
  }

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


void FilesMonitorCollection::do_updateInfoSpace()
{
  boost::mutex::scoped_lock sl(_fileRecordsMutex);

  std::string errorMsg =
    "Failed to update values of items in info space " + _infoSpace->name();

  // Lock the infospace to assure that all items are consistent
  try
  {
    _infoSpace->lock();

    _openFiles = 0;
    _closedFiles = _numberOfErasedRecords;

    for (
      FileRecordList::const_iterator it = _fileRecords.begin(),
        itEnd = _fileRecords.end();
      it != itEnd;
      ++it
    )
    {
      if ( (*it)->whyClosed == FileRecord::notClosed )
        ++_openFiles;
      else
        ++_closedFiles;
    }

    _infoSpace->unlock();
  }
  catch(std::exception &e)
  {
    _infoSpace->unlock();
 
    errorMsg += ": ";
    errorMsg += e.what();
    XCEPT_RAISE(stor::exception::Monitoring, errorMsg);
  }
  catch (...)
  {
    _infoSpace->unlock();
 
    errorMsg += " : unknown exception";
    XCEPT_RAISE(stor::exception::Monitoring, errorMsg);
  }

  try
  {
    // The fireItemGroupChanged locks the infospace
    _infoSpace->fireItemGroupChanged(_infoSpaceItemNames, this);
  }
  catch (xdata::exception::Exception &e)
  {
    XCEPT_RETHROW(stor::exception::Infospace, errorMsg, e);
  }
}


void FilesMonitorCollection::do_reset()
{
  boost::mutex::scoped_lock sl(_fileRecordsMutex);
  _fileRecords.clear();
  _entryCounter = 0;
  _numberOfErasedRecords = 0;
}


std::string FilesMonitorCollection::FileRecord::closingReason()
{
  switch (whyClosed)
  {
    case notClosed:   return "open";
    case stop:        return "run stopped";
    case Nminus2lumi: return "LS changed";
    case timeout:     return "timeout";
    case size:        return "file size";
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
