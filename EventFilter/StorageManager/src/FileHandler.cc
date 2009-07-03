// $Id: FileHandler.cc,v 1.2 2009/06/10 08:15:26 dshpakov Exp $

#include <EventFilter/StorageManager/interface/Exception.h>
#include <EventFilter/StorageManager/interface/FileHandler.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <FWCore/Utilities/interface/GetReleaseVersion.h>

#include <errno.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <sys/stat.h>

using namespace stor;
using namespace std;


FileHandler::FileHandler
(
  FilesMonitorCollection::FileRecordPtr fileRecord,
  const DiskWritingParams& dwParams,
  const long long& maxFileSize
):
_fileRecord(fileRecord),
_firstEntry(utils::getCurrentTime()),
_lastEntry(0),
_closingReason(FilesMonitorCollection::FileRecord::stop),
_diskWritingParams(dwParams),
_maxFileSize(maxFileSize),
_logPath(dwParams._filePath+"/log"),
_logFile(logFile(dwParams)),
_cmsver(edm::getReleaseVersion())
{
  // stripp quotes if present
  if(_cmsver[0]=='"') _cmsver=_cmsver.substr(1,_cmsver.size()-2);

  checkDirectories();

  insertFileInDatabase();
}


//////////////////////
// File bookkeeping //
//////////////////////

void FileHandler::writeToSummaryCatalog() const
{
  ostringstream currentStat;
  string ind(":");
  currentStat << _fileRecord->filePath()               << ind
              << _fileRecord->fileName()               << ind
	      << fileSize()                            << ind 
	      << events()                              << ind
              << utils::timeStamp(_lastEntry)          << ind
	      << (int) (_lastEntry - _firstEntry)      << ind
	      << _fileRecord->whyClosed << endl;
  string currentStatString (currentStat.str());
  ofstream of(_diskWritingParams._fileCatalog.c_str(), ios_base::ate | ios_base::out | ios_base::app );
  of << currentStatString;
  of.close();
}


void FileHandler::updateDatabase() const
{
  std::ostringstream oss;
  oss << "./closeFile.pl "
      << " --FILENAME "     << _fileRecord->fileName() <<  ".dat"
      << " --FILECOUNTER "  << _fileRecord->fileCounter
      << " --NEVENTS "      << events()
      << " --FILESIZE "     << fileSize()                          
      << " --STARTTIME "    << (int) _firstEntry
      << " --STOPTIME "     << (int) _lastEntry
      << " --STATUS "       << "closed"
      << " --RUNNUMBER "    << _fileRecord->runNumber
      << " --LUMISECTION "  << _fileRecord->lumiSection
      << " --PATHNAME "     << _fileRecord->filePath()
      << " --HOSTNAME "     << _diskWritingParams._hostName
      << " --SETUPLABEL "   << _diskWritingParams._setupLabel
      << " --STREAM "       << _fileRecord->streamLabel                      
      << " --INSTANCE "     << _diskWritingParams._smInstanceString
      << " --SAFETY "       << _diskWritingParams._initialSafetyLevel
      << " --APPVERSION "   << _cmsver
      << " --APPNAME CMSSW"
      << " --TYPE streamer"               
      << " --DEBUGCLOSE "   << _fileRecord->whyClosed
      << " --CHECKSUM "     << hex << _adlerstream
      << " --CHECKSUMIND "  << hex << _adlerindex
      << "\n";

  ofstream of(_logFile.c_str(), ios_base::ate | ios_base::out | ios_base::app );
  of << oss.str().c_str();
  of.close();
}


void FileHandler::insertFileInDatabase() const
{
  std::ostringstream oss;
  oss << "./insertFile.pl "
      << " --FILENAME "     << _fileRecord->fileName() <<  ".dat"
      << " --FILECOUNTER "  << _fileRecord->fileCounter
      << " --NEVENTS "      << events()
      << " --FILESIZE "     << fileSize()
      << " --STARTTIME "    << (int) _firstEntry
      << " --STOPTIME 0"
      << " --STATUS open"
      << " --RUNNUMBER "    << _fileRecord->runNumber
      << " --LUMISECTION "  << _fileRecord->lumiSection
      << " --PATHNAME "     << _fileRecord->filePath()
      << " --HOSTNAME "     << _diskWritingParams._hostName
      << " --SETUPLABEL "   << _diskWritingParams._setupLabel
      << " --STREAM "       << _fileRecord->streamLabel
      << " --INSTANCE "     << _diskWritingParams._smInstanceString
      << " --SAFETY "       << _diskWritingParams._initialSafetyLevel
      << " --APPVERSION "   << _cmsver
      << " --APPNAME CMSSW"
      << " --TYPE streamer"               
      << " --CHECKSUM 0"
      << " --CHECKSUMIND 0"
      << "\n";

  ofstream of(_logFile.c_str(), ios_base::ate | ios_base::out | ios_base::app );
  of << oss.str().c_str();
  of.close();
}


const bool FileHandler::tooLarge(const unsigned long& dataSize)
{
  if ( ((fileSize() + dataSize) > _maxFileSize) && (events() > 0) )
  {
    _closingReason = FilesMonitorCollection::FileRecord::size;
    return true;
  }
  else
  {
    return false;
  }
}


const int FileHandler::events() const
{
  return _fileRecord->eventCount;
}


const long long FileHandler::fileSize() const
{
  return _fileRecord->fileSize;
}


/////////////////////////////
// File system interaction //
/////////////////////////////


void FileHandler::moveFileToClosed(const bool& useIndexFile)
{
  string openIndexFileName      = _fileRecord->completeFileName() + ".ind";
  string openStreamerFileName   = _fileRecord->completeFileName() + ".dat";

  size_t openStreamerFileSize = checkFileSizeMatch(openStreamerFileName, fileSize());

  makeFileReadOnly(openStreamerFileName);
  if (useIndexFile) makeFileReadOnly(openIndexFileName);

  _fileRecord->whyClosed = _closingReason;
  string closedIndexFileName    = _fileRecord->completeFileName() + ".ind";
  string closedStreamerFileName = _fileRecord->completeFileName() + ".dat";

  if (useIndexFile) renameFile(openIndexFileName, closedIndexFileName);
  renameFile(openStreamerFileName, closedStreamerFileName);

  checkFileSizeMatch(closedStreamerFileName, openStreamerFileSize);
}


size_t FileHandler::checkFileSizeMatch(const string& fileName, const size_t& size) const
{
  struct stat64 statBuff;
  int statStatus = stat64(fileName.c_str(), &statBuff);
  if ( statStatus != 0 )
  {
    std::ostringstream msg;
    msg << "Error checking the status of open file "
      << fileName;
    XCEPT_RAISE(stor::exception::DiskWriting, msg.str());
  }
  
  if ( sizeMismatch(size, statBuff.st_size) )
  {
    std::ostringstream msg;
    msg << "Found an unexpected file size when trying to move "
      << "the file to the closed state.  File " << fileName
      << " has an actual size of " << statBuff.st_size
      << " instead of the expected size of " << size;
    XCEPT_RAISE(stor::exception::DiskWriting, msg.str());
  }

  return statBuff.st_size;
}


bool FileHandler::sizeMismatch(const double& initialSize, const double& finalSize) const
{
  if (_diskWritingParams._exactFileSizeTest) {
    if (initialSize != finalSize) {
      return true;
    }
  }
  else {
    double pctDiff = calcPctDiff(initialSize, finalSize);
    if (pctDiff > 0.1) {return true;}
  }
  return false;
}


void FileHandler::makeFileReadOnly(const string& fileName) const
{
  int ronly  = chmod(fileName.c_str(), S_IREAD|S_IRGRP|S_IROTH);
  if (ronly != 0) {
    std::ostringstream msg;
    msg << "Unable to change permissions of " << fileName
      << " to read only." << std::endl;
    XCEPT_RAISE(stor::exception::DiskWriting, msg.str());
  }
}


void FileHandler::renameFile(const string& openFileName, const string& closedFileName) const
{
  int result = rename( openFileName.c_str(), closedFileName.c_str() );
  if (result != 0) {
    std::ostringstream msg;
    msg << "Unable to move " << openFileName << " to "
      << closedFileName << ".  Possibly the storage manager "
      << "disk areas are full.";
    XCEPT_RAISE(stor::exception::DiskWriting, msg.str());
  }
}


const string FileHandler::logFile(const DiskWritingParams& dwp) const
{
  time_t rawtime = time(0);
  tm * ptm;
  ptm = localtime(&rawtime);

  ostringstream logfilename;
  logfilename << _logPath << "/"
              << setfill('0') << std::setw(4) << ptm->tm_year+1900
              << setfill('0') << std::setw(2) << ptm->tm_mon+1
              << setfill('0') << std::setw(2) << ptm->tm_mday
              << "-" << dwp._hostName
              << "-" << dwp._smInstanceString
              << ".log";
  return logfilename.str();
}


void FileHandler::checkDirectories() const
{
  utils::checkDirectory(_diskWritingParams._filePath);
  utils::checkDirectory(_fileRecord->baseFilePath);
  utils::checkDirectory(_fileRecord->baseFilePath + "/open");
  utils::checkDirectory(_fileRecord->baseFilePath + "/closed");
  utils::checkDirectory(_logPath);
}


const double FileHandler::calcPctDiff(const double& value1, const double& value2) const
{
  if (value1 == value2) return 0;
  double largerValue = value1;
  double smallerValue = value2;
  if (value1 < value2) {
    largerValue = value2;
    smallerValue = value1;
  }
  return ( largerValue > 0 ? (largerValue - smallerValue) / largerValue : 0 );
}



/////////////////////////////
// File information dumper //
/////////////////////////////

void FileHandler::info(ostream& os) const
{
  os << _fileRecord->fileCounter << " "
     << _fileRecord->completeFileName() << " " 
     << events() << " "
     << fileSize();
}



/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
