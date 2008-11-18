// $Id: StreamService.cc,v 1.14 2008/08/21 09:00:39 loizides Exp $

#include <EventFilter/StorageManager/interface/StreamService.h>
#include <EventFilter/StorageManager/interface/ProgressMarker.h>
#include <EventFilter/StorageManager/interface/Parameter.h>
#include "EventFilter/StorageManager/interface/Configurator.h"

#include <iostream>
#include <iomanip>
#include <sys/time.h> 
#include <sys/stat.h>
#include <sys/statfs.h>

using namespace edm;
using namespace std;
using boost::shared_ptr;
using stor::ProgressMarker;

//
// *** destructor
//
StreamService::~StreamService()
{
}


//
// *** check that the file system exists and that there
// *** is enough disk space
//
bool StreamService::checkFileSystem() const
{
  struct statfs64 buf;
  int retVal = statfs64(filePath_.c_str(), &buf);
  if(retVal!=0)
    {
      std::cout << "StreamService: " << "Could not stat output filesystem for path " 
		<< filePath_ << std::endl;
      return false;
    }
  
  unsigned int btotal = 0;
  unsigned int bfree = 0;
  unsigned int blksize = 0;
  if(retVal==0)
    {
      blksize = buf.f_bsize;
      btotal = buf.f_blocks;
      bfree  = buf.f_bfree;
    }
  double dfree = double(bfree)/double(btotal);
  double dusage = 1. - dfree;

  if(dusage>highWaterMark_)
    {
      cout << "StreamService: " << "Output filesystem for path " << filePath_ 
	   << " is more than " << highWaterMark_*100 << "% full " << endl;
      return false;
    }

  return true;
}


//
// *** file private data member from parameter set
//
void StreamService::setStreamParameter()
{
  // some parameters common to streams are given in the XML file
  // these are defaults, actually set at configure time

  // 02-Sep-2008, KAB:  NOTE that most, if not all, of these parameters are
  // overwritten with either defaults from stor::Parameter or values set 
  // in the SM configuration (confdb/online or xml/offline).
  // The overwrite from the Parameter class happens in ServiceManager::manageInitMsg.

  streamLabel_        = parameterSet_.getParameter<string> ("streamLabel");
  maxSize_ = 1048576 * (long long) parameterSet_.getParameter<int> ("maxSize");
  fileName_           = ""; // set by setFileName
  filePath_           = ""; // set by setFilePath
  setupLabel_         = ""; // set by setSetupLabel
  highWaterMark_      = 0.9;// set by setHighWaterMark
  lumiSectionTimeOut_ = 45; // set by setLumiSectionTimeOut
  sourceId_           = ""; // set by setSourceId
  // report(cout, 4);
}


//
// *** get all files in this run (including open files)
// return for each the count, filename, number of events, file size separated by a space
//
std::list<std::string> StreamService::getFileList()
{
  boost::mutex::scoped_lock sl(list_lock_);

  std::list<std::string> files_=outputSummaryClosed_;
  for (OutputMapIterator it = outputMap_.begin() ; it != outputMap_.end(); ++it) {
    std::ostringstream entry;
    entry << it->first->fileCounter() << " " 
          << it->first->completeFileName() << " " 
          << it->first->events() << " "
          << it->first->fileSize();
    files_.push_back(entry.str());
  }

  return files_;
}

//
// *** Copy file string from OutputMap into OutputMapClosed
//
void StreamService::fillOutputSummaryClosed(const boost::shared_ptr<FileRecord> &file)
{
  boost::mutex::scoped_lock sl(list_lock_);

  std::ostringstream entry;
  entry << file->fileCounter() << " " 
        << file->completeFileName() << " " 
        << file->events() << " "
        << file->fileSize();
  outputSummaryClosed_.push_back(entry.str());
}


//
// *** get all open (current) files
//
std::list<std::string> StreamService::getCurrentFileList()
{
  std::list<std::string> files_;
  for (OutputMapIterator it = outputMap_.begin(), itEnd = outputMap_.end(); it != itEnd; ++it) {
    files_.push_back(it->first->completeFileName());
  }
  return files_;
}

//
// *** override maxSize from cfg if xdaq parameter was set 
//
void StreamService::setMaxFileSize(int x)
{
  maxFileSizeInMB_ = x;
  if(maxFileSizeInMB_ > 0)
    maxSize_ = 1048576 * (long long) maxFileSizeInMB_;
}

//
// *** get the current time
//
double StreamService::getCurrentTime() const
{
  struct timeval now;
  struct timezone dummyTZ;
  gettimeofday(&now, &dummyTZ);
  return ((double) now.tv_sec + ((double) now.tv_usec / 1000000.0));
}
