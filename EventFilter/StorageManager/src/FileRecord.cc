// $Id: FileRecord.cc,v 1.8 2008/04/29 19:09:07 loizides Exp $

#include <EventFilter/StorageManager/interface/FileRecord.h>
#include <EventFilter/StorageManager/interface/Configurator.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <FWCore/Utilities/interface/GetReleaseVersion.h>

#include <errno.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <sys/stat.h>

using namespace edm;
using namespace std;

//
// *** FileRecord
//
FileRecord::FileRecord(int lumi, const string &file, const string &path):
  smParameter_(stor::Configurator::instance()->getParameter()),
  fileName_(file),
  basePath_(path),
  fileSystem_(""),
  workingDir_("/open/"),
  mailBoxPath_(path+"/mbox"),
  logPath_(path+"/log"),
  logFile_(logFile()),
  setupLabel_(""),
  streamLabel_(""),
  cmsver_(getReleaseVersion()),
  lumiSection_(lumi),
  runNumber_(0),
  fileCounter_(0),
  fileSize_(0), 
  events_(0), 
  firstEntry_(0.0), 
  lastEntry_(0.0)
{
   // stripp quotes if present
   if(cmsver_[0]=='"') cmsver_=cmsver_.substr(1,cmsver_.size()-2);
}


//
// *** write mailbox entry 
// 
void FileRecord::writeToMailBox()
{
  ostringstream oss;
  oss << mailBoxPath_ << "/" << fileName_ << fileCounterStr() << ".smry";
  ofstream of(oss.str().c_str());
  of << completeFileName();
  of.close();
}


//
// *** write summary information in catalog
//
void FileRecord::writeToSummaryCatalog()
{
  ostringstream currentStat;
  string ind(":");
  currentStat << workingDir()           << ind
	      << fileName()  
	      << fileCounterStr()       << ind
	      << fileSize()             << ind 
	      << events()               << ind
              << timeStamp(lastEntry()) << ind
	      << (int) (lastEntry()-firstEntry()) << endl;
  string currentStatString (currentStat.str());
  ofstream of(smParameter_->fileCatalog().c_str(), ios_base::ate | ios_base::out | ios_base::app );
  of << currentStatString;
  of.close();
}


//
// *** write notify info to file
//
void FileRecord::notifyTier0()
{

  std::ostringstream oss;
  oss << "./notifyTier0.pl "
      << " --FILENAME "     << fileName() << fileCounterStr() <<  ".dat"
      << " --COUNT "        << fileCounter_                       
      << " --NEVENTS "      << events_                            
      << " --FILESIZE "     << fileSize_                          
      << " --START_TIME "   << (int) firstEntry()
      << " --STOP_TIME "    << (int) lastEntry()
      << " --STATUS "       << "closed"
      << " --RUNNUMBER "    << runNumber_                         
      << " --LUMISECTION "  << lumiSection_                      
      << " --PATHNAME "     << filePath()
      << " --HOSTNAME "     << smParameter_->host()
      << " --DATASET "      << setupLabel_ 
      << " --STREAM "       << streamLabel_                      
      << " --INSTANCE "     << smParameter_->smInstance()        
      << " --SAFETY "       << smParameter_->initialSafetyLevel()
      << " --APP_VERSION "  << cmsver_
      << " --APP_NAME CMSSW"
      << " --TYPE streamer"               
      << " --CHECKSUM 0\n";

  ofstream of(logFile_.c_str(), ios_base::ate | ios_base::out | ios_base::app );
  of << oss.str().c_str();
  of.close();
}


//
// *** write file information to log
//
void FileRecord::updateDatabase()
{
  std::ostringstream oss;
  oss << "./closeFile.pl "
      << " --FILENAME "     << fileName() << fileCounterStr() <<  ".dat"
      << " --COUNT "        << fileCounter_                       
      << " --NEVENTS "      << events_                            
      << " --FILESIZE "     << fileSize_                          
      << " --START_TIME "   << (int) firstEntry()
      << " --STOP_TIME "    << (int) lastEntry()
      << " --STATUS "       << "closed"
      << " --RUNNUMBER "    << runNumber_                         
      << " --LUMISECTION "  << lumiSection_                      
      << " --PATHNAME "     << filePath()
      << " --HOSTNAME "     << smParameter_->host()
      << " --DATASET "      << setupLabel_ 
      << " --STREAM "       << streamLabel_                      
      << " --INSTANCE "     << smParameter_->smInstance()        
      << " --SAFETY "       << smParameter_->initialSafetyLevel()
      << " --APP_VERSION "  << cmsver_
      << " --APP_NAME CMSSW"
      << " --TYPE streamer"               
      << " --CHECKSUM 0\n";

  ofstream of(logFile_.c_str(), ios_base::ate | ios_base::out | ios_base::app );
  of << oss.str().c_str();
  of.close();
}


//
// *** write file information to log
//
void FileRecord::insertFileInDatabase()
{
  std::ostringstream oss;
  oss << "./insertFile.pl "
      << " --FILENAME "     << fileName() << fileCounterStr() <<  ".dat"
      << " --COUNT "        << fileCounter_                       
      << " --NEVENTS "      << events_                            
      << " --FILESIZE "     << fileSize_                          
      << " --START_TIME "   << (int) firstEntry()
      << " --STOP_TIME "    << (int) lastEntry()
      << " --STATUS "       << "open"
      << " --RUNNUMBER "    << runNumber_                         
      << " --LUMISECTION "  << lumiSection_                      
      << " --PATHNAME "     << filePath()
      << " --HOSTNAME "     << smParameter_->host()
      << " --DATASET "      << setupLabel_ 
      << " --STREAM "       << streamLabel_                      
      << " --INSTANCE "     << smParameter_->smInstance()        
      << " --SAFETY "       << smParameter_->initialSafetyLevel()
      << " --APP_VERSION "  << cmsver_
      << " --APP_NAME CMSSW"
      << " --TYPE streamer"               
      << " --CHECKSUM 0\n";

  ofstream of(logFile_.c_str(), ios_base::ate | ios_base::out | ios_base::app );
  of << oss.str().c_str();
  of.close();
}


//
// *** return a formatted string for the file counter
//
string FileRecord::fileCounterStr() const
{
  std::ostringstream oss;
  oss << "." << setfill('0') << std::setw(4) << fileCounter_;
  return oss.str();
}


//
// *** return the full path
//
string FileRecord::filePath() const
{
  return ( basePath_ + fileSystem_ + workingDir_);
}


//
// *** return the complete file name and path (w/o file ending)
//
string FileRecord::completeFileName() const
{
  return ( basePath_ + fileSystem_ + workingDir_ + fileName_ + fileCounterStr() );
}


// 
// *** set the current file system
// 
void FileRecord::fileSystem(int i)
{
  std::ostringstream oss;
  oss << "/" << setfill('0') << std::setw(2) << i; 
  fileSystem_ = oss.str();
}


//
// *** move index and streamer file to "closed" directory
//
void FileRecord::moveFileToClosed()
{
  struct stat initialStatBuff, finalStatBuff;
  int statStatus;
  double pctDiff;
  bool sizeMismatch;

  string openIndexFileName      = completeFileName() + ".ind";
  string openStreamerFileName   = completeFileName() + ".dat";
  statStatus = stat(openStreamerFileName.c_str(), &initialStatBuff);
  if (statStatus != 0) {
    throw cms::Exception("FileRecord", "moveFileToClosed")
      << "Error checking the status of open file "
      << openStreamerFileName << ".  Has the file moved unexpectedly?"
      << std::endl;
  }
  sizeMismatch = false;
  if (smParameter_->exactFileSizeTest()) {
    if (fileSize_ != initialStatBuff.st_size) {
      sizeMismatch = true;
    }
  }
  else {
    pctDiff = calcPctDiff(fileSize_, initialStatBuff.st_size);
    if (pctDiff > 0.1) {sizeMismatch = true;}
  }
  if (sizeMismatch) {
    throw cms::Exception("FileRecord", "moveFileToClosed")
      << "Found an unexpected open file size when trying to move "
      << "the file to the closed state.  File " << openStreamerFileName
      << " has an actual size of " << initialStatBuff.st_size
      << " instead of the expected size of " << fileSize_ << std::endl;
  }

  int ronly  = chmod(openStreamerFileName.c_str(), S_IREAD|S_IRGRP|S_IROTH);
  ronly     += chmod(openIndexFileName.c_str(), S_IREAD|S_IRGRP|S_IROTH);
  if (ronly != 0) {
    throw cms::Exception("FileRecord", "moveFileToClosed")
      << "Unable to change permissions of " << openStreamerFileName << " and "
      << openIndexFileName << "to read only." << std::endl;
  }

  workingDir_ = "/closed/";
  string closedIndexFileName    = completeFileName() + ".ind";
  string closedStreamerFileName = completeFileName() + ".dat";

  int result = rename( openIndexFileName.c_str()    , closedIndexFileName.c_str() );
  result    += rename( openStreamerFileName.c_str() , closedStreamerFileName.c_str() );
  if (result != 0) {
    throw cms::Exception("FileRecord", "moveFileToClosed")
      << "Unable to move " << openStreamerFileName << " to "
      << closedStreamerFileName << ".  Possibly the storage manager "
      << "disk areas are full." << std::endl;
  }

  statStatus = stat(closedStreamerFileName.c_str(), &finalStatBuff);
  if (statStatus != 0) {
    throw cms::Exception("FileRecord", "moveFileToClosed")
      << "Error checking the status of closed file "
      << closedStreamerFileName << ".  This file was copied from "
      << openStreamerFileName << ", and the copy seems to have failed."
      << std::endl;
  }
  sizeMismatch = false;
  if (smParameter_->exactFileSizeTest()) {
    if (initialStatBuff.st_size != finalStatBuff.st_size) {
      sizeMismatch = true;
    }
  }
  else {
    pctDiff = calcPctDiff(initialStatBuff.st_size, finalStatBuff.st_size);
    if (pctDiff > 0.1) {sizeMismatch = true;}
  }
  if (sizeMismatch) {
    throw cms::Exception("FileRecord", "moveFileToClosed")
      << "Error moving " << openStreamerFileName << " to "
      << closedStreamerFileName << ".  The closed file size ("
      << finalStatBuff.st_size << ") is different than the open file size ("
      << initialStatBuff.st_size << ").  Possibly the storage manager "
      << "disk areas are full." << std::endl;
  }
}


string FileRecord::timeStamp(double time) const
{
  time_t rawtime = (time_t)time;
  tm * ptm;
  ptm = localtime(&rawtime);
  ostringstream timeStampStr;
  string colon(":");
  string slash("/");
  timeStampStr << setfill('0') << std::setw(2) << ptm->tm_mday      << slash 
	       << setfill('0') << std::setw(2) << ptm->tm_mon+1     << slash
	       << setfill('0') << std::setw(4) << ptm->tm_year+1900 << colon
               << setfill('0') << std::setw(2) << ptm->tm_hour      << slash
	       << setfill('0') << std::setw(2) << ptm->tm_min       << slash
	       << setfill('0') << std::setw(2) << ptm->tm_sec;
  return timeStampStr.str();
}


string FileRecord::logFile() const
{
  time_t rawtime = time(0);
  tm * ptm;
  ptm = localtime(&rawtime);

  ostringstream logfilename;
  logfilename << logPath_ << "/"
              << setfill('0') << std::setw(4) << ptm->tm_year+1900
              << setfill('0') << std::setw(2) << ptm->tm_mon+1
              << setfill('0') << std::setw(2) << ptm->tm_mday
              << "-" << smParameter_->host()
              << "-" << smParameter_->smInstance()
              << ".log";
  return logfilename.str();
}


void FileRecord::checkDirectories() const
{
  checkDirectory(basePath());
  checkDirectory(fileSystem());
  checkDirectory(fileSystem()+"/open");
  checkDirectory(fileSystem()+"/closed");
  checkDirectory(mailBoxPath_);
  checkDirectory(logPath_);
}


void FileRecord::checkDirectory(const string &path) const
{
  struct stat buf;

  int retVal = stat(path.c_str(), &buf);
  if(retVal !=0 )
    {
      edm::LogError("StorageManager") << "Directory " << path
				      << " does not exist. Error=" << errno ;
      throw cms::Exception("FileRecord","checkDirectory")
            << "Directory " << path << " does not exist. Error=" << errno << std::endl;
    }
}


double FileRecord::calcPctDiff(long long value1, long long value2) const
{
  if (value1 == value2) {return 0.0;}
  long long largerValue = value1;
  long long smallerValue = value2;
  if (value1 < value2) {
    largerValue = value2;
    smallerValue = value1;
  }
  return ((double) largerValue - (double) smallerValue) / (double) largerValue;
}


//
// *** report status of FileRecord
//
void FileRecord::report(ostream &os, int indentation) const
{
  string prefix(indentation, ' ');
  os << "\n";
  os << prefix << "------------- FileRecord -------------\n";
  os << prefix << "fileName            " << fileName_                   << "\n";
  os << prefix << "basePath_           " << basePath_                   << "\n";  
  os << prefix << "fileSystem_         " << fileSystem_                 << "\n";
  os << prefix << "workingDir_         " << workingDir_                 << "\n";
  os << prefix << "mailBoxPath_        " << mailBoxPath_                << "\n";
  os << prefix << "logPath_            " << logPath_                    << "\n";
  os << prefix << "logFile_            " << logFile_                    << "\n";
  os << prefix << "setupLabel_         " << setupLabel_                 << "\n";
  os << prefix << "streamLabel_        " << streamLabel_                << "\n";
  os << prefix << "hostName_           " << smParameter_->host()        << "\n";
  os << prefix << "fileCatalog()       " << smParameter_->fileCatalog() << "\n"; 
  os << prefix << "lumiSection_        " << lumiSection_                << "\n";
  os << prefix << "runNumber_          " << runNumber_                  << "\n";
  os << prefix << "fileCounter_        " << fileCounter_                << "\n";
  os << prefix << "fileSize            " << fileSize_                   << "\n";
  os << prefix << "events              " << events_                     << "\n";
  os << prefix << "first entry         " << firstEntry_                 << "\n";
  os << prefix << "last entry          " << lastEntry_                  << "\n";
  os << prefix << "-----------------------------------------\n";  
}
