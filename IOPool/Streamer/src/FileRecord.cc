// $Id: FileRecord.cc,v 1.6 2007/01/10 18:06:09 klute Exp $

#include "IOPool/Streamer/interface/FileRecord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
FileRecord::FileRecord(int lumi, string file, string path):
  fileName_(file),
  basePath_(path),
  fileSystem_(""),
  workingDir_("/open/"),
  statFileName_("summaryCatalog.txt"),
  mailBoxPath_(path+"/mbox"),
  lumiSection_(lumi),
  fileCounter_(0),
  fileSize_(0), 
  events_(0), 
  firstEntry_(0.0), 
  lastEntry_(0.0)
{
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
  ofstream of(statFileName_.c_str(), ios_base::ate | ios_base::out | ios_base::app );
  of << currentStatString;
  of.close();
}


//
// *** return a formatted string for the file counter
//
string FileRecord::fileCounterStr()
{
  std::ostringstream oss;
  oss << "." << setfill('0') << std::setw(4) << fileCounter_;
  return oss.str();
}


//
// *** return the full path
//
string FileRecord::filePath()
{
  return ( basePath_ + fileSystem_ + workingDir_);
}


//
// *** return the complete file name and path (w/o file ending)
//
string FileRecord::completeFileName()
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
  string openIndexFileName      = completeFileName() + ".ind";
  string openStreamerFileName   = completeFileName() + ".dat";
  workingDir_ = "/closed/";
  string closedIndexFileName    = completeFileName() + ".ind";
  string closedStreamerFileName = completeFileName() + ".dat";

  int result = rename( openIndexFileName.c_str()    , closedIndexFileName.c_str() );
  result    += rename( openStreamerFileName.c_str() , closedStreamerFileName.c_str() );
 
  if (result != 0 )
    cout << " *** FileRecord::closeFile()  Houston there is a problem moving " 
	 << openStreamerFileName << " to " << closedStreamerFileName << endl;
 
}


string FileRecord::timeStamp(double time)
{
  time_t rawtime = (time_t) time ;
  tm * ptm;
  ptm = localtime ( &rawtime );
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


void FileRecord::checkDirectories()
{
  checkDirectory(basePath());
  checkDirectory(fileSystem());
  checkDirectory(fileSystem()+"/open");
  checkDirectory(fileSystem()+"/closed");
  checkDirectory(mailBoxPath_);
}


void FileRecord::checkDirectory(string path)
{
  struct stat buf;

  int retVal = stat(path.c_str(), &buf);
  if(retVal !=0 )
    {
      edm::LogError("testStorageManager") << "Directory " << path
					  << " does not exist. Error=" << errno ;
    }
}


//
// *** report status of FileRecord
//
void FileRecord::report(ostream &os, int indentation) const
{
  string prefix(indentation, ' ');
  os << "\n";
  os << prefix << "------------- FileRecord -------------\n";
  os << prefix << "fileName            " << fileName_       << "\n";
  os << prefix << "basePath_           " << basePath_       << "\n";  
  os << prefix << "workingDir_         " << workingDir_     << "\n";
  os << prefix << "fileSystem_         " << fileSystem_     << "\n";
  os << prefix << "statFileName_       " << statFileName_   << "\n";
  os << prefix << "mailBoxPath_        " << mailBoxPath_    << "\n";
  os << prefix << "lumiSection_        " << lumiSection_    << "\n";
  os << prefix << "fileCounter_        " << fileCounter_    << "\n";
  os << prefix << "fileSize            " << fileSize_       << "\n";
  os << prefix << "events              " << events_         << "\n";
  os << prefix << "first entry         " << firstEntry_     << "\n";
  os << prefix << "last entry          " << lastEntry_      << "\n";
  os << prefix << "-----------------------------------------\n";  
}


