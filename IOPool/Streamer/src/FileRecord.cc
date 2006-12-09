// $Id: FileRecord.cc,v 1.1 2006/11/29 10:10:16 klute Exp $

#include "IOPool/Streamer/interface/FileRecord.h"

#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>

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
  fileSize_(0), events_(0), firstEntry_(0.0), lastEntry_(0.0)
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
  currentStat << fileName_    << ind 
	      << fileSize_    << ind 
	      << events_      << endl;
  string currentStatString (currentStat.str());
  ofstream of(statFileName_.c_str(), ios_base::ate | ios_base::out | ios_base::app );
  //of << currentStat;
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


