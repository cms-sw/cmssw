// $Id: OutputService.cc,v 1.3 2008/01/22 19:28:36 muzaffar Exp $

#include <EventFilter/StorageManager/interface/OutputService.h>

#include <iostream>
#include <sys/time.h> 
 
using namespace edm;
using namespace std;
using boost::shared_ptr;


//
// *** OutputService
//
OutputService::OutputService(boost::shared_ptr<FileRecord> file, 
			     InitMsgView const& view):
  file_(file)
{
  string streamerFileName = file_ -> filePath() + file_ -> fileName() + file_ -> fileCounterStr() + ".dat";
  string indexFileName    = file_ -> filePath() + file_ -> fileName() + file_ -> fileCounterStr() + ".ind";

  writer_ = shared_ptr<StreamerFileWriter> (new StreamerFileWriter(streamerFileName, indexFileName));
  writeHeader(view);

  file_ -> firstEntry(getTimeStamp());
  file_ -> insertFileInDatabase();
}


//
// *** call close file
//
OutputService::~OutputService()
{
  closeFile();
}


// 
// *** write file header
// *** write increase file size by file header size
//
void OutputService::writeHeader(InitMsgView const& view)
{
  writer_ -> doOutputHeader(view);
  file_   -> increaseFileSize(view.size());
}


//
// *** write event to file
// *** increase the file size
// *** update time of last entry 
// *** increase event count
//
void OutputService::writeEvent(EventMsgView const& view)
{
  writer_ -> doOutputEvent(view);
  file_   -> increaseFileSize(view.size());
  file_   -> lastEntry(getTimeStamp());
  file_   -> increaseEventCount();
}


// 
// *** stop file write
// *** add end of file record size to file size
// *** move file to "closed" directory
// *** write to summary catalog
// *** write to mail box
//
void OutputService::closeFile()
{
  writer_ -> stop();
  file_   -> increaseFileSize(writer_->getStreamEOFSize());
  file_   -> moveFileToClosed();
  file_   -> writeToSummaryCatalog();
// file_   -> writeToMailBox();
  file_   -> updateDatabase();
  file_   -> notifyTier0();
}


// 
// *** get the current time stamp
//
double OutputService::getTimeStamp() const
{
  struct timeval now;
  struct timezone dummyTZ;
  gettimeofday(&now, &dummyTZ);
  return (double) now.tv_sec + (double) now.tv_usec / 1000000.0;
}


//
// *** report status of OutputService
//
void OutputService::report(ostream &os, int indentation) const
{
  string prefix(indentation, ' ');
  os << prefix << "------------- OutputService -------------\n";
  file_ -> report(os,indentation);
  double time = (double) file_ -> lastEntry() - (double) file_ -> firstEntry();
  double rate = (time>0) ? (double) file_ -> events() / (double) time : 0.; 
  double tput = (time>0) ? (double) file_ -> fileSize() / ((double) time * 1048576.) : 0.; 
  os << prefix << "rate                " << rate            << " evts/s\n";
  os << prefix << "throughput          " << tput            << " MB/s\n";
  os << prefix << "time                " << time            << " s\n";
  os << prefix << "-----------------------------------------\n";  
}
