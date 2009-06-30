// $Id: FRDOutputService.cc,v 1.5 2008/05/13 18:06:46 loizides Exp $

#include <EventFilter/StorageManager/interface/FRDOutputService.h>
#include <IOPool/Streamer/interface/FRDEventMessage.h>

#include <iostream>
 
using namespace edm;
using namespace std;
using boost::shared_ptr;


//
// *** FRDOutputService
//
FRDOutputService::FRDOutputService(boost::shared_ptr<FileRecord> file)
{
  file_ = file;

  string fileName = file_ -> filePath() + file_ -> fileName() + file_ -> fileCounterStr() + ".dat";

  writer_ = shared_ptr<FRDEventFileWriter> (new FRDEventFileWriter(fileName));

  file_ -> firstEntry(getTimeStamp());
  file_ -> insertFileInDatabase();
}


//
// *** call close file
//
FRDOutputService::~FRDOutputService()
{
  //std::cout << "FRDOutputService Destructor called." << std::endl;
  closeFile();
}


//
// *** write event to file
// *** increase the file size
// *** update time of last entry 
// *** increase event count
//
void FRDOutputService::writeEvent(const uint8 * const bufPtr)
{
  FRDEventMsgView view((void *) bufPtr);
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
void FRDOutputService::closeFile()
{
  writer_ -> stop();
  file_   -> moveErrorFileToClosed();
  file_   -> writeToSummaryCatalog();
  file_   -> updateDatabase();
}


//
// *** report status of OutputService
//
void FRDOutputService::report(ostream &os, int indentation) const
{
  string prefix(indentation, ' ');
  os << prefix << "------------- FRDOutputService -------------\n";
  file_ -> report(os,indentation);
  double time = (double) file_ -> lastEntry() - (double) file_ -> firstEntry();
  double rate = (time>0) ? (double) file_ -> events() / (double) time : 0.; 
  double tput = (time>0) ? (double) file_ -> fileSize() / ((double) time * 1048576.) : 0.; 
  os << prefix << "rate                " << rate            << " evts/s\n";
  os << prefix << "throughput          " << tput            << " MB/s\n";
  os << prefix << "time                " << time            << " s\n";
  os << prefix << "-----------------------------------------\n";  
}
