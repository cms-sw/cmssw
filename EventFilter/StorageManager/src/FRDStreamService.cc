// $Id: FRDStreamService.cc,v 1.3 2008/09/04 17:44:18 biery Exp $

#include <EventFilter/StorageManager/interface/FRDStreamService.h>
#include <EventFilter/StorageManager/interface/ProgressMarker.h>
#include <EventFilter/StorageManager/interface/Parameter.h>
#include "EventFilter/StorageManager/interface/Configurator.h"
#include "EventFilter/StorageManager/interface/FRDOutputService.h"  

#include <iostream>
#include <iomanip>
#include <sys/stat.h>
#include <sys/statfs.h>

using namespace edm;
using namespace std;
using boost::shared_ptr;
using stor::ProgressMarker;

//
// *** construct stream service from parameter set
//
FRDStreamService::FRDStreamService(ParameterSet const& pset)
{
  parameterSet_ = pset;
  runNumber_ = 0;
  lumiSection_ = 0;
  numberOfFileSystems_ = 0;
  maxFileSizeInMB_ = 0;
  maxSize_ = 0;
  highWaterMark_ = 0;
  lumiSectionTimeOut_ = 0;
  ntotal_ = 0;

  setStreamParameter();
}


// 
// *** event loop for stream service
//
bool FRDStreamService::nextEvent(const uint8 * const bufPtr)
{
  ProgressMarker::instance()->processing(true);
  FRDEventMsgView view((void *) bufPtr);

  // accept all Error events, so no call to any sort of acceptEvents() method...

  runNumber_   = view.run();
  lumiSection_ = 1;  // Error message doesn't yet have lumi section number
                     // *and* we want to keep all Error events in the same
                     // file, for now


  shared_ptr<OutputService> outputService = getOutputService(view);
  ProgressMarker::instance()->processing(false);
  
  ProgressMarker::instance()->writing(true);
  outputService->writeEvent(bufPtr);
  ProgressMarker::instance()->writing(false);
  return true;
}


//
// *** close all files on stop signal
//
void FRDStreamService::stop()
{
  for (OutputMapIterator it = outputMap_.begin(); it != outputMap_.end(); ) {
    boost::shared_ptr<FileRecord> fd(it->first);
    outputMap_.erase(it++);
    fillOutputSummaryClosed(fd);
  }
}


// 
// *** close all output service of the previous lumi-section 
// *** when lumiSectionTimeOut seconds have passed since the
// *** appearance of the new lumi section and make a record of the file
// !!! Deprecated - use closeTimedOutFiles() instead !!!
// 
void FRDStreamService::closeTimedOutFiles(int lumi, double timeoutdiff)
{
  // since we are currently storing all events in a single file,
  // we never close files at lumi section boundaries

  return;
}

// 
// *** close all output service when lumiSectionTimeOut seconds have passed
// *** since the most recent event was added
// 
void FRDStreamService::closeTimedOutFiles()
{
  // since we are currently storing all events in a single file,
  // we never close files at lumi section boundaries

  return;
}

//
// *** find output service in map or return a new one
// *** rule: only one file for each lumi section is output map
//
boost::shared_ptr<OutputService> FRDStreamService::getOutputService(FRDEventMsgView const& view)
{
  for (OutputMapIterator it = outputMap_.begin(); it != outputMap_.end(); ++it) {
       if (it->first->lumiSection() == lumiSection_) {
	  if (checkEvent(it->first, view))
	    return it->second;
	  else {
            boost::shared_ptr<FileRecord> fd(it->first);
            outputMap_.erase(it);
            fillOutputSummaryClosed(fd);
            break;
	  }
      }
  }
  return newOutputService();
}


// 
// *** generate file descriptor
// *** generate output service
// *** add ouput service to output map
// *** add ouput service to output summary
//
boost::shared_ptr<OutputService> FRDStreamService::newOutputService()
{
  boost::shared_ptr<FileRecord> file = generateFileRecord();

  shared_ptr<OutputService> outputService(new FRDOutputService(file));
  outputMap_[file] = outputService;

  return outputService;
}


//
// *** perform checks before writing the event
// *** so far ... check the event will fit into the file 
//
bool FRDStreamService::checkEvent(shared_ptr<FileRecord> file, FRDEventMsgView const& view) const
{
  if (file->fileSize() + static_cast<long long>(view.size()) > maxSize_ && file->events() > 0)
    return false;

  return true;
}


//
// *** generate a unique file descriptor
//
//     The run number, stream name and storage manager instance have 
//     to be part of the file name. I have added the lumi section, 
//     but in any case we have to make sure that file names are
//     unique. 
//
//     Keep a list of file names and check if file name 
//     was not already used in this run. 
//
boost::shared_ptr<FileRecord> FRDStreamService::generateFileRecord()
{
  std::ostringstream oss;   
  oss    << setupLabel_ 
	 << "." << setfill('0') << std::setw(8) << runNumber_ 
	 << "." << setfill('0') << std::setw(4) << lumiSection_
	 << "." << streamLabel_ 
	 << "." << fileName_
	 << "." << setfill('0') << std::setw(2) << sourceId_;
  string fileName = oss.str();

  shared_ptr<FileRecord> fd = shared_ptr<FileRecord>(new FileRecord(lumiSection_, fileName, filePath_));    
  ++ntotal_;

  boost::mutex::scoped_lock sl(list_lock_);
  map<string, int>::iterator it = outputSummary_.find(fileName);
  if(it==outputSummary_.end()) {
     outputSummary_.insert(std::pair<string, int>(fileName,0));
  } else {
     ++it->second;
     fd->setFileCounter(it->second);
  }

  if (numberOfFileSystems_ > 0)
    fd->fileSystem((runNumber_ + atoi(sourceId_.c_str()) + ntotal_) % numberOfFileSystems_); 
  
  fd->checkDirectories();
  fd->setRunNumber(runNumber_);
  fd->setStreamLabel(streamLabel_);
  fd->setSetupLabel(setupLabel_);

  // fd->report(cout, 12);
  return fd;
}

//
// *** report the status of stream service
//
void FRDStreamService::report(ostream &os, int indentation) const
{
  string prefix(indentation, ' ');
  os << "\n";
  os << prefix << "------------- FRDStreamService -------------\n";
  os << prefix << "fileName            " << fileName_              << "\n";
  os << prefix << "filePath            " << filePath_              << "\n";
  os << prefix << "sourceId            " << sourceId_              << "\n";
  os << prefix << "setupLabel          " << setupLabel_            << "\n";
  os << prefix << "streamLabel         " << streamLabel_           << "\n";
  os << prefix << "maxSize             " << maxSize_               << "\n";
  os << prefix << "highWaterMark       " << highWaterMark_         << "\n";
  os << prefix << "lumiSectionTimeOut  " << lumiSectionTimeOut_    << "\n";
  os << prefix << "no. active files    " << outputMap_.size()      << "\n";
  os << prefix << "no. files           " << outputSummary_.size()  << "\n";
  os << prefix << "-----------------------------------------\n";
}
