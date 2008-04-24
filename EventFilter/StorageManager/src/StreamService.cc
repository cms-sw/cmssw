// $Id: StreamService.cc,v 1.7 2008/04/21 12:15:03 loizides Exp $

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
// *** construct stream service from 
// *** parameter set and init message
//
StreamService::StreamService(ParameterSet const& pset, InitMsgView const& view):
   parameterSet_(pset),
   runNumber_(0),
   lumiSection_(0),
   numberOfFileSystems_(0),
   maxFileSizeInMB_(0),
   maxSize_(0),
   highWaterMark_(0),
   lumiSectionTimeOut_(0),
   ntotal_(0)
{
  saveInitMessage(view);
  initializeSelection(view);
  setStreamParameter();
}


// 
// *** event loop for stream service
//
bool StreamService::nextEvent(EventMsgView const& view)
{
  ProgressMarker::instance()->processing(true);
  if (!acceptEvent(view))
    {
      ProgressMarker::instance()->processing(false);
      return false;
    }
  runNumber_   = view.run();
  lumiSection_ = view.lumi();

  setNotifyFile();

  shared_ptr<OutputService> outputService = getOutputService(view);
  ProgressMarker::instance()->processing(false);
  
  ProgressMarker::instance()->writing(true);
  outputService->writeEvent(view);
  closeTimedOutFiles();
  ProgressMarker::instance()->writing(false);
  return true;
}


//
// *** close all files on stop signal
//
void StreamService::stop()
{
  for (OutputMapIterator it = outputMap_.begin(); it != outputMap_.end(); ) {
    boost::shared_ptr<FileRecord> fd(it->first);
    outputMap_.erase(it++);
    fillOutputSummaryClosed(fd);
  }
  renameNotifyFile();
}


// 
// *** close all output service which have not received an events 
// *** for lumiSectionTimeOut seconds and make a record of the file
// 
void StreamService::closeTimedOutFiles()
{
  double currentTime = getCurrentTime();
  for (OutputMapIterator it = outputMap_.begin(); it != outputMap_.end(); ) {
      if (currentTime - it->second->lastEntry() > lumiSectionTimeOut_) {
        boost::shared_ptr<FileRecord> fd(it->first);
        outputMap_.erase(it++);
        fillOutputSummaryClosed(fd);
      } else 
        ++it;
  }
}


//
// *** find output service in map or return a new one
// *** rule: only one file for each lumi section is output map
//
boost::shared_ptr<OutputService> StreamService::getOutputService(EventMsgView const& view)
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
// *** handle lock file
//
boost::shared_ptr<OutputService> StreamService::newOutputService()
{
  boost::shared_ptr<FileRecord> file = generateFileRecord();
  InitMsgView view(&saved_initmsg_[0]);

  shared_ptr<OutputService> outputService(new OutputService(file, view));
  outputMap_[file] = outputService;

  handleLock(file); 
  return outputService;
}


//
// *** handle lock for this stream
// *** has to be in sync with other streams
//
void StreamService::handleLock(shared_ptr<FileRecord> file)
{
  string lockFileName = currentLockPath_ + ".lock";
  remove(lockFileName.c_str());

  currentLockPath_ = file->filePath();  
  lockFileName = currentLockPath_ + ".lock";

  ofstream *lockFile  = new ofstream(lockFileName.c_str(), ios_base::ate | ios_base::out | ios_base::app);
  delete(lockFile);
}


//
// *** perform checks before writing the event
// *** so far ... check the event will fit into the file 
//
bool StreamService::checkEvent(shared_ptr<FileRecord> file, EventMsgView const& view) const
{
  if (file->fileSize() + static_cast<long long>(view.size()) > maxSize_ && file->events() > 0)
    return false;

  return true;
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
// *** initialize stream selection
// 
void StreamService::initializeSelection(InitMsgView const& initView)
{
  Strings triggerNameList;
  initView.hltTriggerNames(triggerNameList);
  eventSelector_.reset(new EventSelector(parameterSet_.getUntrackedParameter("SelectEvents", ParameterSet()),triggerNameList));
}


//
// *** accept event according to their high level trigger bits
//
bool StreamService::acceptEvent(EventMsgView const& view) 
{
  std::vector<unsigned char> hlt_out;
  hlt_out.resize(1 + (view.hltCount()-1)/4);
  view.hltTriggerBits(&hlt_out[0]);
  int num_paths = view.hltCount();
  bool rc = (eventSelector_->wantAll() || eventSelector_->acceptEvent(&hlt_out[0], num_paths));
  return rc;
}


//
// *** save init message need to open new output service
//
void StreamService::saveInitMessage(InitMsgView const& view)
{
  saved_initmsg_[0] = '\0';
  char* pos           = &saved_initmsg_[0];
  unsigned char* from = view.startAddress();
  unsigned int dsize  = view.size();
  copy(from,from+dsize,pos);
}


//
// *** file private data member from parameter set
//
void StreamService::setStreamParameter()
{
  // some parameters common to streams are given in the XML file
  // these are defaults, actually set at configure time

  streamLabel_        = parameterSet_.getParameter<string> ("streamLabel");
  maxSize_ = 1048576 * (long long) parameterSet_.getParameter<int> ("maxSize");
  fileName_           = ""; // set by setFileName
  filePath_           = ""; // set by setFilePath
  mailboxPath_        = ""; // set by setMathBoxPath
  setupLabel_         = ""; // set by setSetupLabel
  highWaterMark_      = 0.9;// set by setHighWaterMark
  lumiSectionTimeOut_ = 10; // set by setLumiSectionTimeOut
  sourceId_           = ""; // set by setSourceId
  // report(cout, 4);
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
boost::shared_ptr<FileRecord> StreamService::generateFileRecord()
{
  std::ostringstream oss;   
  oss    << setupLabel_ 
	 << "." << setfill('0') << std::setw(8) << runNumber_ 
	 << "." << setfill('0') << std::setw(4) << lumiSection_
	 << "." << streamLabel_ 
	 << "." << fileName_
         << "." << sourceId_;
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
  fd->setNotifyFile(notifyFileName_ + ".do_not_touch");

  // fd->report(cout, 12);
  return fd;
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


//
// *** report the status of stream service
//
void StreamService::report(ostream &os, int indentation) const
{
  string prefix(indentation, ' ');
  os << "\n";
  os << prefix << "------------- StreamService -------------\n";
  os << prefix << "fileName            " << fileName_              << "\n";
  os << prefix << "filePath            " << filePath_              << "\n";
  os << prefix << "sourceId            " << sourceId_              << "\n";
  os << prefix << "mailboxPath         " << mailboxPath_           << "\n";
  os << prefix << "setupLabel          " << setupLabel_            << "\n";
  os << prefix << "streamLabel         " << streamLabel_           << "\n";
  os << prefix << "maxSize             " << maxSize_               << "\n";
  os << prefix << "highWaterMark       " << highWaterMark_         << "\n";
  os << prefix << "lumiSectionTimeOut  " << lumiSectionTimeOut_    << "\n";
  os << prefix << "no. active files    " << outputMap_.size()      << "\n";
  os << prefix << "no. files           " << outputSummary_.size()  << "\n";
  os << prefix << "-----------------------------------------\n";
}

//
// *** create a filename for notify data
//
string StreamService::createNotifyFile() const
{
  time_t rawtime = time(0);
  tm * ptm;
  ptm = localtime(&rawtime);
  int min = ptm->tm_min / 5; //every 5 minutes a new file

  boost::shared_ptr<stor::Parameter> smp = 
     stor::Configurator::instance()->getParameter();

  ostringstream notfilename;
  notfilename << filePath_ << "/mbox/"
              << smp->host()
              << "-" << smp->smInstance() << "-"
              << setfill('0') << std::setw(4) << ptm->tm_year+1900
              << setfill('0') << std::setw(2) << ptm->tm_mon+1
              << setfill('0') << std::setw(2) << ptm->tm_mday 
              << setfill('0') << std::setw(2) << ptm->tm_hour
              << "-"
              << setfill('0') << std::setw(2) << min;
  return notfilename.str();
}

//
// *** rename notify file
//
void StreamService::renameNotifyFile()
{
  if(notifyFileName_.empty()) return;

  string notfile(notifyFileName_);
  ostringstream osf1;
  osf1 << notfile << ".do_not_touch";
  ostringstream osf2;
  osf2 << notfile << ".notify";
  notifyFileName_.clear();
  
  // test if a file was written
  if(1) {
    struct stat sb;
    int st = stat(osf1.str().c_str(), &sb);
    if (st != 0) return;
  }

  // test if a file was already moved
  if(1) {
    struct stat sb;
    int st = stat(osf2.str().c_str(), &sb);
    if (st == 0) {
       time_t rawtime = time(0);
       osf2.clear();
       osf2 << notfile 
            << "-scat-" 
            << (int) rawtime
            << ".notify";
    }
  }

  int result = rename(osf1.str().c_str(), osf2.str().c_str());
  cout << "renamed : " << result << " " << osf1.str() << " --- " << osf2.str() << endl;
  if (result != 0) {
    throw cms::Exception("StreamService", "renameNotifyFile")
       << "Unable to move " << osf1.str() << " to " << osf2.str()
       << ". Possibly the storage manager mailbox area is full." << std::endl;
  }
}


//
// *** set new notify file for all file descriptors
//
void StreamService::setNotifyFile()
{
  // get a potentially new notify name
  string newNotName(createNotifyFile());
  if (newNotName==notifyFileName_) return; 

  renameNotifyFile();
  notifyFileName_= newNotName;
  for (OutputMapIterator it = outputMap_.begin(), itEnd = outputMap_.end(); it != itEnd; ++it) {
     it->first->setNotifyFile(notifyFileName_ + ".do_not_touch");
  }
}
