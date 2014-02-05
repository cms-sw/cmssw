#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/types.h>
#include <sys/file.h>
#include <unistd.h>
#include <vector>
#include <fstream>
#include <zlib.h>
#include <stdio.h>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem/fstream.hpp>

#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/EventID.h"

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "EventFilter/FEDInterface/interface/GlobalEventNumber.h"
#include "EventFilter/FEDInterface/interface/fed_header.h"
#include "EventFilter/FEDInterface/interface/fed_trailer.h"

#include "EventFilter/Utilities/plugins/FedRawDataInputSource.h"
#include "EventFilter/Utilities/plugins/FastMonitoringService.h"

#include "EventFilter/Utilities/interface/DataPointDefinition.h"

//JSON file reader
#include "EventFilter/Utilities/interface/reader.h"

#include <boost/lexical_cast.hpp>

using namespace jsoncollector;

FedRawDataInputSource::FedRawDataInputSource(edm::ParameterSet const& pset,
                                             edm::InputSourceDescription const& desc) :
  edm::RawInputSource(pset, desc),
  defPath_(pset.getUntrackedParameter<std::string> ("buDefPath", "$CMSSW_BASE/src/EventFilter/Utilities/plugins/budef.jsd")),
  eventChunkSize_(pset.getUntrackedParameter<unsigned int> ("eventChunkSize",16)*1048576),
  eventChunkBlock_(pset.getUntrackedParameter<unsigned int> ("eventChunkBlock",eventChunkSize_/1048576)*1048576),
  numConcurrentReads_(pset.getUntrackedParameter<unsigned int> ("numConcurrentReads",1)),
  deleteFileAfterRead_(pset.getUntrackedParameter<bool> ("deleteFileAfterRead",true)),//TODO
  getLSFromFilename_(pset.getUntrackedParameter<bool> ("getLSFromFilename", true)),
  verifyAdler32_(pset.getUntrackedParameter<bool> ("verifyAdler32", true)),
  testModeNoBuilderUnit_(edm::Service<evf::EvFDaqDirector>()->getTestModeNoBuilderUnit()),
  runNumber_(edm::Service<evf::EvFDaqDirector>()->getRunNumber()),
  buInputDir_(edm::Service<evf::EvFDaqDirector>()->buBaseDir()),
  fuOutputDir_(edm::Service<evf::EvFDaqDirector>()->fuBaseDir()),
  daqProvenanceHelper_(edm::TypeID(typeid(FEDRawDataCollection))),
  eventID_(),
  currentLumiSection_(0),
  currentInputJson_(""),
  currentInputEventCount_(0),
  //eorFileSeen_(false),
  //dataBuffer_(new unsigned char[eventChunkSize_]),//!
  bufferCursor_(dataBuffer_),
  bufferLeft_(0),
  dpd_(nullptr),
  eventsThisLumi_(0)
{
  char thishost[256];
  gethostname(thishost, 255);
  edm::LogInfo("FedRawDataInputSource") << "test mode: "
                                        << testModeNoBuilderUnit_ << ", read-ahead chunk size: " << (eventChunkSize_/1048576)
                                        << " MB on host " << thishost;

  daqProvenanceHelper_.daqInit(productRegistryUpdate(), processHistoryRegistryForUpdate());
  setNewRun();
  setRunAuxiliary(new edm::RunAuxiliary(runNumber_, edm::Timestamp::beginOfTime(),
					edm::Timestamp::invalidTimestamp()));

  dpd_ = new DataPointDefinition();
  DataPointDefinition::getDataPointDefinitionFor(defPath_, *dpd_);

  //make sure that chunk size is N * block size
  assert(eventChunkSize_>=eventChunkBlock_);
  readBlocks_ = eventChunkSize_/eventChunkBlock_;
  if (readBlocks_*eventChunkBlock_ != eventChunkSize_)
    eventChunkSize_=readBlocks_*eventChunkBlock_;

  if (!numConcurrentReads_)
   throw cms::Exception("FedRawDataInputSource::FedRawDataInputSource") <<
	           "no readers enabled in numConcurrentReads parameter";

  for (unsigned int i=0;i<numConcurrentReads_;i++) {
    std::atomic<bool> threadStarted = false;
    workerThreads_.push(new std::thread(&FedRawDataInputSource::readWorker,this,i,threadStarted));
    workerJob_.push_back(ReaderInfo(nullptr,nullptr));
    cvReader_.emplace_back();
    while (!threadStarted) {}//wait until they populate tids
  }

  //todo: keep it for later deleting
  for (unsigned int i=0;i<numConcurrentReads_+1;i++) {
    freeChunks_.push(new InputChunk(i,eventChunkSize_));
  }

  //this thread opens new files and dispatches reading to worker readers
  std::unique_ptr<std::thread>(new std::thread(&FedRawDataInputSource::readSupervisor,this));
}

FedRawDataInputSource::~FedRawDataInputSource()
{
  if (fileDescriptor_!=-1)
    close(fileDescriptor_);
  fileDescriptor_=-1;
}

bool FedRawDataInputSource::checkNextEvent()
{
  switch (cacheNextEvent() ) {
    case evf::EvFDaqDirector::runEnded: {
      resetLuminosityBlockAuxiliary();
      closeCurrentFile();
      return false;
    }
    /* not possible
    case evf::EvFDaqDirector::noFile: {
      edm::LogInfo("FedRawDataInputSource") << "No rawdata files at this time";
      return true;
    }
    */
    case evf::EvFDaqDirector::newLumi: {
      //open new lumi here!
      edm::LogInfo("FedRawDataInputSource") << "New lumisection was detected: " << currentLumiSection_;
      return true;
    }
    default: {
      if (!getLSFromFilename_) {
        //get new lumi from file header
        maybeOpenNewLumiSection( event_->lumi() );
        if (fms_) fms_->reportEventsThisLumiInSource(event_->lumi(),eventsThisLumi_);
      }

      eventID_ = edm::EventID(event_->run(), currentLumiSection_, event_->event());

      setEventCached();

      return true;
    }
  }
}

//TODO: do part of this at supervisor thread, and new lumiblockAux and timestamp later
void FedRawDataInputSource::maybeOpenNewLumiSection(const uint32_t lumiSection)
{
  if (!luminosityBlockAuxiliary()
    || luminosityBlockAuxiliary()->luminosityBlock() != lumiSection) {

    if ( currentLumiSection_ > 0 ) {
      const std::string fuEoLS =
        edm::Service<evf::EvFDaqDirector>()->getEoLSFilePathOnFU(currentLumiSection_);
      struct stat buf;
      bool found = (stat(fuEoLS.c_str(), &buf) == 0);
      if ( !found ) {
        int eol_fd = open(fuEoLS.c_str(), O_RDWR|O_CREAT, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH);
        close(eol_fd);
      }
    }

    currentLumiSection_ = lumiSection;

    resetLuminosityBlockAuxiliary();

    timeval tv;
    gettimeofday(&tv, 0);
    const edm::Timestamp lsopentime( (unsigned long long) tv.tv_sec * 1000000 + (unsigned long long) tv.tv_usec );

    edm::LuminosityBlockAuxiliary* luminosityBlockAuxiliary =
      new edm::LuminosityBlockAuxiliary(
        runAuxiliary()->run(),
        lumiSection, lsopentime,
        edm::Timestamp::invalidTimestamp());

    setLuminosityBlockAuxiliary(luminosityBlockAuxiliary);
  }
}

//TODO: handle new buffers
evf::EvFDaqDirector::FileStatus FedRawDataInputSource::cacheNextEvent()
{
  const size_t headerSize = (4 + 1024) * sizeof(uint32); //minimal size to fit any version of FRDEventHeader
  if ( bufferLeft_ < headerSize )
  {
    const evf::EvFDaqDirector::FileStatus status = readNextChunkIntoBuffer();
    if ( bufferLeft_ == 0 ) return status;
    if ( bufferLeft_ < headerSize )
    {
      throw cms::Exception("FedRawDataInputSource::cacheNextEvent") <<
        "Premature end of input file while reading event header";
    }
  }
  event_.reset( new FRDEventMsgView(bufferCursor_) );

  const uint32_t msgSize = event_->size();
  if ( bufferLeft_ < msgSize )
  {
    if ( readNextChunkIntoBuffer() != evf::EvFDaqDirector::sameFile || bufferLeft_ < msgSize )
    {
      throw cms::Exception("FedRawDataInputSource::cacheNextEvent") <<
        "Premature end of input file while reading event data";
    }
    event_.reset( new FRDEventMsgView(bufferCursor_) );
  }

  if ( verifyAdler32_ && event_->version() >= 3 )
  {
    uint32_t adler = adler32(0L,Z_NULL,0);
    adler = adler32(adler,(Bytef*)event_->payload(),event_->eventSize());

    if ( adler != event_->adler32() ) {
      throw cms::Exception("FedRawDataInputSource::cacheNextEvent") <<
        "Found a wrong Adler32 checksum: expected 0x" << std::hex << event_->adler32() <<
        " but calculated 0x" << adler;
    }
  }

  bufferLeft_ -= msgSize;
  bufferCursor_ += msgSize;

  return evf::EvFDaqDirector::sameFile;
}
//now in worker threads
evf::EvFDaqDirector::FileStatus FedRawDataInputSource::readNextChunkIntoBuffer()
{
  //this function is called when we reach the end of the buffer (i.e. bytes to read are more than bytes left in buffer)
  if (bufferLeft_ == 0) { //in the rare case the last byte barely fit
    for (unsigned int i=0;i<readBlocks_;i++)
    {
      const ssize_t last = ::read(fileDescriptor_,( void*) (dataBuffer_+bufferLeft_), eventChunkBlock_);
      if ( last > 0 )
        bufferLeft_+=last;
    }
  }
  else {
    const uint32_t chunksize = eventChunkSize_ - bufferLeft_;
    const uint32_t blockcount=chunksize/eventChunkBlock_;
    const uint32_t leftsize = chunksize%eventChunkBlock_;
    memcpy((void*) dataBuffer_, bufferCursor_, bufferLeft_);

    for (uint32_t i=0;i<blockcount;i++) {
      const ssize_t last = ::read(fileDescriptor_,( void*) (dataBuffer_+bufferLeft_), eventChunkBlock_);
      if ( last > 0 )
        bufferLeft_ += last;
    }
    if (leftsize) {
      const ssize_t last = ::read(fileDescriptor_,( void*)( dataBuffer_+bufferLeft_), leftsize);
      if ( last > 0 )
        bufferLeft_+=last;
    }
  }

  if (bufferLeft_ == 0) { // no more data in this file
    closeCurrentFile();

    const evf::EvFDaqDirector::FileStatus status = openNextFile();
    if ( status == evf::EvFDaqDirector::newFile ) {
      for (unsigned int i=0;i<readBlocks_;i++)
      {
        const uint32_t last = ::read(fileDescriptor_,( void*) (dataBuffer_+bufferLeft_), eventChunkBlock_);
        bufferLeft_+=last;
      }
    }
    else {
      return status;
    }
  }
  bufferCursor_ = dataBuffer_; // reset the cursor at the beginning of the buffer
  return evf::EvFDaqDirector::sameFile;
}

//now in supervisor
/*
 * should be called by the main thread after file is fully processed, maybe delete it after buffering
 * */
void FedRawDataInputSource::closeCurrentFile(std::string& fileName)
{
//  if (fileDescriptor_!=-1) {

    edm::LogInfo("FedRawDataInputSource") << "Closing input file " << openFile_.string();

//    close(fileDescriptor_);
//    fileDescriptor_=-1;

    if (!testModeNoBuilderUnit_) {
      boost::filesystem::remove(fileName); // won't work in case of forked children
    } else {
      renameToNextFree(fileName);
    }
//  }
}

//now in workerSupervisor
/*obsolete
evf::EvFDaqDirector::FileStatus FedRawDataInputSource::openNextFile()
{
  evf::EvFDaqDirector::FileStatus status;

  while( (status = searchForNextFile()) == evf::EvFDaqDirector::noFile ) {
    edm::LogInfo("FedRawDataInputSource") << "No file for me... sleep and try again..." << std::endl;
    usleep(100000);
  }
  return status;
}
*/

/*OK*/
void FedRawDataInputSource::read(edm::EventPrincipal& eventPrincipal)
{
  if (!currentInputEventCount_) {
    throw cms::Exception("RuntimeError")  << "There are more events than advertised in the input JSON:"
                                          << currentInputJson_.string();
  }

  currentInputEventCount_--;//TODO
  std::auto_ptr<FEDRawDataCollection> rawData(new FEDRawDataCollection);
  edm::Timestamp tstamp = fillFEDRawDataCollection(rawData);

  edm::EventAuxiliary aux(eventID_, processGUID(), tstamp, true,
                          edm::EventAuxiliary::PhysicsTrigger);
  makeEvent(eventPrincipal, aux);

  edm::WrapperOwningHolder edp(new edm::Wrapper<FEDRawDataCollection>(rawData),
                               edm::Wrapper<FEDRawDataCollection>::getInterface());

  eventPrincipal.put(daqProvenanceHelper_.constBranchDescription_, edp,
                     daqProvenanceHelper_.dummyProvenance_);

  return;
}

/*OK*/
edm::Timestamp FedRawDataInputSource::fillFEDRawDataCollection(std::auto_ptr<FEDRawDataCollection>& rawData) const
{
  edm::Timestamp tstamp;
  uint32_t eventSize = event_->eventSize();
  char* event = (char*)event_->payload();

  while (eventSize > 0) {
    eventSize -= sizeof(fedt_t);
    const fedt_t* fedTrailer = (fedt_t*) (event + eventSize);
    const uint32_t fedSize = FED_EVSZ_EXTRACT(fedTrailer->eventsize) << 3; //trailer length counts in 8 bytes
    eventSize -= (fedSize - sizeof(fedh_t));
    const fedh_t* fedHeader = (fedh_t *) (event + eventSize);
    const uint16_t fedId = FED_SOID_EXTRACT(fedHeader->sourceid);
    if (fedId == FEDNumbering::MINTriggerGTPFEDID) {
      evf::evtn::evm_board_setformat(fedSize);
      const uint64_t gpsl = evf::evtn::getgpslow((unsigned char*) fedHeader);
      const uint64_t gpsh = evf::evtn::getgpshigh((unsigned char*) fedHeader);
      tstamp = edm::Timestamp(static_cast<edm::TimeValue_t> ((gpsh << 32) + gpsl));
    }
    FEDRawData& fedData = rawData->FEDData(fedId);
    fedData.resize(fedSize);
    memcpy(fedData.data(), event + eventSize, fedSize);
  }
  assert(eventSize == 0);

  return tstamp;
}

/* replaces
evf::EvFDaqDirector::FileStatus FedRawDataInputSource::searchForNextFile()
{
  if(currentInputEventCount_!=0){
    throw cms::Exception("RuntimeError") << "Went to search for next file but according to BU more events in "
                                         << currentInputJson_.string();
  }

  std::string nextFile;
  uint32_t ls;

  edm::LogInfo("FedRawDataInputSource") << "Asking for next file... to the DaqDirector";

  if (!fms_) {
    try {
       fms_ = (evf::FastMonitoringService *) (edm::Service<evf::MicroStateService>().operator->());
    } catch (...){
      edm::LogWarning("FedRawDataInputSource") << "FastMonitoringService not found";
    }
  }

  if (fms_) fms_->startedLookingForFile();

  evf::EvFDaqDirector::FileStatus status =
    edm::Service<evf::EvFDaqDirector>()->updateFuLock(ls,nextFile);
  if ( status == evf::EvFDaqDirector::newFile ) {
    edm::LogInfo("FedRawDataInputSource") << "The director says to grab: " << nextFile;

    if (fms_) fms_->stoppedLookingForFile(ls);

    boost::filesystem::path jsonFile(nextFile);
    jsonFile.replace_extension(".jsn");
    assert( grabNextJsonFile(jsonFile) );
    openDataFile(nextFile);
  }
  while( getLSFromFilename_ && ls > currentLumiSection_ ) {
    maybeOpenNewLumiSection(ls);
    status = evf::EvFDaqDirector::newLumi;
  }
  if (fms_) fms_->reportEventsThisLumiInSource(ls,eventsThisLumi_);
  eventsThisLumi_=0;

  return status;
}

*/

/* OK */
int FedRawDataInputSource::grabNextJsonFile(boost::filesystem::path const& jsonSourcePath)
{
  std::string data;
  try {
    // assemble json destination path
    boost::filesystem::path jsonDestPath(fuOutputDir_);

    std::ostringstream fileNameWithPID;
    fileNameWithPID << jsonSourcePath.stem().string() << "_pid"
                    << std::setfill('0') << std::setw(5) << getpid() << ".jsn";
    const boost::filesystem::path filePathWithPID(fileNameWithPID.str());
    jsonDestPath /= fileNameWithPID.str();

    edm::LogInfo("FedRawDataInputSource") << " JSON rename " << jsonSourcePath << " to "
                                          << jsonDestPath;

    if ( testModeNoBuilderUnit_ )
      boost::filesystem::copy(jsonSourcePath,jsonDestPath);
    else {
      //boost::filesystem::rename(jsonSourcePath,jsonDestPath);
      boost::filesystem::copy(jsonSourcePath,jsonDestPath);
      boost::filesystem::remove(jsonSourcePath);
    }

    currentInputJson_ = jsonDestPath; // store location for later deletion.
    boost::filesystem::ifstream ij(jsonDestPath);
    Json::Value deserializeRoot;
    Json::Reader reader;

    if (!reader.parse(ij, deserializeRoot))
      throw std::runtime_error("Cannot deserialize input JSON file");

    //read BU JSON
    std::string data;
    DataPoint dp;
    dp.deserialize(deserializeRoot);
    bool success = false;
    for (unsigned int i=0;i<dpd_->getNames().size();i++) {
      if (dpd_->getNames().at(i)=="NEvents")
	if (i<dp.getData().size()) {
	  data = dp.getData()[i];
	  success=true;
	}
    }
    if (!success) {
      if (dp.getData().size())
	data = dp.getData()[0];
      else
	throw cms::Exception("FedRawDataInputSource::grabNextJsonFile") <<
	  " error reading number of events from BU JSON: No input value" << data;
    }
    //    eventsThisLumi_=currentInputEventCount_; //! later (TODO)
    return boost::lexical_cast<int>(data);
  }

  catch (const boost::filesystem::filesystem_error& ex)
  {
    // Input dir gone?
    edm::LogError("FedRawDataInputSource") << " - grabNextFile BOOST FILESYSTEM ERROR CAUGHT: " << ex.what()
                  << " - Maybe the BU run dir disappeared? Ending process with code 0...";
    _exit(0);
  }
  catch (std::runtime_error e)
  {
    // Another process grabbed the file and NFS did not register this
     edm::LogError("FedRawDataInputSource") << " - grabNextFile runtime Exception: " << e.what() << std::endl;
  }
  catch (std::exception e)
  {
    // BU run directory disappeared?
    edm::LogError("FedRawDataInputSource") << " - grabNextFileSOME OTHER EXCEPTION OCCURED!!!! ->" << e.what()
                                           << std::endl;
  }

  catch( boost::bad_lexical_cast const& ) {
    edm::LogError("FedRawDataInputSource") << " error reading number of events from BU JSON. Input value is " << data;
  }

  return -1;
}
/* obsolete
int FedRawDataInputSource::openDataFile(std::string const& nextFile)
*/

//OK (.h)
void FedRawDataInputSource::renameToNextFree(std::string& fileName) const
{
  boost::filesystem::path source(fileName);
  boost::filesystem::path destination( edm::Service<evf::EvFDaqDirector>()->getJumpFilePath() );

  edm::LogInfo("FedRawDataInputSource") << "Instead of delete, RENAME: " << openFile_
                                        << " to: " << destination.string();
  boost::filesystem::rename(source,destination);
  boost::filesystem::rename(source.replace_extension(".jsn"),destination.replace_extension(".jsn"));
}

void FedRawDataInputSource::preForkReleaseResources()
{}

void FedRawDataInputSource::postForkReacquireResources(boost::shared_ptr<edm::multicore::MessageReceiverForSource>)
{
  InputSource::rewind();
  setRunAuxiliary(
                  new edm::RunAuxiliary(runNumber_, edm::Timestamp::beginOfTime(),
					edm::Timestamp::invalidTimestamp()));
}

void FedRawDataInputSource::rewind_()
{}


void FedRawDataInputSource::readSupervisor() {

  currentLumiSection = 0;
  unsigned int eventsThisLumi = 0;
	
  while (!stop_condition) {//TODO


    //this needs to be done when old file is removed from the queue or so
    //  if(currentInputEventCount_!=0){
    //    throw cms::Exception("RuntimeError") << "Went to search for next file but according to BU more events in "
    //                                         << currentInputJson_.string();
    //  }


    //look for at least one free thread and chunk

    while (workerPool_.empty() || freeChunks.empty()) {
      usleep(100000);
    }

    //look for a new file

    std::string nextFile;
    uint32_t ls;
    uint32_t fileSize;

    edm::LogInfo("FedRawDataInputSource") << "Asking for next file... to the DaqDirector";

    if (!fms_) {
      try {
	fms_ = (evf::FastMonitoringService *) (edm::Service<evf::MicroStateService>().operator->());
      } catch (...){
	edm::LogWarning("FedRawDataInputSource") << "FastMonitoringService not found";
      }
    }

    if (fms_) fms_->startedLookingForFile();

    evf::EvFDaqDirector::FileStatus status =  evf::EvFDaqDirector::noFile;

    //try until there is file
    while (status == evf::EvFDaqDirector::noFile) {
      status = edm::Service<evf::EvFDaqDirector>()->updateFuLock(ls,nextFile,fileSize);

      if ( status == evf::EvFDaqDirector::runEnded ) {
	fileQueue_.push(new InputFile(evf::EvFDaqDirector::runEnded));
	quit_workers_ = true;
	//TODO:take lock, signal on all cvs, join...
	usleep(100000);
	    return;
      }
      while( getLSFromFilename_ && ls > currentLumiSection ) {
	currentLumiSection++;
	//TODO:maybe try to read/create EOL file now?
	if (ls==currentLumiSection+1)
	  fileQueue_.push(new InputFile(evf::EvFDaqDirector::newLumi, currrentLumiSection));
      }

      if (status == evf::EvFDaqDirector::noFile) {
	      edm::LogInfo("FedRawDataInputSource") << "No file for me... sleep and try again..." << std::endl;
	      usleep(100000);
      }
    }

    if ( status == evf::EvFDaqDirector::newFile ) {
      edm::LogInfo("FedRawDataInputSource") << "The director says to grab: " << nextFile;

      if (fms_) fms_->stoppedLookingForFile(ls);

      boost::filesystem::path jsonFile(nextFile);
      jsonFile.replace_extension(".jsn");
      int eventsInNewFile = grabNextJsonFile(jsonFile);
      assert( eventsInNewFile>=0 );

//      int fileDescriptor = openDataFile(nextFile);
//      close(fileDescriptor);

      if (fms_) fms_->reportEventsThisLumiInSource(ls,eventsThisLumi_);
      eventsThisLumi_=0;

      // later:
      // maybeOpenNewLumiSection(ls);


      //calculate needed chunks
      unsigned int neededChunks = fileSize/eventChunkSize_;
      if (fileSizee%eventChunkSize_) neededChunks++;


      //queue a file with empty chunk vector
      
      Inputfile * newFile = new InputFile(evf::EvFDaqDirector::FileStatus::newFile,ls,newFile,fileSize,neededChunks,eventsInNewFile);
      fileQueue_.push(newFile);

      for (unsigned int i=0;i<neededChunks;i++) {

	//get thread
        unsigned int newTid;
	while (!workerPool_.try_pop(newTid)) {usleep(100000);}

        InputChunk * newChunk;
        while (!freeChunks.try_pop(newChunk)) {usleep(100000);}

	unsigned int toRead = eventChunkSize_;
	if (i==neededChunks-1 && fileSize%eventChunkSize_) toRead = fileSize%eventChunkSize_;
	newChunk->reset(i*eventChunkSize_,toRead);

        std::unique_lock<std::mutex> lk(m);

        WorkerJob_[newTid].first=newFile;
        WorkerJob_[newTid].second=newChunk;

	ls.unlock();
	cvReader[newTid].notifyOne();

	//add chunk to the file concurrent vector
	newFile->addChunk(newChunk);
      }
    }
  }
}

void FedRawDataInputSource::readWorker(unsigned int tid, std::atomic<bool> &started) {

  bool init = true;

  while (!quit_workers_) {

    workerJob_[newTid].first=nullptr;
    workerJob_[newTid].first=nullptr;

    std::unique_lock<std::mutex> lk(m);
    if (quit_workers_) return;

    //this thread is free: add itself to the available queue
    workerPool_.push(tid);

    if (init) {
      started = true;
      init = false;
    }
    cv[tid].wait(lk);

    InputFile * file;
    InputChunk * chunk;
    //wait until update from supervisor reaches this thread
    while (! file = workerJob_[tid].first) {}
    while (! chunk = workerJob_[tid].second) {}

    int fileDescriptor = open(file->fileName_c_str(), O_RDONLY);
    off_t pos = lseek(fileDescriptor,chunk->offset_,SEEK_SET);

    if (fileDescriptor>=1 && )
      edm::LogInfo("FedRawDataInputSource") << " thread id " << tid << " opened file " << file->fileName_ << " at offset " << pos; 
    }
    else
    {
    //TODO: pipe exception to the main thread
    throw cms::Exception("FedRawDataInputSource::readWorker") <<
      " failed to open file " << file->fileName_ << " fd:" << fileDescriptor <<
      " or seek to offset " << chunk->offset_ << ", lseek returned:" << pos;
    }

    unsigned int bufferLeft = 0;
    for (unsigned int i=0;i<readBlocks_;i++)
    {
      const ssize_t last = ::read(fileDescriptor,( void*) (chunk->dataBuffer_+bufferLeft), eventChunkBlock_);
      if ( last > 0 )
	bufferLeft+=last;
      if (last < eventChunkBlock_) {
	//debug check
	assert(chunk->usedSize_==(i-1)*eventChunkBlock_+last);
	break;
      }
    }
    close(fileDescriptor);
  }
}


// define this class as an input source
DEFINE_FWK_INPUT_SOURCE( FedRawDataInputSource);

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
