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
  numBuffers_(pset.getUntrackedParameter<unsigned int> ("numBuffers",1)),
  getLSFromFilename_(pset.getUntrackedParameter<bool> ("getLSFromFilename", true)),
  verifyAdler32_(pset.getUntrackedParameter<bool> ("verifyAdler32", true)),
  testModeNoBuilderUnit_(edm::Service<evf::EvFDaqDirector>()->getTestModeNoBuilderUnit()),
  runNumber_(edm::Service<evf::EvFDaqDirector>()->getRunNumber()),
  buInputDir_(edm::Service<evf::EvFDaqDirector>()->buBaseDir()),
  fuOutputDir_(edm::Service<evf::EvFDaqDirector>()->fuBaseDir()),
  daqProvenanceHelper_(edm::TypeID(typeid(FEDRawDataCollection))),
  eventID_(),
  currentLumiSection_(0),
  eventsThisLumi_(0),
  dpd_(nullptr)
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
  DataPointDefinition::getDataPointDefinitionFor(defPath_, dpd_);

  //make sure that chunk size is N * block size
  assert(eventChunkSize_>=eventChunkBlock_);
  readBlocks_ = eventChunkSize_/eventChunkBlock_;
  if (readBlocks_*eventChunkBlock_ != eventChunkSize_)
    eventChunkSize_=readBlocks_*eventChunkBlock_;

  if (!numBuffers_)
   throw cms::Exception("FedRawDataInputSource::FedRawDataInputSource") <<
	           "no reading enabled with numBuffers parameter 0";
  
  numConcurrentReads_=numBuffers_-1;
  singleBufferMode_ = !(numBuffers_>1);

  //het handles to DaqDirector and FastMonitoringService because it isn't acessible in readSupervisor thread

  try {
    fms_ = (evf::FastMonitoringService *) (edm::Service<evf::MicroStateService>().operator->());
  } catch (...){
    edm::LogWarning("FedRawDataInputSource") << "FastMonitoringService not found";
    assert(0);//test
  }

  try {
    daqDirector_ = (evf::EvFDaqDirector *) (edm::Service<evf::EvFDaqDirector>().operator->());
    if (fms_) daqDirector_->setFMS(fms_);
  } catch (...){
    edm::LogWarning("FedRawDataInputSource") << "EvFDaqDirector not found";
    assert(0);//test
  }

  if (!singleBufferMode_)
  {
    for (unsigned int i=0;i<numConcurrentReads_;i++)
    {
      std::unique_lock<std::mutex> lk(startupLock_);
      //issue a memory fence here and in threads (constructor was segfaulting without this)
      thread_quit_signal.push_back(false);
      workerJob_.push_back(ReaderInfo(nullptr,nullptr));
      cvReader_.push_back(new std::condition_variable);
      atomic_thread_fence(std::memory_order_release);
      workerThreads_.push_back(new std::thread(&FedRawDataInputSource::readWorker,this,i));
      startupCv_.wait(lk);
    }

    //should delete chunks when run stops
    for (unsigned int i=0;i<numBuffers_;i++) {
      freeChunks_.push(new InputChunk(i,eventChunkSize_));
    }

    //this thread opens new files and dispatches reading to worker readers
    readSupervisorThread_.reset(new std::thread(&FedRawDataInputSource::readSupervisor,this));
  }
  else {
    dataBuffer_ = new unsigned char[eventChunkSize_];
  }
}

FedRawDataInputSource::~FedRawDataInputSource()
{
  quit_threads_=true;

  //delete any remaining open files
  for (auto it = filesToDelete_.begin();it!=filesToDelete_.end();it++) {
    deleteFile(it->second->fileName_);
    delete it->second;
  } 
  if (!singleBufferMode_)  {
    readSupervisorThread_->join();
    /*for (unsigned int i=0;i<numConcurrentReads_+1;i++) {
      InputChunk *ch;
      while (!freeChunks_.try_pop(ch)) {}
      delete ch;
    }*/
    for (unsigned int i=0;i<numConcurrentReads_;i++) delete cvReader_[i];
  }
  if (dataBuffer_) delete dataBuffer_;
}

bool FedRawDataInputSource::checkNextEvent()
{
  switch (singleBufferMode_ ? cacheNextEvent() : nextEvent() ) {
    case evf::EvFDaqDirector::runEnded: {
      if (fms_) fms_->reportEventsThisLumiInSource(currentLumiSection_,eventsThisLumi_);
      eventsThisLumi_=0;
      resetLuminosityBlockAuxiliary();
       edm::LogInfo("FedRawDataInputSource") << "----------------RUN ENDED----------------";
      return false;
    }
    case evf::EvFDaqDirector::noFile: {
      //this is not reachable
      return true;
    }
    case evf::EvFDaqDirector::newLumi: {
      edm::LogInfo("FedRawDataInputSource") << "New lumisection was detected: " << currentLumiSection_;
      //std::cout << "--------------NEW LUMI---------------" << std::endl;
      return true;
    }
    default: {
      if (!getLSFromFilename_) {
        //get new lumi from file header
	if (event_->lumi() > currentLumiSection_) {
          if (fms_) fms_->reportEventsThisLumiInSource(currentLumiSection_,eventsThisLumi_);
	  eventsThisLumi_=0;
          maybeOpenNewLumiSection( event_->lumi() );
	}
      }
      eventID_ = edm::EventID(event_->run(), currentLumiSection_, event_->event());

      setEventCached();

      return true;
    }
  }
}

void FedRawDataInputSource::maybeOpenNewLumiSection(const uint32_t lumiSection)
{
  if (!luminosityBlockAuxiliary()
    || luminosityBlockAuxiliary()->luminosityBlock() != lumiSection) {

    if ( currentLumiSection_ > 0 ) {
      const std::string fuEoLS =
        daqDirector_->getEoLSFilePathOnFU(currentLumiSection_);
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

inline evf::EvFDaqDirector::FileStatus FedRawDataInputSource::nextEvent()
{
   evf::EvFDaqDirector::FileStatus status = evf::EvFDaqDirector::noFile;
   while ((status = getNextEvent())==evf::EvFDaqDirector::noFile) {}
   return status;
}

inline evf::EvFDaqDirector::FileStatus FedRawDataInputSource::getNextEvent()
{
  const size_t headerSize = (4 + 1024) * sizeof(uint32); //minimal size to fit any version of FRDEventHeader

  if (setExceptionState_) threadError(); 
  if (!currentFile_)
  {
    if (!streamFileTrackerPtr_) {
      streamFileTrackerPtr_ = daqDirector_->getStreamFileTracker();
      nStreams_ = streamFileTrackerPtr_->size();
      if (nStreams_>10) checkEvery_=nStreams_;
    } 

    evf::EvFDaqDirector::FileStatus status = evf::EvFDaqDirector::noFile;
    if (!fileQueue_.try_pop(currentFile_))
    {
      edm::LogInfo("FedRawDataInputSource") << "No rawdata files at this time";
      usleep(100000);
      return evf::EvFDaqDirector::noFile;
    }
    status = currentFile_->status_;
    if ( status == evf::EvFDaqDirector::runEnded)
    {
      delete currentFile_;
      currentFile_=nullptr;
      return status;
    }

    else if (status == evf::EvFDaqDirector::newLumi) 
    {
      if (getLSFromFilename_) {
	if (currentFile_->lumi_ > currentLumiSection_) {
          if (fms_) fms_->reportEventsThisLumiInSource(currentLumiSection_,eventsThisLumi_);
	  eventsThisLumi_=0;
      	  maybeOpenNewLumiSection(currentFile_->lumi_);
	}
      }
      else {//let this be picked up from next event
        status = evf::EvFDaqDirector::noFile;
      }

      delete currentFile_;
      currentFile_=nullptr;
      return status;
    }
    else if (status == evf::EvFDaqDirector::newFile) {
      currentFileIndex_++;
      daqDirector_->updateFileIndex(currentFileIndex_);
    }
    else
      assert(0);
  }

  //file is empty
  if (!currentFile_->fileSize_) {
    //empty file: try to open new lumi
    assert(currentFile_->nChunks_==0);
    if (getLSFromFilename_)
      if (currentFile_->lumi_ > currentLumiSection_) {
        if (fms_) fms_->reportEventsThisLumiInSource(currentLumiSection_,eventsThisLumi_);
	eventsThisLumi_=0;
        maybeOpenNewLumiSection(currentFile_->lumi_);
      }
    filesToDelete_.push_back(std::pair<int,InputFile*>(currentFileIndex_,currentFile_));
    currentFile_=nullptr;
    return evf::EvFDaqDirector::noFile;
  }

  //file is finished
  if (currentFile_->bufferPosition_==currentFile_->fileSize_) {
    //release last chunk (it is never released elsewhere)
    freeChunks_.push(currentFile_->chunks_[currentFile_->currentChunk_]);
    if (currentFile_->nEvents_!=currentFile_->nProcessed_)
    {
      throw cms::Exception("RuntimeError") 
	<< "Fully processed " << currentFile_->nProcessed_ 
        << " from the file " << currentFile_->fileName_ 
	<< " but according to BU JSON there should be " 
	<< currentFile_->nEvents_ << " events";

    }
    //delete currentFile_;
    filesToDelete_.push_back(std::pair<int,InputFile*>(currentFileIndex_,currentFile_));
    currentFile_=nullptr;
    return evf::EvFDaqDirector::noFile;
  }


  //file is too short
  if (currentFile_->fileSize_ - currentFile_->bufferPosition_ < headerSize)
  {
    throw cms::Exception("FedRawDataInputSource::cacheNextEvent") <<
      "Premature end of input file while reading event header";
  }

  //wait for the current chunk to become added to the vector
  while (!currentFile_->waitForChunk(currentFile_->currentChunk_)) {
    usleep(100000);
    if (setExceptionState_) threadError(); 
  }

  //check if header is at the boundary of two chunks
  chunkIsFree_ = false;
  unsigned char *dataPosition;

  //read header, copy it to a single chunk if necessary
  bool chunkEnd = currentFile_->advance(dataPosition,headerSize);

  event_.reset( new FRDEventMsgView(dataPosition) );
  if (event_->size()>eventChunkSize_) {
    throw cms::Exception("FedRawDataInputSource::nextEvent")
	    << " event id:"<< event_->event()<< " lumi:" << event_->lumi()
	    << " run:" << event_->run() << " of size:" << event_->size() 
	    << " bytes does not fit into a chunk of size:" << eventChunkSize_ << " bytes";
  }

  const uint32_t msgSize = event_->size()-headerSize;

  if (currentFile_->fileSize_ - currentFile_->bufferPosition_ < msgSize)
  {
    throw cms::Exception("FedRawDataInputSource::nextEvent") <<
      "Premature end of input file while reading event data";
  }

  if (chunkEnd) {
    //header was at the chunk boundary, we will have to move payload as well
    currentFile_->moveToPreviousChunk(msgSize,headerSize);
    chunkIsFree_ = true;
  }
  else {
    //header was contiguous, but check if payload fits the chunk
    if (eventChunkSize_ - currentFile_->chunkPosition_ < msgSize) {
      //rewind to header start position
      currentFile_->rewindChunk(headerSize);
      //copy event to a chunk start and move pointers
      chunkEnd = currentFile_->advance(dataPosition,headerSize+msgSize);
      assert(chunkEnd);
      chunkIsFree_=true;
      //header is moved
      event_.reset( new FRDEventMsgView(dataPosition) );
    }
    else {
      //everything is in a single chunk, only move pointers forward
      chunkEnd = currentFile_->advance(dataPosition,msgSize);
      assert(!chunkEnd);
      chunkIsFree_=false;
    }

  }

  if ( verifyAdler32_ && event_->version() >= 3 )
  {
    uint32_t adler = adler32(0L,Z_NULL,0);
    adler = adler32(adler,(Bytef*)event_->payload(),event_->eventSize());

    if ( adler != event_->adler32() ) {
      throw cms::Exception("FedRawDataInputSource::nextEvent") <<
        "Found a wrong Adler32 checksum: expected 0x" << std::hex << event_->adler32() <<
        " but calculated 0x" << adler;
    }
  }
  currentFile_->nProcessed_++;

  return evf::EvFDaqDirector::sameFile;
}

void FedRawDataInputSource::deleteFile(std::string const& fileName)
{
  const boost::filesystem::path filePath(fileName);
  if (!testModeNoBuilderUnit_) {
    edm::LogInfo("FedRawDataInputSource") << "Deleting input file " << fileName;
    boost::filesystem::remove(filePath); // won't work in case of forked children
  } else {
    edm::LogInfo("FedRawDataInputSource") << "Renaming input file " << fileName;
    renameToNextFree(fileName);
  }
}

void FedRawDataInputSource::read(edm::EventPrincipal& eventPrincipal)
{
  if (singleBufferMode_) {
    if (!currentInputEventCount_) {
      throw cms::Exception("RuntimeError")  << "There are more events than advertised in the input JSON:"
	<< currentInputJson_.string();
    }
    currentInputEventCount_--;
  }
  std::auto_ptr<FEDRawDataCollection> rawData(new FEDRawDataCollection);
  edm::Timestamp tstamp = fillFEDRawDataCollection(rawData);

  edm::EventAuxiliary aux(eventID_, processGUID(), tstamp, true,
                          edm::EventAuxiliary::PhysicsTrigger);
  makeEvent(eventPrincipal, aux);

  edm::WrapperOwningHolder edp(new edm::Wrapper<FEDRawDataCollection>(rawData),
                               edm::Wrapper<FEDRawDataCollection>::getInterface());

  eventPrincipal.put(daqProvenanceHelper_.constBranchDescription_, edp,
                     daqProvenanceHelper_.dummyProvenance_);

  eventsThisLumi_++;

  //this old file check runs no more often than every 10 events
  if (!(++eventsThisRun_%(checkEvery_))) {
    //delete files that are not in processing
    auto it = filesToDelete_.begin();
    while (it!=filesToDelete_.end()) {
      bool fileIsBeingProcessed = false;
      for (unsigned int i=0;i<nStreams_;i++) {
	if (it->first == streamFileTrackerPtr_->at(i)) {
		fileIsBeingProcessed = true;
		break;
	}
      }
      if (!fileIsBeingProcessed) {
        deleteFile(it->second->fileName_);
        delete it->second;
	it = filesToDelete_.erase(it);
      }
      else it++;
    }

  }
  if (chunkIsFree_) freeChunks_.push(currentFile_->chunks_[currentFile_->currentChunk_-1]);
  chunkIsFree_=false;
  return;
}

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

  catch( boost::bad_lexical_cast const& ) {
    edm::LogError("FedRawDataInputSource") << " error parsing number of events from BU JSON. Input value is " << data;
  }

  catch (std::exception e)
  {
    // BU run directory disappeared?
    edm::LogError("FedRawDataInputSource") << " - grabNextFileSOME OTHER EXCEPTION OCCURED!!!! ->" << e.what()
                                           << std::endl;
  }

  return -1;
}

void FedRawDataInputSource::renameToNextFree(std::string const& fileName) const
{
  boost::filesystem::path source(fileName);
  boost::filesystem::path destination( daqDirector_->getJumpFilePath() );

  edm::LogInfo("FedRawDataInputSource") << "Instead of delete, RENAME: " << fileName
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


void FedRawDataInputSource::readSupervisor()
{
  bool stop=false;
  unsigned int currentLumiSection = 0;
	
  while (!stop) {

    //wait for at least one free thread and chunk
    int counter=0;
    while ((workerPool_.empty() || freeChunks_.empty())) {
      usleep(100000);
      counter++;
      if (!(counter%10)) edm::LogInfo("FedRawDataInputSource") << " No free chunks or threads..." << std::endl;
      if (quit_threads_) {stop=true;break;}
    }

    //look for a new file
    std::string nextFile;
    uint32_t ls;
    uint32_t fileSize;

    edm::LogInfo("FedRawDataInputSource") << "Asking for next file... to the DaqDirector";

    if (fms_) fms_->startedLookingForFile();

    evf::EvFDaqDirector::FileStatus status =  evf::EvFDaqDirector::noFile;

    while (status == evf::EvFDaqDirector::noFile) {
      if (quit_threads_) {
	stop=true;
	break;
      }
      else
	status = daqDirector_->updateFuLock(ls,nextFile,fileSize);

      if ( status == evf::EvFDaqDirector::runEnded) {
	fileQueue_.push(new InputFile(evf::EvFDaqDirector::runEnded));
	stop=true;
	break;
      }
      if( getLSFromFilename_ && ls > currentLumiSection ) {
	currentLumiSection = ls;
	fileQueue_.push(new InputFile(evf::EvFDaqDirector::newLumi, currentLumiSection));
      }

      int dbgcount =0;
      if (status == evf::EvFDaqDirector::noFile) {
	dbgcount++;
	if (!(dbgcount%10))
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

      //calculate n. of needed chunks
      unsigned int neededChunks = fileSize/eventChunkSize_;
      if (fileSize%eventChunkSize_) neededChunks++;

      InputFile * newInputFile = new InputFile(evf::EvFDaqDirector::FileStatus::newFile,ls,nextFile,fileSize,neededChunks,eventsInNewFile,this);
      fileQueue_.push(newInputFile);

      for (unsigned int i=0;i<neededChunks;i++) {

	//get thread
        unsigned int newTid;
	while (!workerPool_.try_pop(newTid)) {
		usleep(100000);
	}

        InputChunk * newChunk;
        while (!freeChunks_.try_pop(newChunk)) {
		usleep(100000);
	}

        std::unique_lock<std::mutex> lk(mReader_);

	unsigned int toRead = eventChunkSize_;
	if (i==neededChunks-1 && fileSize%eventChunkSize_) toRead = fileSize%eventChunkSize_;
	newChunk->reset(i*eventChunkSize_,toRead,i);

        workerJob_[newTid].first=newInputFile;
        workerJob_[newTid].second=newChunk;

	//wake up the worker thread
	cvReader_[newTid]->notify_one();
      }
    }
  }
  //make sure threads finish reading
  unsigned numFinishedThreads = 0;
  while (numFinishedThreads < workerThreads_.size()) {
    unsigned int tid;
    while (!workerPool_.try_pop(tid)) {usleep(10000);}
    std::unique_lock<std::mutex> lk(mReader_);
    thread_quit_signal[tid]=true;
    cvReader_[tid]->notify_one();
    numFinishedThreads++;
  }
  for (unsigned int i=0;i<workerThreads_.size();i++) {
	  workerThreads_[i]->join();
	  delete workerThreads_[i];
  }
}

void FedRawDataInputSource::readWorker(unsigned int tid)
{
  bool init = true;
  atomic_thread_fence(std::memory_order_acquire);

  while (1) {

    std::unique_lock<std::mutex> lk(mReader_);
    workerJob_[tid].first=nullptr;
    workerJob_[tid].first=nullptr;

    assert(!thread_quit_signal[tid]);//should never get it here
    workerPool_.push(tid);

    if (init) {
      std::unique_lock<std::mutex> lk(startupLock_);
      init = false;
      //lk.unlock();
      startupCv_.notify_one();
    }
    cvReader_[tid]->wait(lk);

    if (thread_quit_signal[tid])  return;

    InputFile * file;
    InputChunk * chunk;

    //leaving this here for now
    int count = 1;//DEBUG
    while (count && ! (file = workerJob_[tid].first)) {count--;}
    while (count && ! (chunk = workerJob_[tid].second)) {count--;}
    while (count && chunk->readComplete_) {count--;}
    assert(count>0);

    int fileDescriptor = open(file->fileName_.c_str(), O_RDONLY);
    off_t pos = lseek(fileDescriptor,chunk->offset_,SEEK_SET);

    if (fileDescriptor>=1)
      edm::LogInfo("FedRawDataInputSource") << " thread id " << tid << " opened file " << file->fileName_ << " at offset " << pos; 
    else
    {
      edm::LogError("FedRawDataInputSource") <<
      "readWorker failed to open file " << file->fileName_ << " fd:" << fileDescriptor <<
      " or seek to offset " << chunk->offset_ << ", lseek returned:" << pos;
      setExceptionState_=true;
      return;

    }

    unsigned int bufferLeft = 0;
    for (unsigned int i=0;i<readBlocks_;i++)
    {
      const ssize_t last = ::read(fileDescriptor,( void*) (chunk->buf_+bufferLeft), eventChunkBlock_);
      if ( last > 0 )
	bufferLeft+=last;
      if (last < eventChunkBlock_) {
	//debug check
	assert(chunk->usedSize_==i*eventChunkBlock_+last);
	break;
      }
    }
    close(fileDescriptor);

    chunk->readComplete_=true;//this is atomic to secure the sequential buffer fill before becoming available for processing)
    file->chunks_[chunk->fileIndex_]=chunk;//put the completed chunk in the file chunk vector at predetermined index

  }
}

void FedRawDataInputSource::threadError()
{
  quit_threads_=true;
  throw cms::Exception("FedRawDataInputSource:threadError") << " file reader thread error ";

}


inline bool FedRawDataInputSource::InputFile::advance(unsigned char* & dataPosition, const size_t size)
{
  //wait for chunk
  while (!waitForChunk(currentChunk_)) {
    usleep(100000);
    if (parent_->exceptionState()) parent_->threadError(); 
  }

  dataPosition = chunks_[currentChunk_]->buf_+ chunkPosition_;
  size_t currentLeft = chunks_[currentChunk_]->size_ - chunkPosition_;
  
  if (currentLeft < size) {

    //we need next chunk
    while (!waitForChunk(currentChunk_+1)) {
      usleep(100000);
      if (parent_->exceptionState()) parent_->threadError(); 
    }
    //copy everything to beginning of the first chunk
    dataPosition-=chunkPosition_;
    assert(dataPosition==chunks_[currentChunk_]->buf_);
    memmove(chunks_[currentChunk_]->buf_, chunks_[currentChunk_]->buf_+chunkPosition_, currentLeft);
    memcpy(chunks_[currentChunk_]->buf_ + currentLeft, chunks_[currentChunk_+1]->buf_, size - currentLeft);
    //set pointers at the end of the old data position
    bufferPosition_+=size;
    chunkPosition_=size-currentLeft;
    currentChunk_++;
    return true;
  }
  else {
    chunkPosition_+=size;
    bufferPosition_+=size;
    return false;
  }
}

inline void FedRawDataInputSource::InputFile::moveToPreviousChunk(const size_t size, const size_t offset)
{
  //this will fail in case of events that are too large
  assert(size < chunks_[currentChunk_]->size_ - chunkPosition_);
  assert(size - offset < chunks_[currentChunk_]->size_);
  memcpy(chunks_[currentChunk_-1]->buf_+offset,chunks_[currentChunk_]->buf_+chunkPosition_,size);
  chunkPosition_+=size;
  bufferPosition_+=size;
}

inline void FedRawDataInputSource::InputFile::rewindChunk(const size_t size) {
  chunkPosition_-=size;
  bufferPosition_-=size;
}

/***
 *
 *
 * SINGLE-BUFFER METHODS
 *
 *
 * **/

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

evf::EvFDaqDirector::FileStatus FedRawDataInputSource::readNextChunkIntoBuffer()
{
  //this function is called when we reach the end of the buffer (i.e. bytes to read are more than bytes left in buffer)
  if (fileDescriptor_>-1) {
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
  }
  else assert(bufferLeft_==0);

  if (bufferLeft_ == 0) { // no more data in this file

    if (fileDescriptor_!=-1)
    {
      std::string fileName = openFile_.string();
      edm::LogInfo("FedRawDataInputSource") << "Closing input file " << fileName;
      close(fileDescriptor_);
      fileDescriptor_=-1;
      //will be deleted after no streams are processing events from this file
      filesToDelete_.push_back(std::pair<int,InputFile*>(currentFileIndex_,new InputFile(fileName)));
    }

    evf::EvFDaqDirector::FileStatus status;
    while( (status = searchForNextFile()) == evf::EvFDaqDirector::noFile )
    {
      edm::LogInfo("FedRawDataInputSource") << "No file for me... sleep and try again..." << std::endl;
      usleep(100000);
    }

    if ( status == evf::EvFDaqDirector::newFile ) {

      //update file index for the later file deletion
      currentFileIndex_++;
      daqDirector_->updateFileIndex(currentFileIndex_);

      if (!streamFileTrackerPtr_) {
	streamFileTrackerPtr_ = daqDirector_->getStreamFileTracker();
	nStreams_ = streamFileTrackerPtr_->size();
	if (nStreams_>10) checkEvery_=nStreams_;
      }

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

evf::EvFDaqDirector::FileStatus FedRawDataInputSource::searchForNextFile()
{
  if(currentInputEventCount_!=0){
    throw cms::Exception("RuntimeError") << "Went to search for next file but according to BU more events in "
      << currentInputJson_.string();
  }

  std::string nextFile;
  uint32_t ls;
  uint32_t fsize;

  edm::LogInfo("FedRawDataInputSource") << "Asking for next file... to the DaqDirector";

  if (fms_) fms_->startedLookingForFile();

  evf::EvFDaqDirector::FileStatus status =
    edm::Service<evf::EvFDaqDirector>()->updateFuLock(ls,nextFile,fsize);
  if ( status == evf::EvFDaqDirector::newFile ) {
    edm::LogInfo("FedRawDataInputSource") << "The director says to grab: " << nextFile;

    if (fms_) fms_->stoppedLookingForFile(ls);

    boost::filesystem::path jsonFile(nextFile);
    jsonFile.replace_extension(".jsn");
    currentInputEventCount_ =  grabNextJsonFile(jsonFile);
    assert( currentInputEventCount_ >=0 );

    //open a new file
    fileDescriptor_ = open(nextFile.c_str(), O_RDONLY);
    if (fileDescriptor_ != -1) {
      openFile_ = nextFile;
      edm::LogInfo("FedRawDataInputSource") << " opened file " << nextFile;
    }
    else
    {
      throw cms::Exception("FedRawDataInputSource::openDataFile") <<
	" failed to open file " << nextFile << " fd:" << fileDescriptor_;
    }
  }
  while( getLSFromFilename_ && ls > currentLumiSection_ ) {
    if (fms_) fms_->reportEventsThisLumiInSource(currentLumiSection_,eventsThisLumi_);
    eventsThisLumi_=0;
    maybeOpenNewLumiSection(ls);
    status = evf::EvFDaqDirector::newLumi;
  }
  return status;
}

// define this class as an input source
DEFINE_FWK_INPUT_SOURCE( FedRawDataInputSource);

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
