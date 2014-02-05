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
  std::unique_ptr<std::thread> readSupervisorThread_.reset(new std::thread(&FedRawDataInputSource::readSupervisor,this));
}

FedRawDataInputSource::~FedRawDataInputSource()
{
  quit_threads_=true;
  readSupervisorThread_.join();
  for (unsigned int i=0;i<numConcurrentReads_+1;i++) {
    InputChunk *ch;
    while (!freeChunks_.try_pop(ch)) {}
    delete ch;
  }
}

bool FedRawDataInputSource::checkNextEvent()
{
  switch (cacheNextEvent() ) {
    case evf::EvFDaqDirector::runEnded: {
      resetLuminosityBlockAuxiliary();
      //TODO see when to delete it
      closeCurrentFile();
      return false;
    }
    case evf::EvFDaqDirector::noFile: {
      return true;
    }
    case evf::EvFDaqDirector::newLumi: {
      edm::LogInfo("FedRawDataInputSource") << "New lumisection was detected: " << currentLumiSection_;
      return true;
    }
    default: {
      if (!getLSFromFilename_) {
        //get new lumi from file header
        maybeOpenNewLumiSection( event_->lumi() );//TODO:put this also in a loop to create all intermediate EOL files
        if (fms_) fms_->reportEventsThisLumiInSource(event_->lumi(),eventsThisLumi_);//?????
      }

      eventID_ = edm::EventID(event_->run(), currentLumiSection_, event_->event());

      setEventCached();

      return true;
    }
  }
}

//not touching this
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

evf::EvFDaqDirector::FileStatus FedRawDataInputSource::cacheNextEvent()
{
  const size_t headerSize = (4 + 1024) * sizeof(uint32); //minimal size to fit any version of FRDEventHeader
  evf::EvFDaqDirector::FileStatus status;

  if (setExceptionState_) threadError(); 
  if (!currentFile_)
  {
    if (!fileQueue_.try_pop(currentFile_))
    {
      edm::LogInfo("FedRawDataInputSource") << "No rawdata files at this time";
      usleep(10000);
      return evf::EvFDaqDirector::noFile;
      //TODO act on signals that interrupt CMSSW
    }
    if ( status = currentFile_->status_ == evf::EvFDaqDirector::runEnded)
    {
      delete currentFile_;
      currentFile_=nullptr;
      return status;
    }

    if (status = currentFile_->status_ == evf::EvFDaqDirector::newLumi) 
    {
      if (getLSFromFilename_) {
	while (currentFile_->lumi_ > currentLumiSection_)  //TODO:check if we always get this from supervisor
      	  maybeOpenNewLumiSection(currentLumiSection_+1);
	 //TODO: count to FMS
      }
      else 
        status = evf::EvFDaqDirector::noFile;
      delete currentFile_;
      currentFile_=nullptr;
      return status;
    }
  }

  assert(status!=evf::EvFDaqDirector::noFile);//shouldn't happen here

  //file is empty
  if (!currentFile_->fileSize_) {
    //empty file: try to open new lumi only
    assert(currentFile_->nChunks_==0);
    if (getLSFromFilename_)
      maybeOpenNewLumisection(currentFile_->lumi_);
    if (!deleteFileAfterRead_) {
      closeCurrentFile(currentFile_->fileName_);
    }
    delete currentFile_;
    currentfile_=nullptr;
    return evf::EvFDaqDirector::noFile;
  }

  //file is finished
  if (currentFile_->bufferPosition_==currentFile_->fileSize_) {
    freeChunks_.push(currentFile_->chunks_[currentFile_->currentChunk_]);//release last chunk
    if (!deleteFileAfterRead_) {
      closeCurrentFile(currentFile_->fileName_);
    }
    if (currentFile_->nEvents_!=currentFile_->nProcessed_)
    {
      throw cms::Exception("RuntimeError") 
	<< "Fully processed " << currentFile_->nProcessed_ 
        << " from the file " << currentFile_->fileName_ 
	<< " but according to BU JSON there should be " 
	<< currentFile_->nEvents_ << " events";

    }
    delete currentFile_;
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
  unsigned char *dataPosition = currentFile_->chunks_[currentFile_->currentChunk_]->buf_+chunkPosition_;
  if (eventChunkSize_ - currentFile_->chunkPosition_ < headerSize) {

      //we need next chunk
      while (!currentFile_->waitForChunk(currentFile_->currentChunk_+1)) {
	      usleep(100000);
	      if (setExceptionState_) threadError(); 
      }

      if (currentFile_->chunks_[currentFile_->currentChunk_]->index_ +1 
		      != currentFile_->chunks_[currentFile_->currentChunk_+1]->index_)
      {
	dataPosition_-=chunkPosition_;
	memcpy(currentFile_->chunks_[currentFile_->currentChunk_]->buf_,
	       currentFile_->chunks_[currentFile_->currentChunk_]->buf_+chunkPosition_,
	       eventChunkSize_-currentFile_->chunkPosition_);
	memcpy(currentFile_->chunks_[currentFile_->currentChunk_]->buf_+chunkPosition_,
	       currentFile_->chunks_[currentFile_->currentChunk_+1]->buf_,
	       headerSize - (eventChunkSize_ - currentFile_->chunkPosition_));
	//move buffer pointers
      }
	chunkPosition_=headerSize - (eventChunkSize_ - currentFile_->chunkPosition_);
	chunkIsFree_=true;
	currentChunk_++;
	bufferPosition_+=headerSize;
  }
  else {
    chunkPosition_+=headerSize;
    bufferPosition_+=headerSize;
  }

  event_.reset( new FRDEventMsgView(dataPosition) );

  //release chunk
  if (chunkIsFree) freeChunks_.push(currentFile_->chunks_[currentFile_->currentChunk_-1]);
  chunkIsFree_=false;

  const uint32_t msgSize = event_->size();

  //file is too short
  if (currentFile_->fileSize_ - currentFile_->bufferPosition_ < msgSize)
  {
      throw cms::Exception("FedRawDataInputSource::cacheNextEvent") <<
        "Premature end of input file while reading event data";
  }

  while (!currentFile_->waitForChunk(currentFile_->currentChunk_)) {
	  usleep(100000);
	  if (setExceptionState_) threadError(); 
  }

//if event payload is found in two chunks (this overwrites the header, but we have read it already)
  unsigned char *dataPosition = currentFile_->chunks_[currentFile_->currentChunk_]->buf_+chunkPosition_;
  if (eventChunkSize_ - currentFile_->chunkPosition_ < msgSize)
  {
      while (!currentFile_->waitForChunk(currentFile_->currentChunk_+1)) {
	      usleep(100000);
	  if (setExceptionState_) threadError(); 
      }

      if (currentFile_->chunks_[currentFile_->currentChunk_]->index_ +1 
		      != currentFile_->chunks_[currentFile_->currentChunk_+1]->index_)
      {
	dataPosition_-=chunkPosition_;
	memcpy(currentFile_->chunks_[currentFile_->currentChunk_]->buf_,
	       currentFile_->chunks_[currentFile_->currentChunk_]->buf_+chunkPosition_,
	       eventChunkSize_-currentFile_->chunkPosition_);
	memcpy(currentFile_->chunks_[currentFile_->currentChunk_]->buf_+chunkPosition_,
	       currentFile_->chunks_[currentFile_->currentChunk_+1]->buf_,
	       msgSize - (eventChunkSize_ - currentFile_->chunkPosition_));
	//move buffer pointers
      }
	chunkPosition_=msgSize - (eventChunkSize_ - currentFile_->chunkPosition_);
	chunkIsFree_++;
	currentChunk_++;
	bufferPosition_+=msgSize;
  }
  else {
    chunkPosition_+=msgSize;
     bufferPosition_+=msgSize;
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
  currentFile_->nProcessed_++;

  return evf::EvFDaqDirector::sameFile;
}

/*
 * should be called by the main thread after file is fully processed, maybe delete it after buffering
 * */
void FedRawDataInputSource::closeCurrentFile(std::string& fileName)
{
    edm::LogInfo("FedRawDataInputSource") << "Closing input file " << fileName.string();

    if (!testModeNoBuilderUnit_) {
      boost::filesystem::remove(fileName); // won't work in case of forked children
    } else {
      renameToNextFree(fileName);
    }
}

//now in workerSupervisor
/*obsolete
evf::EvFDaqDirector::FileStatus FedRawDataInputSource::openNextFile()
{
}
*/

/*OK*/
void FedRawDataInputSource::read(edm::EventPrincipal& eventPrincipal)
{

  std::auto_ptr<FEDRawDataCollection> rawData(new FEDRawDataCollection);
  edm::Timestamp tstamp = fillFEDRawDataCollection(rawData);

  edm::EventAuxiliary aux(eventID_, processGUID(), tstamp, true,
                          edm::EventAuxiliary::PhysicsTrigger);
  makeEvent(eventPrincipal, aux);

  edm::WrapperOwningHolder edp(new edm::Wrapper<FEDRawDataCollection>(rawData),
                               edm::Wrapper<FEDRawDataCollection>::getInterface());

  eventPrincipal.put(daqProvenanceHelper_.constBranchDescription_, edp,
                     daqProvenanceHelper_.dummyProvenance_);

  if (chunkIsFree_) freeChunks_.push(currentFile_->chunks_[currentFile_->currentChunk_-1]);
  chunkIsFree_=false;
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
    edm::LogError("FedRawDataInputSource") << " error parsing number of events from BU JSON. Input value is " << data;
  }

  return -1;
}

//OK
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
  //bool stop_condition
	
  while (!quit_threads_) {

    //wait for at least one free thread and chunk

    while ((workerPool_.empty() || freeChunks_.empty()) && !quit_threads) {
      usleep(100000);
    }
    if (quit_threads_) break;
    purgeOldFiles(false);

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

      if ( status == evf::EvFDaqDirector::runEnded) { //TODO:stop condition
	fileQueue_.push(new InputFile(evf::EvFDaqDirector::runEnded));
	quit_threads_ = true;
	//TODO:handle thread shutdown
	//usleep(100000);
	//purgeOldFiles(true);
	break;
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
	if (quit_threads_) break;
	purgeOldFiles(false);
      }
    }
    if ( status == evf::EvFDaqDirector::newFile ) {
      edm::LogInfo("FedRawDataInputSource") << "The director says to grab: " << nextFile;

      if (fms_) fms_->stoppedLookingForFile(ls);

      boost::filesystem::path jsonFile(nextFile);
      jsonFile.replace_extension(".jsn");
      int eventsInNewFile = grabNextJsonFile(jsonFile);
      assert( eventsInNewFile>=0 );

      //if (fms_) fms_->reportEventsThisLumiInSource(ls,eventsThisLumi_);//TODO
      //eventsThisLumi_=0;

      //calculate needed chunks
      unsigned int neededChunks = fileSize/eventChunkSize_;
      if (fileSizee%eventChunkSize_) neededChunks++;


      //queue a file with empty chunk vector
      CompletitionCounter *cc;
      if (deleteFileAfterRead_ && neededChunks) {
        cc=new CompletitionCounter(newFile, neededChunks);
        openFileTracker_.push(cc);
      }
      
      Inputfile * newInputFile = new InputFile(evf::EvFDaqDirector::FileStatus::newFile,ls,newFile,fileSize,neededChunks,eventsInNewFile,cc);
      fileQueue_.push(newInputFile);

      for (unsigned int i=0;i<neededChunks;i++) {

	//get thread
        unsigned int newTid;
	while (!workerPool_.try_pop(newTid)) {
		usleep(100000);
                if (quit_threads_) break;
	        purgeOldFiles(false);
	}

        InputChunk * newChunk;
        while (!freeChunks_.try_pop(newChunk)) {
		usleep(100000);
                if (quit_threads_) break;
	        purgeOldFiles(false);
	}

        //std::unique_lock<std::mutex> lk(mReader_);//not needed (maybe)

	unsigned int toRead = eventChunkSize_;
	if (i==neededChunks-1 && fileSize%eventChunkSize_) toRead = fileSize%eventChunkSize_;
	newChunk->reset(i*eventChunkSize_,toRead,i);


        WorkerJob_[newTid].first=newInputFile;
        WorkerJob_[newTid].second=newChunk;
	//ls.unlock();//done here so below is guaranteed to be propagated after resetting the chunk;

	//wake up the worker thread
	cvReader[newTid].notifyOne();
      }
    }
  }
  //make sure threads are woken up to shutdown
  std::unique_lock<std::mutex> lk(mReader_);
  for (unsigned int i=0;i<workerThreads_.size();i++) {
    cvReader_[i].notifyOne();
  }
  lk.unlock();
  for (unsigned int i=0;i<workerThreads_.size();i++) {
    workerThreads_[i]->join();
    delete workerThreads_[i];
  }
  purgeOldFiles(true);
}

void FedRawDataInputSource::readWorker(unsigned int tid, std::atomic<bool> &started) {

  bool init = true;

  while (!quit_threads_) {

    workerJob_[newTid].first=nullptr;
    workerJob_[newTid].first=nullptr;

    std::unique_lock<std::mutex> lk(mReader_);
    if (quit_threads_) return;

    //this thread is free: add itself to the available queue
    workerPool_.push(tid);

    if (init) {
      started = true;
      init = false;
    }
    cv[tid].wait(lk);

    if (quit_threads_)  return;

    InputFile * file;
    InputChunk * chunk;

    //condition variable most likely ensures cache coherency for there, but let's leave it here for now
    while (! file = workerJob_[tid].first) {}
    while (! chunk = workerJob_[tid].second) {}
    while (chunk->readComplete_) {}

    CompletitionCounter * cc = file->cc_;

    int fileDescriptor = open(file->fileName_c_str(), O_RDONLY);
    off_t pos = lseek(fileDescriptor,chunk->offset_,SEEK_SET);

    if (fileDescriptor>=1 && )
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
      const ssize_t last = ::read(fileDescriptor,( void*) (chunk->dataBuffer_+bufferLeft), eventChunkBlock_);
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
    file->chunks_[chunk->fileIndex]=chunk;//put the completed chunk

    if (deleteFileAfterRead_) 
      cc->chunksComplete_.fetch_add(1,std::memory_order_release);

  }
}

void FedRawDataInputSource::purgeOldFiles(bool checkAll) {

  if (!deleteFileAfterRead_) return;
  while (!openFileTracker_.empty()) {
	  CompletitionCounter *cc = openFileTracker_.front();
	  if (cc->chunksComplete_==cc->nChunks_) {
                closeCurrentFile(cc->fileName_);
		openFileTracker_.pop();
		delete cc;
	  }
	  else if (!checkAll) break;//do not traverse all if the oldest file is not yet fully buffered
  }
}

void FedRawDataInputSource::threadError() {

  quit_threads_=true;
  throw cms::Exception("FedRawDataInputSource:threadError") << " file reader thread error ";

}
//TODO: fix event counter for reporting to the fastmonitoring service
//      EOL file create (decide what to do with the "maybe" function)

// define this class as an input source
DEFINE_FWK_INPUT_SOURCE( FedRawDataInputSource);

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
