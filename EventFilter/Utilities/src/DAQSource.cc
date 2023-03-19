#include <sstream>
#include <unistd.h>
#include <vector>
#include <chrono>
#include <algorithm>

#include "EventFilter/Utilities/interface/DAQSource.h"
#include "EventFilter/Utilities/interface/DAQSourceModels.h"
#include "EventFilter/Utilities/interface/DAQSourceModelsFRD.h"
#include "EventFilter/Utilities/interface/DAQSourceModelsScouting.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/UnixSignalHandlers.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

#include "EventFilter/Utilities/interface/FastMonitoringService.h"
#include "EventFilter/Utilities/interface/DataPointDefinition.h"
#include "EventFilter/Utilities/interface/FFFNamingSchema.h"
#include "EventFilter/Utilities/interface/crc32c.h"

//JSON file reader
#include "EventFilter/Utilities/interface/reader.h"

using namespace evf::FastMonState;

DAQSource::DAQSource(edm::ParameterSet const& pset, edm::InputSourceDescription const& desc)
    : edm::RawInputSource(pset, desc),
      dataModeConfig_(pset.getUntrackedParameter<std::string>("dataMode")),
      eventChunkSize_(uint64_t(pset.getUntrackedParameter<unsigned int>("eventChunkSize")) << 20),
      maxChunkSize_(uint64_t(pset.getUntrackedParameter<unsigned int>("maxChunkSize")) << 20),
      eventChunkBlock_(uint64_t(pset.getUntrackedParameter<unsigned int>("eventChunkBlock")) << 20),
      numBuffers_(pset.getUntrackedParameter<unsigned int>("numBuffers")),
      maxBufferedFiles_(pset.getUntrackedParameter<unsigned int>("maxBufferedFiles")),
      alwaysStartFromFirstLS_(pset.getUntrackedParameter<bool>("alwaysStartFromFirstLS", false)),
      verifyChecksum_(pset.getUntrackedParameter<bool>("verifyChecksum")),
      useL1EventID_(pset.getUntrackedParameter<bool>("useL1EventID")),
      testTCDSFEDRange_(pset.getUntrackedParameter<std::vector<unsigned int>>("testTCDSFEDRange")),
      listFileNames_(pset.getUntrackedParameter<std::vector<std::string>>("fileNames")),
      fileListMode_(pset.getUntrackedParameter<bool>("fileListMode")),
      fileListLoopMode_(pset.getUntrackedParameter<bool>("fileListLoopMode", false)),
      runNumber_(edm::Service<evf::EvFDaqDirector>()->getRunNumber()),
      processHistoryID_(),
      currentLumiSection_(0),
      eventsThisLumi_(0),
      rng_(std::chrono::system_clock::now().time_since_epoch().count()) {
  char thishost[256];
  gethostname(thishost, 255);

  if (maxChunkSize_ == 0)
    maxChunkSize_ = eventChunkSize_;
  else if (maxChunkSize_ < eventChunkSize_)
    throw cms::Exception("DAQSource::DAQSource") << "maxChunkSize must be equal or larger than eventChunkSize";

  if (eventChunkBlock_ == 0)
    eventChunkBlock_ = eventChunkSize_;
  else if (eventChunkBlock_ > eventChunkSize_)
    throw cms::Exception("DAQSource::DAQSource") << "eventChunkBlock must be equal or smaller than eventChunkSize";

  edm::LogInfo("DAQSource") << "Construction. read-ahead chunk size -: " << std::endl
                            << (eventChunkSize_ >> 20) << " MB on host " << thishost << " in mode " << dataModeConfig_;

  uint16_t MINTCDSuTCAFEDID = FEDNumbering::MINTCDSuTCAFEDID;
  uint16_t MAXTCDSuTCAFEDID = FEDNumbering::MAXTCDSuTCAFEDID;

  if (!testTCDSFEDRange_.empty()) {
    if (testTCDSFEDRange_.size() != 2) {
      throw cms::Exception("DAQSource::DAQSource") << "Invalid TCDS Test FED range parameter";
    }
    MINTCDSuTCAFEDID = testTCDSFEDRange_[0];
    MAXTCDSuTCAFEDID = testTCDSFEDRange_[1];
  }

  //load mode class based on parameter
  if (dataModeConfig_ == "FRD") {
    dataMode_.reset(new DataModeFRD(this));
  } else if (dataModeConfig_ == "FRDStriped") {
    dataMode_.reset(new DataModeFRDStriped(this));
  } else
    throw cms::Exception("DAQSource::DAQSource") << "Unknown data mode " << dataModeConfig_;

  daqDirector_ = edm::Service<evf::EvFDaqDirector>().operator->();

  dataMode_->setTCDSSearchRange(MINTCDSuTCAFEDID, MAXTCDSuTCAFEDID);
  dataMode_->setTesting(pset.getUntrackedParameter<bool>("testing", false));

  long autoRunNumber = -1;
  if (fileListMode_) {
    autoRunNumber = initFileList();
    if (!fileListLoopMode_) {
      if (autoRunNumber < 0)
        throw cms::Exception("DAQSource::DAQSource") << "Run number not found from filename";
      //override run number
      runNumber_ = (edm::RunNumber_t)autoRunNumber;
      daqDirector_->overrideRunNumber((unsigned int)autoRunNumber);
    }
  }

  dataMode_->makeDirectoryEntries(daqDirector_->getBUBaseDirs(), daqDirector_->runString());

  auto& daqProvenanceHelpers = dataMode_->makeDaqProvenanceHelpers();
  for (const auto& daqProvenanceHelper : daqProvenanceHelpers)
    processHistoryID_ = daqProvenanceHelper->daqInit(productRegistryUpdate(), processHistoryRegistryForUpdate());
  setNewRun();
  //todo:autodetect from file name (assert if names differ)
  setRunAuxiliary(new edm::RunAuxiliary(runNumber_, edm::Timestamp::beginOfTime(), edm::Timestamp::invalidTimestamp()));

  //make sure that chunk size is N * block size
  assert(eventChunkSize_ >= eventChunkBlock_);
  readBlocks_ = eventChunkSize_ / eventChunkBlock_;
  if (readBlocks_ * eventChunkBlock_ != eventChunkSize_)
    eventChunkSize_ = readBlocks_ * eventChunkBlock_;

  if (!numBuffers_)
    throw cms::Exception("DAQSource::DAQSource") << "no reading enabled with numBuffers parameter 0";

  numConcurrentReads_ = numBuffers_ - 1;
  assert(numBuffers_ > 1);
  readingFilesCount_ = 0;

  if (!crc32c_hw_test())
    edm::LogError("DAQSource::DAQSource") << "Intel crc32c checksum computation unavailable";

  //get handles to DaqDirector and FastMonitoringService because getting them isn't possible in readSupervisor thread
  if (fileListMode_) {
    try {
      fms_ = static_cast<evf::FastMonitoringService*>(edm::Service<evf::MicroStateService>().operator->());
    } catch (cms::Exception const&) {
      edm::LogInfo("DAQSource") << "No FastMonitoringService found in the configuration";
    }
  } else {
    fms_ = static_cast<evf::FastMonitoringService*>(edm::Service<evf::MicroStateService>().operator->());
    if (!fms_) {
      throw cms::Exception("DAQSource") << "FastMonitoringService not found";
    }
  }

  daqDirector_ = edm::Service<evf::EvFDaqDirector>().operator->();
  if (!daqDirector_)
    cms::Exception("DAQSource") << "EvFDaqDirector not found";

  edm::LogInfo("DAQSource") << "EvFDaqDirector/Source configured to use file service";
  //set DaqDirector to delete files in preGlobalEndLumi callback
  daqDirector_->setDeleteTracking(&fileDeleteLock_, &filesToDelete_);
  if (fms_) {
    daqDirector_->setFMS(fms_);
    fms_->setInputSource(this);
    fms_->setInState(inInit);
    fms_->setInStateSup(inInit);
  }
  //should delete chunks when run stops
  for (unsigned int i = 0; i < numBuffers_; i++) {
    freeChunks_.push(new InputChunk(eventChunkSize_));
  }

  quit_threads_ = false;

  for (unsigned int i = 0; i < numConcurrentReads_; i++) {
    std::unique_lock<std::mutex> lk(startupLock_);
    //issue a memory fence here and in threads (constructor was segfaulting without this)
    thread_quit_signal.push_back(false);
    workerJob_.push_back(ReaderInfo(nullptr, nullptr));
    cvReader_.push_back(new std::condition_variable);
    tid_active_.push_back(0);
    threadInit_.store(false, std::memory_order_release);
    workerThreads_.push_back(new std::thread(&DAQSource::readWorker, this, i));
    startupCv_.wait(lk);
  }

  runAuxiliary()->setProcessHistoryID(processHistoryID_);
}

DAQSource::~DAQSource() {
  quit_threads_ = true;

  //delete any remaining open files
  if (!fms_ || !fms_->exceptionDetected()) {
    std::unique_lock<std::mutex> lkw(fileDeleteLock_);
    for (auto it = filesToDelete_.begin(); it != filesToDelete_.end(); it++)
      it->second.reset();
  } else {
    //skip deleting files with exception
    std::unique_lock<std::mutex> lkw(fileDeleteLock_);
    for (auto it = filesToDelete_.begin(); it != filesToDelete_.end(); it++) {
      if (fms_->isExceptionOnData(it->second->lumi_))
        it->second->unsetDeleteFile();
      else
        it->second.reset();
    }
    //disable deleting current file with exception
    if (currentFile_.get())
      if (fms_->isExceptionOnData(currentFile_->lumi_))
        currentFile_->unsetDeleteFile();
  }

  if (startedSupervisorThread_) {
    readSupervisorThread_->join();
  } else {
    //join aux threads in case the supervisor thread was not started
    for (unsigned int i = 0; i < workerThreads_.size(); i++) {
      std::unique_lock<std::mutex> lk(mReader_);
      thread_quit_signal[i] = true;
      cvReader_[i]->notify_one();
      lk.unlock();
      workerThreads_[i]->join();
      delete workerThreads_[i];
    }
  }
  for (unsigned int i = 0; i < numConcurrentReads_; i++)
    delete cvReader_[i];
  /*
  for (unsigned int i=0;i<numConcurrentReads_+1;i++) {
    InputChunk *ch;
    while (!freeChunks_.try_pop(ch)) {}
    delete ch;
  }
  */
}

void DAQSource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("File-based Filter Farm input source for reading raw data from BU ramdisk (unified)");
  desc.addUntracked<std::string>("dataMode", "FRD")->setComment("Data mode: event 'FRD', 'FRSStriped', 'ScoutingRun2'");
  desc.addUntracked<unsigned int>("eventChunkSize", 64)->setComment("Input buffer (chunk) size");
  desc.addUntracked<unsigned int>("maxChunkSize", 0)
      ->setComment("Maximum chunk size allowed if buffer is resized to fit data. If 0 is specifier, use chunk size");
  desc.addUntracked<unsigned int>("eventChunkBlock", 0)
      ->setComment(
          "Block size used in a single file read call (must be smaller or equal to the initial chunk buffer size). If "
          "0 is specified, use chunk size.");

  desc.addUntracked<unsigned int>("numBuffers", 2)->setComment("Number of buffers used for reading input");
  desc.addUntracked<unsigned int>("maxBufferedFiles", 2)
      ->setComment("Maximum number of simultaneously buffered raw files");
  desc.addUntracked<unsigned int>("alwaysStartFromfirstLS", false)
      ->setComment("Force source to start from LS 1 if server provides higher lumisection number");
  desc.addUntracked<bool>("verifyChecksum", true)
      ->setComment("Verify event CRC-32C checksum of FRDv5 and higher or Adler32 with v3 and v4");
  desc.addUntracked<bool>("useL1EventID", false)
      ->setComment("Use L1 event ID from FED header if true or from TCDS FED if false");
  desc.addUntracked<std::vector<unsigned int>>("testTCDSFEDRange", std::vector<unsigned int>())
      ->setComment("[min, max] range to search for TCDS FED ID in test setup");
  desc.addUntracked<bool>("fileListMode", false)
      ->setComment("Use fileNames parameter to directly specify raw files to open");
  desc.addUntracked<std::vector<std::string>>("fileNames", std::vector<std::string>())
      ->setComment("file list used when fileListMode is enabled");
  desc.setAllowAnything();
  descriptions.add("source", desc);
}

edm::RawInputSource::Next DAQSource::checkNext() {
  if (!startedSupervisorThread_) {
    std::unique_lock<std::mutex> lk(startupLock_);

    //this thread opens new files and dispatches reading to worker readers
    readSupervisorThread_ = std::make_unique<std::thread>(&DAQSource::readSupervisor, this);
    startedSupervisorThread_ = true;

    startupCv_.wait(lk);
  }

  //signal hltd to start event accounting
  if (!currentLumiSection_)
    daqDirector_->createProcessingNotificationMaybe();
  setMonState(inWaitInput);

  auto nextEvent = [this]() {
    auto getNextEvent = [this]() {
      //for some models this is always true (if one event is one block)
      if (dataMode_->dataBlockCompleted()) {
        return getNextDataBlock();
      } else {
        return getNextEventFromDataBlock();
      }
    };

    evf::EvFDaqDirector::FileStatus status = evf::EvFDaqDirector::noFile;
    while ((status = getNextEvent()) == evf::EvFDaqDirector::noFile) {
      if (edm::shutdown_flag.load(std::memory_order_relaxed))
        break;
    }
    return status;
  };

  switch (nextEvent()) {
    case evf::EvFDaqDirector::runEnded: {
      //maybe create EoL file in working directory before ending run
      struct stat buf;
      //also create EoR file in FU data directory
      bool eorFound = (stat(daqDirector_->getEoRFilePathOnFU().c_str(), &buf) == 0);
      if (!eorFound) {
        int eor_fd = open(daqDirector_->getEoRFilePathOnFU().c_str(),
                          O_RDWR | O_CREAT,
                          S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH);
        close(eor_fd);
      }
      reportEventsThisLumiInSource(currentLumiSection_, eventsThisLumi_);
      eventsThisLumi_ = 0;
      resetLuminosityBlockAuxiliary();
      edm::LogInfo("DAQSource") << "----------------RUN ENDED----------------";
      return Next::kStop;
    }
    case evf::EvFDaqDirector::noFile: {
      //this is not reachable
      return Next::kEvent;
    }
    case evf::EvFDaqDirector::newLumi: {
      //std::cout << "--------------NEW LUMI---------------" << std::endl;
      return Next::kEvent;
    }
    default: {
      if (fileListMode_ || fileListLoopMode_)
        eventRunNumber_ = runNumber_;
      else
        eventRunNumber_ = dataMode_->run();

      setEventCached();

      return Next::kEvent;
    }
  }
}

void DAQSource::maybeOpenNewLumiSection(const uint32_t lumiSection) {
  if (!luminosityBlockAuxiliary() || luminosityBlockAuxiliary()->luminosityBlock() != lumiSection) {
    currentLumiSection_ = lumiSection;

    resetLuminosityBlockAuxiliary();

    timeval tv;
    gettimeofday(&tv, nullptr);
    const edm::Timestamp lsopentime((unsigned long long)tv.tv_sec * 1000000 + (unsigned long long)tv.tv_usec);

    edm::LuminosityBlockAuxiliary* lumiBlockAuxiliary = new edm::LuminosityBlockAuxiliary(
        runAuxiliary()->run(), lumiSection, lsopentime, edm::Timestamp::invalidTimestamp());

    setLuminosityBlockAuxiliary(lumiBlockAuxiliary);
    luminosityBlockAuxiliary()->setProcessHistoryID(processHistoryID_);

    edm::LogInfo("DAQSource") << "New lumi section was opened. LUMI -: " << lumiSection;
  }
}

evf::EvFDaqDirector::FileStatus DAQSource::getNextEventFromDataBlock() {
  setMonState(inChecksumEvent);

  bool found = dataMode_->nextEventView();
  //file(s) completely parsed
  if (!found) {
    if (dataMode_->dataBlockInitialized()) {
      dataMode_->setDataBlockInitialized(false);
      //roll position to the end of the file to close it
      currentFile_->bufferPosition_ = currentFile_->fileSize_;
    }
    return evf::EvFDaqDirector::noFile;
  }

  if (verifyChecksum_ && !dataMode_->checksumValid()) {
    if (fms_)
      fms_->setExceptionDetected(currentLumiSection_);
    throw cms::Exception("DAQSource::getNextEvent") << dataMode_->getChecksumError();
  }
  setMonState(inCachedEvent);

  currentFile_->nProcessed_++;

  return evf::EvFDaqDirector::sameFile;
}

evf::EvFDaqDirector::FileStatus DAQSource::getNextDataBlock() {
  if (setExceptionState_)
    threadError();
  if (!currentFile_.get()) {
    evf::EvFDaqDirector::FileStatus status = evf::EvFDaqDirector::noFile;
    setMonState(inWaitInput);
    if (!fileQueue_.try_pop(currentFile_)) {
      //sleep until wakeup (only in single-buffer mode) or timeout
      std::unique_lock<std::mutex> lkw(mWakeup_);
      if (cvWakeup_.wait_for(lkw, std::chrono::milliseconds(100)) == std::cv_status::timeout || !currentFile_.get())
        return evf::EvFDaqDirector::noFile;
    }
    status = currentFile_->status_;
    if (status == evf::EvFDaqDirector::runEnded) {
      setMonState(inRunEnd);
      currentFile_.reset();
      return status;
    } else if (status == evf::EvFDaqDirector::runAbort) {
      throw cms::Exception("DAQSource::getNextEvent") << "Run has been aborted by the input source reader thread";
    } else if (status == evf::EvFDaqDirector::newLumi) {
      setMonState(inNewLumi);
      if (currentFile_->lumi_ > currentLumiSection_) {
        reportEventsThisLumiInSource(currentLumiSection_, eventsThisLumi_);
        eventsThisLumi_ = 0;
        maybeOpenNewLumiSection(currentFile_->lumi_);
      }
      currentFile_.reset();
      return status;
    } else if (status == evf::EvFDaqDirector::newFile) {
      currentFileIndex_++;
    } else
      assert(false);
  }
  setMonState(inProcessingFile);

  //file is empty
  if (!currentFile_->fileSize_) {
    readingFilesCount_--;
    //try to open new lumi
    assert(currentFile_->nChunks_ == 0);
    if (currentFile_->lumi_ > currentLumiSection_) {
      reportEventsThisLumiInSource(currentLumiSection_, eventsThisLumi_);
      eventsThisLumi_ = 0;
      maybeOpenNewLumiSection(currentFile_->lumi_);
    }
    //immediately delete empty file
    currentFile_.reset();
    return evf::EvFDaqDirector::noFile;
  }

  //file is finished
  if (currentFile_->bufferPosition_ == currentFile_->fileSize_) {
    readingFilesCount_--;
    //release last chunk (it is never released elsewhere)
    freeChunks_.push(currentFile_->chunks_[currentFile_->currentChunk_]);
    if (currentFile_->nEvents_ >= 0 && currentFile_->nEvents_ != int(currentFile_->nProcessed_)) {
      throw cms::Exception("DAQSource::getNextEvent")
          << "Fully processed " << currentFile_->nProcessed_ << " from the file " << currentFile_->fileName_
          << " but according to BU JSON there should be " << currentFile_->nEvents_ << " events";
    }
    if (!daqDirector_->isSingleStreamThread() && !fileListMode_) {
      //put the file in pending delete list;
      std::unique_lock<std::mutex> lkw(fileDeleteLock_);
      filesToDelete_.push_back(
          std::pair<int, std::unique_ptr<RawInputFile>>(currentFileIndex_, std::move(currentFile_)));
    } else {
      //in single-thread and stream jobs, events are already processed
      currentFile_.reset();
    }
    return evf::EvFDaqDirector::noFile;
  }

  //assert(currentFile_->status_ == evf::EvFDaqDirector::newFile);

  //handle RAW file header
  if (currentFile_->bufferPosition_ == 0 && currentFile_->rawHeaderSize_ > 0) {
    if (currentFile_->fileSize_ <= currentFile_->rawHeaderSize_) {
      if (currentFile_->fileSize_ < currentFile_->rawHeaderSize_)
        throw cms::Exception("DAQSource::getNextEvent") << "Premature end of input file while reading file header";

      edm::LogWarning("DAQSource") << "File with only raw header and no events received in LS " << currentFile_->lumi_;
      if (currentFile_->lumi_ > currentLumiSection_) {
        reportEventsThisLumiInSource(currentLumiSection_, eventsThisLumi_);
        eventsThisLumi_ = 0;
        maybeOpenNewLumiSection(currentFile_->lumi_);
      }
    }

    //advance buffer position to skip file header (chunk will be acquired later)
    currentFile_->chunkPosition_ += currentFile_->rawHeaderSize_;
    currentFile_->bufferPosition_ += currentFile_->rawHeaderSize_;
  }

  //file is too short
  if (currentFile_->fileSize_ - currentFile_->bufferPosition_ < dataMode_->headerSize()) {
    throw cms::Exception("DAQSource::getNextEvent") << "Premature end of input file while reading event header";
  }

  //multibuffer mode
  //wait for the current chunk to become added to the vector
  setMonState(inWaitChunk);
  while (!currentFile_->waitForChunk(currentFile_->currentChunk_)) {
    usleep(10000);
    if (setExceptionState_)
      threadError();
  }
  setMonState(inChunkReceived);

  //check if header is at the boundary of two chunks
  chunkIsFree_ = false;
  unsigned char* dataPosition;

  //read event header, copy it to a single chunk if necessary
  bool chunkEnd = currentFile_->advance(dataPosition, dataMode_->headerSize());

  //get buffer size of current chunk (can be resized)
  uint64_t currentChunkSize = currentFile_->currentChunkSize();

  dataMode_->makeDataBlockView(dataPosition, currentChunkSize, currentFile_->fileSizes_, currentFile_->rawHeaderSize_);

  const size_t msgSize = dataMode_->dataBlockSize() - dataMode_->headerSize();

  if (currentFile_->fileSize_ - currentFile_->bufferPosition_ < msgSize) {
    throw cms::Exception("DAQSource::getNextEvent") << "Premature end of input file while reading event data";
  }

  //for cross-buffer models
  if (chunkEnd) {
    //header was at the chunk boundary, we will have to move payload as well
    currentFile_->moveToPreviousChunk(msgSize, dataMode_->headerSize());
    chunkIsFree_ = true;
  } else {
    //header was contiguous, but check if payload fits the chunk
    if (currentChunkSize - currentFile_->chunkPosition_ < msgSize) {
      //rewind to header start position
      currentFile_->rewindChunk(dataMode_->headerSize());
      //copy event to a chunk start and move pointers

      setMonState(inWaitChunk);

      //can already move buffer
      chunkEnd = currentFile_->advance(dataPosition, dataMode_->headerSize() + msgSize);

      setMonState(inChunkReceived);

      assert(chunkEnd);
      chunkIsFree_ = true;
      //header is moved
      dataMode_->makeDataBlockView(
          dataPosition, currentFile_->currentChunkSize(), currentFile_->fileSizes_, currentFile_->rawHeaderSize_);
    } else {
      //everything is in a single chunk, only move pointers forward
      chunkEnd = currentFile_->advance(dataPosition, msgSize);
      assert(!chunkEnd);
      chunkIsFree_ = false;
    }
  }
  //prepare event
  return getNextEventFromDataBlock();
}

void DAQSource::read(edm::EventPrincipal& eventPrincipal) {
  setMonState(inReadEvent);

  dataMode_->readEvent(eventPrincipal);

  eventsThisLumi_++;
  setMonState(inReadCleanup);

  //resize vector if needed
  while (streamFileTracker_.size() <= eventPrincipal.streamID())
    streamFileTracker_.push_back(-1);

  streamFileTracker_[eventPrincipal.streamID()] = currentFileIndex_;

  //this old file check runs no more often than every 10 events
  if (!((currentFile_->nProcessed_ - 1) % (checkEvery_))) {
    //delete files that are not in processing
    std::unique_lock<std::mutex> lkw(fileDeleteLock_);
    auto it = filesToDelete_.begin();
    while (it != filesToDelete_.end()) {
      bool fileIsBeingProcessed = false;
      for (unsigned int i = 0; i < nStreams_; i++) {
        if (it->first == streamFileTracker_.at(i)) {
          fileIsBeingProcessed = true;
          break;
        }
      }
      if (!fileIsBeingProcessed && !(fms_ && fms_->isExceptionOnData(it->second->lumi_))) {
        it = filesToDelete_.erase(it);
      } else
        it++;
    }
  }
  if (dataMode_->dataBlockCompleted() && chunkIsFree_) {
    freeChunks_.push(currentFile_->chunks_[currentFile_->currentChunk_ - 1]);
    chunkIsFree_ = false;
  }
  setMonState(inNoRequest);
  return;
}

void DAQSource::rewind_() {}

void DAQSource::dataArranger() {}

void DAQSource::readSupervisor() {
  bool stop = false;
  unsigned int currentLumiSection = 0;

  {
    std::unique_lock<std::mutex> lk(startupLock_);
    startupCv_.notify_one();
  }

  uint32_t ls = 0;
  uint32_t monLS = 1;
  uint32_t lockCount = 0;
  uint64_t sumLockWaitTimeUs = 0.;

  bool requireHeader = dataMode_->requireHeader();

  while (!stop) {
    //wait for at least one free thread and chunk
    int counter = 0;

    while (workerPool_.empty() || freeChunks_.empty() || readingFilesCount_ >= maxBufferedFiles_) {
      //report state to monitoring
      if (fms_) {
        bool copy_active = false;
        for (auto j : tid_active_)
          if (j)
            copy_active = true;
        if (readingFilesCount_ >= maxBufferedFiles_)
          setMonStateSup(inSupFileLimit);
        else if (freeChunks_.empty()) {
          if (copy_active)
            setMonStateSup(inSupWaitFreeChunkCopying);
          else
            setMonStateSup(inSupWaitFreeChunk);
        } else {
          if (copy_active)
            setMonStateSup(inSupWaitFreeThreadCopying);
          else
            setMonStateSup(inSupWaitFreeThread);
        }
      }
      std::unique_lock<std::mutex> lkw(mWakeup_);
      //sleep until woken up by condition or a timeout
      if (cvWakeup_.wait_for(lkw, std::chrono::milliseconds(100)) == std::cv_status::timeout) {
        counter++;
        //if (!(counter%50)) edm::LogInfo("DAQSource") << "No free chunks or threads...";
        LogDebug("DAQSource") << "No free chunks or threads...";
      } else {
        assert(!workerPool_.empty() || freeChunks_.empty());
      }
      if (quit_threads_.load(std::memory_order_relaxed) || edm::shutdown_flag.load(std::memory_order_relaxed)) {
        stop = true;
        break;
      }
    }
    //if this is reached, there are enough buffers and threads to proceed or processing is instructed to stop

    if (stop)
      break;

    //look for a new file
    std::string nextFile;
    int64_t fileSizeFromMetadata;

    if (fms_) {
      setMonStateSup(inSupBusy);
      fms_->startedLookingForFile();
    }
    bool fitToBuffer = dataMode_->fitToBuffer();

    evf::EvFDaqDirector::FileStatus status = evf::EvFDaqDirector::noFile;
    uint16_t rawHeaderSize = 0;
    uint32_t lsFromRaw = 0;
    int32_t serverEventsInNewFile = -1;
    int rawFd = -1;

    int backoff_exp = 0;

    //entering loop which tries to grab new file from ramdisk
    while (status == evf::EvFDaqDirector::noFile) {
      //check if hltd has signalled to throttle input
      counter = 0;
      while (daqDirector_->inputThrottled()) {
        if (quit_threads_.load(std::memory_order_relaxed) || edm::shutdown_flag.load(std::memory_order_relaxed))
          break;

        unsigned int nConcurrentLumis = daqDirector_->numConcurrentLumis();
        unsigned int nOtherLumis = nConcurrentLumis > 0 ? nConcurrentLumis - 1 : 0;
        unsigned int checkLumiStart = currentLumiSection > nOtherLumis ? currentLumiSection - nOtherLumis : 1;
        bool hasDiscardedLumi = false;
        for (unsigned int i = checkLumiStart; i <= currentLumiSection; i++) {
          if (daqDirector_->lumisectionDiscarded(i)) {
            edm::LogWarning("DAQSource") << "Source detected that the lumisection is discarded -: " << i;
            hasDiscardedLumi = true;
            break;
          }
        }
        if (hasDiscardedLumi)
          break;

        setMonStateSup(inThrottled);
        if (!(counter % 50))
          edm::LogWarning("DAQSource") << "Input throttled detected, reading files is paused...";
        usleep(100000);
        counter++;
      }

      if (quit_threads_.load(std::memory_order_relaxed) || edm::shutdown_flag.load(std::memory_order_relaxed)) {
        stop = true;
        break;
      }

      assert(rawFd == -1);
      uint64_t thisLockWaitTimeUs = 0.;
      setMonStateSup(inSupLockPolling);
      if (fileListMode_) {
        //return LS if LS not set, otherwise return file
        status = getFile(ls, nextFile, thisLockWaitTimeUs);
        if (status == evf::EvFDaqDirector::newFile) {
          uint16_t rawDataType;
          if (evf::EvFDaqDirector::parseFRDFileHeader(nextFile,
                                                      rawFd,
                                                      rawHeaderSize,  ///possibility to use by new formats
                                                      rawDataType,
                                                      lsFromRaw,
                                                      serverEventsInNewFile,
                                                      fileSizeFromMetadata,
                                                      requireHeader,
                                                      false,
                                                      false) != 0) {
            //error
            setExceptionState_ = true;
            stop = true;
            break;
          }
        }
      } else {
        status = daqDirector_->getNextFromFileBroker(currentLumiSection,
                                                     ls,
                                                     nextFile,
                                                     rawFd,
                                                     rawHeaderSize,  //which format?
                                                     serverEventsInNewFile,
                                                     fileSizeFromMetadata,
                                                     thisLockWaitTimeUs,
                                                     requireHeader);
      }

      setMonStateSup(inSupBusy);

      //cycle through all remaining LS even if no files get assigned
      if (currentLumiSection != ls && status == evf::EvFDaqDirector::runEnded)
        status = evf::EvFDaqDirector::noFile;

      //monitoring of lock wait time
      if (thisLockWaitTimeUs > 0.)
        sumLockWaitTimeUs += thisLockWaitTimeUs;
      lockCount++;
      if (ls > monLS) {
        monLS = ls;
        if (lockCount)
          if (fms_)
            fms_->reportLockWait(monLS, sumLockWaitTimeUs, lockCount);
        lockCount = 0;
        sumLockWaitTimeUs = 0;
      }

      if (status == evf::EvFDaqDirector::runEnded) {
        fileQueue_.push(std::make_unique<RawInputFile>(evf::EvFDaqDirector::runEnded));
        stop = true;
        break;
      }

      //error from filelocking function
      if (status == evf::EvFDaqDirector::runAbort) {
        fileQueue_.push(std::make_unique<RawInputFile>(evf::EvFDaqDirector::runAbort, 0));
        stop = true;
        break;
      }
      //queue new lumisection
      if (ls > currentLumiSection) {
        //new file service
        if (currentLumiSection == 0 && !alwaysStartFromFirstLS_) {
          if (daqDirector_->getStartLumisectionFromEnv() > 1) {
            //start transitions from LS specified by env, continue if not reached
            if (ls < daqDirector_->getStartLumisectionFromEnv()) {
              //skip file if from earlier LS than specified by env
              if (rawFd != -1) {
                close(rawFd);
                rawFd = -1;
              }
              status = evf::EvFDaqDirector::noFile;
              continue;
            } else {
              fileQueue_.push(std::make_unique<RawInputFile>(evf::EvFDaqDirector::newLumi, ls));
            }
          } else if (ls < 100) {
            //look at last LS file on disk to start from that lumisection (only within first 100 LS)
            unsigned int lsToStart = daqDirector_->getLumisectionToStart();

            for (unsigned int nextLS = std::min(lsToStart, ls); nextLS <= ls; nextLS++) {
              fileQueue_.push(std::make_unique<RawInputFile>(evf::EvFDaqDirector::newLumi, nextLS));
            }
          } else {
            //start from current LS
            fileQueue_.push(std::make_unique<RawInputFile>(evf::EvFDaqDirector::newLumi, ls));
          }
        } else {
          //queue all lumisections after last one seen to avoid gaps
          for (unsigned int nextLS = currentLumiSection + 1; nextLS <= ls; nextLS++) {
            fileQueue_.push(std::make_unique<RawInputFile>(evf::EvFDaqDirector::newLumi, nextLS));
          }
        }
        currentLumiSection = ls;
      }
      //else
      if (currentLumiSection > 0 && ls < currentLumiSection) {
        edm::LogError("DAQSource") << "Got old LS (" << ls
                                   << ") file from EvFDAQDirector! Expected LS:" << currentLumiSection
                                   << ". Aborting execution." << std::endl;
        if (rawFd != -1)
          close(rawFd);
        rawFd = -1;
        fileQueue_.push(std::make_unique<RawInputFile>(evf::EvFDaqDirector::runAbort, 0));
        stop = true;
        break;
      }

      int dbgcount = 0;
      if (status == evf::EvFDaqDirector::noFile) {
        setMonStateSup(inSupNoFile);
        dbgcount++;
        if (!(dbgcount % 20))
          LogDebug("DAQSource") << "No file for me... sleep and try again...";

        backoff_exp = std::min(4, backoff_exp);  // max 1.6 seconds
        //backoff_exp=0; // disabled!
        int sleeptime = (int)(100000. * pow(2, backoff_exp));
        usleep(sleeptime);
        backoff_exp++;
      } else
        backoff_exp = 0;
    }
    //end of file grab loop, parse result
    if (status == evf::EvFDaqDirector::newFile) {
      setMonStateSup(inSupNewFile);
      LogDebug("DAQSource") << "The director says to grab -: " << nextFile;

      std::string rawFile;
      //file service will report raw extension
      rawFile = nextFile;

      struct stat st;
      int stat_res = stat(rawFile.c_str(), &st);
      if (stat_res == -1) {
        edm::LogError("DAQSource") << "Can not stat file (" << errno << "):-" << rawFile << std::endl;
        setExceptionState_ = true;
        break;
      }
      uint64_t fileSize = st.st_size;

      if (fms_) {
        setMonStateSup(inSupBusy);
        fms_->stoppedLookingForFile(ls);
        setMonStateSup(inSupNewFile);
      }
      int eventsInNewFile;
      if (fileListMode_) {
        if (fileSize == 0)
          eventsInNewFile = 0;
        else
          eventsInNewFile = -1;
      } else {
        eventsInNewFile = serverEventsInNewFile;
        assert(eventsInNewFile >= 0);
        assert((eventsInNewFile > 0) ==
               (fileSize > rawHeaderSize));  //file without events must be empty or contain only header
      }

      std::pair<bool, std::vector<std::string>> additionalFiles =
          dataMode_->defineAdditionalFiles(rawFile, fileListMode_);
      if (!additionalFiles.first) {
        //skip secondary files from file broker
        if (rawFd > -1)
          close(rawFd);
        continue;
      }

      std::unique_ptr<RawInputFile> newInputFile(new RawInputFile(evf::EvFDaqDirector::FileStatus::newFile,
                                                                  ls,
                                                                  rawFile,
                                                                  !fileListMode_,
                                                                  rawFd,
                                                                  fileSize,
                                                                  rawHeaderSize,  //for which format
                                                                  0,
                                                                  eventsInNewFile,
                                                                  this));

      uint64_t neededSize = fileSize;
      for (const auto& addFile : additionalFiles.second) {
        struct stat buf;
        //wait for secondary files to appear
        unsigned int fcnt = 0;
        while (stat(addFile.c_str(), &buf) != 0) {
          if (fileListMode_) {
            edm::LogError("DAQSource") << "additional file is missing -: " << addFile;
            stop = true;
            setExceptionState_ = true;
            break;
          }
          usleep(10000);
          fcnt++;
          //report and EoR check every 30 seconds
          if ((fcnt && fcnt % 3000 == 0) || quit_threads_.load(std::memory_order_relaxed)) {
            edm::LogWarning("DAQSource") << "Additional file is still missing after 30 seconds -: " << addFile;
            struct stat bufEoR;
            auto secondaryPath = std::filesystem::path(addFile).parent_path();
            auto eorName = std::filesystem::path(daqDirector_->getEoRFileName());
            std::string mainEoR = (std::filesystem::path(daqDirector_->buBaseRunDir()) / eorName).generic_string();
            std::string secondaryEoR = (secondaryPath / eorName).generic_string();
            bool prematureEoR = false;
            if (stat(secondaryEoR.c_str(), &bufEoR) == 0) {
              if (stat(addFile.c_str(), &bufEoR) != 0) {
                edm::LogError("DAQSource")
                    << "EoR file appeared in -: " << secondaryPath << " while waiting for index file " << addFile;
                prematureEoR = true;
              }
            } else if (stat(mainEoR.c_str(), &bufEoR) == 0) {
              //wait another 10 seconds
              usleep(10000000);
              if (stat(addFile.c_str(), &bufEoR) != 0) {
                edm::LogError("DAQSource")
                    << "Main EoR file appeared -: " << mainEoR << " while waiting for index file " << addFile;
                prematureEoR = true;
              }
            }
            if (prematureEoR) {
              //queue EoR since this is not FU error
              fileQueue_.push(std::make_unique<RawInputFile>(evf::EvFDaqDirector::runEnded, 0));
              stop = true;
              break;
            }
          }

          if (quit_threads_) {
            edm::LogError("DAQSource") << "Quitting while waiting for file -: " << addFile;
            stop = true;
            setExceptionState_ = true;
            break;
          }
        }
        LogDebug("DAQSource") << " APPEND NAME " << addFile;
        if (stop)
          break;

        newInputFile->appendFile(addFile, buf.st_size);
        neededSize += buf.st_size;
      }
      if (stop)
        break;

      //calculate number of needed chunks and size if resizing will be applied
      uint16_t neededChunks;
      uint64_t chunkSize;

      if (fitToBuffer) {
        chunkSize = std::min(maxChunkSize_, std::max(eventChunkSize_, neededSize));
        neededChunks = 1;
      } else {
        chunkSize = eventChunkSize_;
        neededChunks = neededSize / eventChunkSize_ + uint16_t((neededSize % eventChunkSize_) > 0);
      }
      newInputFile->setChunks(neededChunks);

      newInputFile->randomizeOrder(rng_);

      readingFilesCount_++;
      auto newInputFilePtr = newInputFile.get();
      fileQueue_.push(std::move(newInputFile));

      for (size_t i = 0; i < neededChunks; i++) {
        if (fms_) {
          bool copy_active = false;
          for (auto j : tid_active_)
            if (j)
              copy_active = true;
          if (copy_active)
            setMonStateSup(inSupNewFileWaitThreadCopying);
          else
            setMonStateSup(inSupNewFileWaitThread);
        }
        //get thread
        unsigned int newTid = 0xffffffff;
        while (!workerPool_.try_pop(newTid)) {
          usleep(100000);
          if (quit_threads_.load(std::memory_order_relaxed)) {
            stop = true;
            break;
          }
        }

        if (fms_) {
          bool copy_active = false;
          for (auto j : tid_active_)
            if (j)
              copy_active = true;
          if (copy_active)
            setMonStateSup(inSupNewFileWaitChunkCopying);
          else
            setMonStateSup(inSupNewFileWaitChunk);
        }
        InputChunk* newChunk = nullptr;
        while (!freeChunks_.try_pop(newChunk)) {
          usleep(100000);
          if (quit_threads_.load(std::memory_order_relaxed)) {
            stop = true;
            break;
          }
        }

        if (newChunk == nullptr) {
          //return unused tid if we received shutdown (nullptr chunk)
          if (newTid != 0xffffffff)
            workerPool_.push(newTid);
          stop = true;
          break;
        }
        if (stop)
          break;
        setMonStateSup(inSupNewFile);

        std::unique_lock<std::mutex> lk(mReader_);

        uint64_t toRead = chunkSize;
        if (i == (uint64_t)neededChunks - 1 && neededSize % chunkSize)
          toRead = neededSize % chunkSize;
        newChunk->reset(i * chunkSize, toRead, i);

        workerJob_[newTid].first = newInputFilePtr;
        workerJob_[newTid].second = newChunk;

        //wake up the worker thread
        cvReader_[newTid]->notify_one();
      }
    }
  }
  setMonStateSup(inRunEnd);
  //make sure threads finish reading
  unsigned int numFinishedThreads = 0;
  while (numFinishedThreads < workerThreads_.size()) {
    unsigned int tid = 0;
    while (!workerPool_.try_pop(tid)) {
      usleep(10000);
    }
    std::unique_lock<std::mutex> lk(mReader_);
    thread_quit_signal[tid] = true;
    cvReader_[tid]->notify_one();
    numFinishedThreads++;
  }
  for (unsigned int i = 0; i < workerThreads_.size(); i++) {
    workerThreads_[i]->join();
    delete workerThreads_[i];
  }
}

void DAQSource::readWorker(unsigned int tid) {
  bool init = true;
  threadInit_.exchange(true, std::memory_order_acquire);

  while (true) {
    tid_active_[tid] = false;
    std::unique_lock<std::mutex> lk(mReader_);
    workerJob_[tid].first = nullptr;
    workerJob_[tid].first = nullptr;

    assert(!thread_quit_signal[tid]);  //should never get it here
    workerPool_.push(tid);

    if (init) {
      std::unique_lock<std::mutex> lk(startupLock_);
      init = false;
      startupCv_.notify_one();
    }
    cvReader_[tid]->wait(lk);

    if (thread_quit_signal[tid])
      return;
    tid_active_[tid] = true;

    RawInputFile* file;
    InputChunk* chunk;

    assert(workerJob_[tid].first != nullptr && workerJob_[tid].second != nullptr);

    file = workerJob_[tid].first;
    chunk = workerJob_[tid].second;

    bool fitToBuffer = dataMode_->fitToBuffer();

    //resize if multi-chunked reading is not possible
    if (fitToBuffer) {
      uint64_t accum = 0;
      for (auto s : file->diskFileSizes_)
        accum += s;
      if (accum > eventChunkSize_) {
        if (!chunk->resize(accum, maxChunkSize_)) {
          edm::LogError("DAQSource")
              << "maxChunkSize can not accomodate the file set. Try increasing chunk size and/or chunk maximum size.";
          if (file->rawFd_ != -1 && (numConcurrentReads_ == 1 || chunk->offset_ == 0))
            close(file->rawFd_);
          setExceptionState_ = true;
          continue;
        } else {
          edm::LogInfo("DAQSource") << "chunk size was increased to " << (chunk->size_ >> 20) << " MB";
        }
      }
    }

    //skip reading initial header size in first chunk if inheriting file descriptor (already set at appropriate position)
    unsigned int bufferLeftInitial = (chunk->offset_ == 0 && file->rawFd_ != -1) ? file->rawHeaderSize_ : 0;
    const uint16_t readBlocks = chunk->size_ / eventChunkBlock_ + uint16_t(chunk->size_ % eventChunkBlock_ > 0);

    auto readPrimary = [&](uint64_t bufferLeft) {
      //BEGIN reading primary file - check if file descriptor is already open
      //in multi-threaded chunked mode, only first thread will use already open fd for reading the first file
      //fd will not be closed in other case (used by other threads)
      int fileDescriptor = -1;
      bool fileOpenedHere = false;

      if (numConcurrentReads_ == 1) {
        fileDescriptor = file->rawFd_;
        file->rawFd_ = -1;
        if (fileDescriptor == -1) {
          fileDescriptor = open(file->fileName_.c_str(), O_RDONLY);
          fileOpenedHere = true;
        }
      } else {
        if (chunk->offset_ == 0) {
          fileDescriptor = file->rawFd_;
          file->rawFd_ = -1;
          if (fileDescriptor == -1) {
            fileDescriptor = open(file->fileName_.c_str(), O_RDONLY);
            fileOpenedHere = true;
          }
        } else {
          fileDescriptor = open(file->fileName_.c_str(), O_RDONLY);
          fileOpenedHere = true;
        }
      }

      if (fileDescriptor == -1) {
        edm::LogError("DAQSource") << "readWorker failed to open file -: " << file->fileName_
                                   << " fd:" << fileDescriptor << " error: " << strerror(errno);
        setExceptionState_ = true;
        return;
      }

      if (fileOpenedHere) {  //fast forward to this chunk position
        off_t pos = lseek(fileDescriptor, chunk->offset_, SEEK_SET);
        if (pos == -1) {
          edm::LogError("DAQSource") << "readWorker failed to seek file -: " << file->fileName_
                                     << " fd:" << fileDescriptor << " to offset " << chunk->offset_
                                     << " error: " << strerror(errno);
          setExceptionState_ = true;
          return;
        }
      }

      LogDebug("DAQSource") << "Reader thread opened file -: TID: " << tid << " file: " << file->fileName_
                            << " at offset " << lseek(fileDescriptor, 0, SEEK_CUR);

      size_t skipped = bufferLeft;
      auto start = std::chrono::high_resolution_clock::now();
      for (unsigned int i = 0; i < readBlocks; i++) {
        ssize_t last;
        edm::LogInfo("DAQSource") << "readWorker read -: " << (int64_t)(chunk->usedSize_ - bufferLeft) << " or "
                                  << (int64_t)eventChunkBlock_;

        //protect against reading into next block
        last = ::read(fileDescriptor,
                      (void*)(chunk->buf_ + bufferLeft),
                      std::min((int64_t)(chunk->usedSize_ - bufferLeft), (int64_t)eventChunkBlock_));

        if (last < 0) {
          edm::LogError("DAQSource") << "readWorker failed to read file -: " << file->fileName_
                                     << " fd:" << fileDescriptor << " last: " << last << " error: " << strerror(errno);
          setExceptionState_ = true;
          break;
        }
        if (last > 0) {
          bufferLeft += last;
        }
        if ((uint64_t)last < eventChunkBlock_) {  //last read
          edm::LogInfo("DAQSource") << "chunkUsedSize" << chunk->usedSize_ << " u-s:" << (chunk->usedSize_ - skipped)
                                    << " ix:" << i * eventChunkBlock_ << " " << (size_t)last;
          //check if this is last block if single file, then total read size must match file size
          if (file->numFiles_ == 1 && !(chunk->usedSize_ - skipped == i * eventChunkBlock_ + (size_t)last)) {
            edm::LogError("DAQSource") << "readWorker failed to read file -: " << file->fileName_
                                       << " fd:" << fileDescriptor << " last:" << last
                                       << " expectedChunkSize:" << chunk->usedSize_
                                       << " readChunkSize:" << (skipped + i * eventChunkBlock_ + last)
                                       << " skipped:" << skipped << " block:" << (i + 1) << "/" << readBlocks
                                       << " error: " << strerror(errno);
            setExceptionState_ = true;
          }
          break;
        }
      }
      if (setExceptionState_)
        return;

      file->fileSizes_[0] = bufferLeft;

      if (chunk->offset_ + bufferLeft == file->diskFileSizes_[0] || bufferLeft == chunk->size_) {
        //file reading finished using this fd
        //or the whole buffer is filled (single sequential file spread over more chunks)
        close(fileDescriptor);
        fileDescriptor = -1;
      } else
        assert(fileDescriptor == -1);

      if (fitToBuffer && bufferLeft != file->diskFileSizes_[0]) {
        edm::LogError("DAQSource") << "mismatch between read file size for file -: " << file->fileNames_[0]
                                   << " read:" << bufferLeft << " expected:" << file->diskFileSizes_[0];
        setExceptionState_ = true;
        return;
      }

      auto end = std::chrono::high_resolution_clock::now();
      auto diff = end - start;
      std::chrono::milliseconds msec = std::chrono::duration_cast<std::chrono::milliseconds>(diff);
      LogDebug("DAQSource") << " finished reading block -: " << (bufferLeft >> 20) << " MB"
                            << " in " << msec.count() << " ms (" << (bufferLeft >> 20) / double(msec.count())
                            << " GB/s)";
    };
    //END primary function

    //SECONDARY files function
    auto readSecondary = [&](uint64_t bufferLeft, unsigned int j) {
      size_t fileLen = 0;

      std::string const& addFile = file->fileNames_[j];
      int fileDescriptor = open(addFile.c_str(), O_RDONLY);

      if (fileDescriptor < 0) {
        edm::LogError("DAQSource") << "readWorker failed to open file -: " << addFile << " fd:" << fileDescriptor
                                   << " error: " << strerror(errno);
        setExceptionState_ = true;
        return;
      }

      LogDebug("DAQSource") << "Reader thread opened file -: TID: " << tid << " file: " << addFile << " at offset "
                            << lseek(fileDescriptor, 0, SEEK_CUR);

      //size_t skipped = 0;//file is newly opened, read with header
      auto start = std::chrono::high_resolution_clock::now();
      for (unsigned int i = 0; i < readBlocks; i++) {
        ssize_t last;

        //protect against reading into next block
        //use bufferLeft for the write offset
        last = ::read(fileDescriptor,
                      (void*)(chunk->buf_ + bufferLeft),
                      std::min((uint64_t)file->diskFileSizes_[j], (uint64_t)eventChunkBlock_));

        if (last < 0) {
          edm::LogError("DAQSource") << "readWorker failed to read file -: " << addFile << " fd:" << fileDescriptor
                                     << " error: " << strerror(errno);
          setExceptionState_ = true;
          close(fileDescriptor);
          break;
        }
        if (last > 0) {
          bufferLeft += last;
          fileLen += last;
          file->fileSize_ += last;
        }
      };

      close(fileDescriptor);
      file->fileSizes_[j] = fileLen;
      assert(fileLen > 0);

      if (fitToBuffer && fileLen != file->diskFileSizes_[j]) {
        edm::LogError("DAQSource") << "mismatch between read file size for file -: " << file->fileNames_[j]
                                   << " read:" << fileLen << " expected:" << file->diskFileSizes_[j];
        setExceptionState_ = true;
        return;
      }

      auto end = std::chrono::high_resolution_clock::now();
      auto diff = end - start;
      std::chrono::milliseconds msec = std::chrono::duration_cast<std::chrono::milliseconds>(diff);
      LogDebug("DAQSource") << " finished reading block -: " << (bufferLeft >> 20) << " MB"
                            << " in " << msec.count() << " ms (" << (bufferLeft >> 20) / double(msec.count())
                            << " GB/s)";
    };

    //randomized order multi-file loop
    for (unsigned int j : file->fileOrder_) {
      if (j == 0) {
        readPrimary(bufferLeftInitial);
      } else
        readSecondary(file->bufferOffsets_[j], j);

      if (setExceptionState_)
        break;
    }

    if (setExceptionState_)
      continue;

    //detect FRD event version. Skip file Header if it exists
    if (dataMode_->dataVersion() == 0 && chunk->offset_ == 0) {
      dataMode_->detectVersion(chunk->buf_, file->rawHeaderSize_);
    }
    assert(dataMode_->versionCheck());

    chunk->readComplete_ =
        true;  //this is atomic to secure the sequential buffer fill before becoming available for processing)
    file->chunks_[chunk->fileIndex_] = chunk;  //put the completed chunk in the file chunk vector at predetermined index
  }
}

void DAQSource::threadError() {
  quit_threads_ = true;
  throw cms::Exception("DAQSource:threadError") << " file reader thread error ";
}

void DAQSource::setMonState(evf::FastMonState::InputState state) {
  if (fms_)
    fms_->setInState(state);
}

void DAQSource::setMonStateSup(evf::FastMonState::InputState state) {
  if (fms_)
    fms_->setInStateSup(state);
}

bool RawInputFile::advance(unsigned char*& dataPosition, const size_t size) {
  //wait for chunk

  while (!waitForChunk(currentChunk_)) {
    sourceParent_->setMonState(inWaitChunk);
    usleep(100000);
    sourceParent_->setMonState(inChunkReceived);
    if (sourceParent_->exceptionState())
      sourceParent_->threadError();
  }

  dataPosition = chunks_[currentChunk_]->buf_ + chunkPosition_;
  size_t currentLeft = chunks_[currentChunk_]->size_ - chunkPosition_;

  if (currentLeft < size) {
    //we need next chunk
    while (!waitForChunk(currentChunk_ + 1)) {
      sourceParent_->setMonState(inWaitChunk);
      usleep(100000);
      sourceParent_->setMonState(inChunkReceived);
      if (sourceParent_->exceptionState())
        sourceParent_->threadError();
    }
    //copy everything to beginning of the first chunk
    dataPosition -= chunkPosition_;
    assert(dataPosition == chunks_[currentChunk_]->buf_);
    memmove(chunks_[currentChunk_]->buf_, chunks_[currentChunk_]->buf_ + chunkPosition_, currentLeft);
    memcpy(chunks_[currentChunk_]->buf_ + currentLeft, chunks_[currentChunk_ + 1]->buf_, size - currentLeft);
    //set pointers at the end of the old data position
    bufferPosition_ += size;
    chunkPosition_ = size - currentLeft;
    currentChunk_++;
    return true;
  } else {
    chunkPosition_ += size;
    bufferPosition_ += size;
    return false;
  }
}

void DAQSource::reportEventsThisLumiInSource(unsigned int lumi, unsigned int events) {
  std::lock_guard<std::mutex> lock(monlock_);
  auto itr = sourceEventsReport_.find(lumi);
  if (itr != sourceEventsReport_.end())
    itr->second += events;
  else
    sourceEventsReport_[lumi] = events;
}

std::pair<bool, unsigned int> DAQSource::getEventReport(unsigned int lumi, bool erase) {
  std::lock_guard<std::mutex> lock(monlock_);
  auto itr = sourceEventsReport_.find(lumi);
  if (itr != sourceEventsReport_.end()) {
    std::pair<bool, unsigned int> ret(true, itr->second);
    if (erase)
      sourceEventsReport_.erase(itr);
    return ret;
  } else
    return std::pair<bool, unsigned int>(false, 0);
}

long DAQSource::initFileList() {
  std::sort(listFileNames_.begin(), listFileNames_.end(), [](std::string a, std::string b) {
    if (a.rfind('/') != std::string::npos)
      a = a.substr(a.rfind('/'));
    if (b.rfind('/') != std::string::npos)
      b = b.substr(b.rfind('/'));
    return b > a;
  });

  if (!listFileNames_.empty()) {
    //get run number from first file in the vector
    std::filesystem::path fileName = listFileNames_[0];
    std::string fileStem = fileName.stem().string();
    if (fileStem.find("file://") == 0)
      fileStem = fileStem.substr(7);
    else if (fileStem.find("file:") == 0)
      fileStem = fileStem.substr(5);
    auto end = fileStem.find('_');

    if (fileStem.find("run") == 0) {
      std::string runStr = fileStem.substr(3, end - 3);
      try {
        //get long to support test run numbers < 2^32
        long rval = std::stol(runStr);
        edm::LogInfo("DAQSource") << "Autodetected run number in fileListMode -: " << rval;
        return rval;
      } catch (const std::exception&) {
        edm::LogWarning("DAQSource") << "Unable to autodetect run number in fileListMode from file -: " << fileName;
      }
    }
  }
  return -1;
}

evf::EvFDaqDirector::FileStatus DAQSource::getFile(unsigned int& ls, std::string& nextFile, uint64_t& lockWaitTime) {
  if (fileListIndex_ < listFileNames_.size()) {
    nextFile = listFileNames_[fileListIndex_];
    if (nextFile.find("file://") == 0)
      nextFile = nextFile.substr(7);
    else if (nextFile.find("file:") == 0)
      nextFile = nextFile.substr(5);
    std::filesystem::path fileName = nextFile;
    std::string fileStem = fileName.stem().string();
    if (fileStem.find("ls"))
      fileStem = fileStem.substr(fileStem.find("ls") + 2);
    if (fileStem.find('_'))
      fileStem = fileStem.substr(0, fileStem.find('_'));

    if (!fileListLoopMode_)
      ls = std::stoul(fileStem);
    else  //always starting from LS 1 in loop mode
      ls = 1 + loopModeIterationInc_;

    //fsize = 0;
    //lockWaitTime = 0;
    fileListIndex_++;
    return evf::EvFDaqDirector::newFile;
  } else {
    if (!fileListLoopMode_)
      return evf::EvFDaqDirector::runEnded;
    else {
      //loop through files until interrupted
      loopModeIterationInc_++;
      fileListIndex_ = 0;
      return getFile(ls, nextFile, lockWaitTime);
    }
  }
}
