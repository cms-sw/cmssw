#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <sys/types.h>
#include <sys/file.h>
#include <sys/time.h>
#include <unistd.h>
#include <vector>
#include <fstream>
#include <zlib.h>
#include <cstdio>
#include <chrono>

#include <boost/algorithm/string.hpp>

#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "DataFormats/TCDS/interface/TCDSRaw.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/UnixSignalHandlers.h"

#include "EventFilter/Utilities/interface/GlobalEventNumber.h"

#include "EventFilter/Utilities/interface/FedRawDataInputSource.h"

#include "EventFilter/Utilities/interface/FastMonitoringService.h"
#include "EventFilter/Utilities/interface/DataPointDefinition.h"
#include "EventFilter/Utilities/interface/FFFNamingSchema.h"

#include "EventFilter/Utilities/interface/AuxiliaryMakers.h"

#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "EventFilter/Utilities/interface/crc32c.h"

//JSON file reader
#include "EventFilter/Utilities/interface/reader.h"

using namespace evf::FastMonState;

FedRawDataInputSource::FedRawDataInputSource(edm::ParameterSet const& pset, edm::InputSourceDescription const& desc)
    : edm::RawInputSource(pset, desc),
      defPath_(pset.getUntrackedParameter<std::string>("buDefPath", "")),
      eventChunkSize_(pset.getUntrackedParameter<unsigned int>("eventChunkSize", 32) * 1048576),
      eventChunkBlock_(pset.getUntrackedParameter<unsigned int>("eventChunkBlock", 32) * 1048576),
      numBuffers_(pset.getUntrackedParameter<unsigned int>("numBuffers", 2)),
      maxBufferedFiles_(pset.getUntrackedParameter<unsigned int>("maxBufferedFiles", 2)),
      getLSFromFilename_(pset.getUntrackedParameter<bool>("getLSFromFilename", true)),
      alwaysStartFromFirstLS_(pset.getUntrackedParameter<bool>("alwaysStartFromFirstLS", false)),
      verifyChecksum_(pset.getUntrackedParameter<bool>("verifyChecksum", true)),
      useL1EventID_(pset.getUntrackedParameter<bool>("useL1EventID", false)),
      testTCDSFEDRange_(
          pset.getUntrackedParameter<std::vector<unsigned int>>("testTCDSFEDRange", std::vector<unsigned int>())),
      fileNames_(pset.getUntrackedParameter<std::vector<std::string>>("fileNames", std::vector<std::string>())),
      fileListMode_(pset.getUntrackedParameter<bool>("fileListMode", false)),
      fileListLoopMode_(pset.getUntrackedParameter<bool>("fileListLoopMode", false)),
      runNumber_(edm::Service<evf::EvFDaqDirector>()->getRunNumber()),
      daqProvenanceHelper_(edm::TypeID(typeid(FEDRawDataCollection))),
      eventID_(),
      processHistoryID_(),
      currentLumiSection_(0),
      tcds_pointer_(nullptr),
      eventsThisLumi_(0) {
  char thishost[256];
  gethostname(thishost, 255);
  edm::LogInfo("FedRawDataInputSource") << "Construction. read-ahead chunk size -: " << std::endl
                                        << (eventChunkSize_ / 1048576) << " MB on host " << thishost;

  if (!testTCDSFEDRange_.empty()) {
    if (testTCDSFEDRange_.size() != 2) {
      throw cms::Exception("FedRawDataInputSource::fillFEDRawDataCollection")
          << "Invalid TCDS Test FED range parameter";
    }
    MINTCDSuTCAFEDID_ = testTCDSFEDRange_[0];
    MAXTCDSuTCAFEDID_ = testTCDSFEDRange_[1];
  }

  long autoRunNumber = -1;
  if (fileListMode_) {
    autoRunNumber = initFileList();
    if (!fileListLoopMode_) {
      if (autoRunNumber < 0)
        throw cms::Exception("FedRawDataInputSource::FedRawDataInputSource") << "Run number not found from filename";
      //override run number
      runNumber_ = (edm::RunNumber_t)autoRunNumber;
      edm::Service<evf::EvFDaqDirector>()->overrideRunNumber((unsigned int)autoRunNumber);
    }
  }

  processHistoryID_ = daqProvenanceHelper_.daqInit(productRegistryUpdate(), processHistoryRegistryForUpdate());
  setNewRun();
  //todo:autodetect from file name (assert if names differ)
  setRunAuxiliary(new edm::RunAuxiliary(runNumber_, edm::Timestamp::beginOfTime(), edm::Timestamp::invalidTimestamp()));

  //make sure that chunk size is N * block size
  assert(eventChunkSize_ >= eventChunkBlock_);
  readBlocks_ = eventChunkSize_ / eventChunkBlock_;
  if (readBlocks_ * eventChunkBlock_ != eventChunkSize_)
    eventChunkSize_ = readBlocks_ * eventChunkBlock_;

  if (!numBuffers_)
    throw cms::Exception("FedRawDataInputSource::FedRawDataInputSource")
        << "no reading enabled with numBuffers parameter 0";

  numConcurrentReads_ = numBuffers_ - 1;
  singleBufferMode_ = !(numBuffers_ > 1);
  readingFilesCount_ = 0;

  if (!crc32c_hw_test())
    edm::LogError("FedRawDataInputSource::FedRawDataInputSource") << "Intel crc32c checksum computation unavailable";

  //get handles to DaqDirector and FastMonitoringService because getting them isn't possible in readSupervisor thread
  if (fileListMode_) {
    try {
      fms_ = static_cast<evf::FastMonitoringService*>(edm::Service<evf::MicroStateService>().operator->());
    } catch (cms::Exception const&) {
      edm::LogInfo("FedRawDataInputSource") << "No FastMonitoringService found in the configuration";
    }
  } else {
    fms_ = static_cast<evf::FastMonitoringService*>(edm::Service<evf::MicroStateService>().operator->());
    if (!fms_) {
      throw cms::Exception("FedRawDataInputSource") << "FastMonitoringService not found";
    }
  }

  daqDirector_ = edm::Service<evf::EvFDaqDirector>().operator->();
  if (!daqDirector_)
    cms::Exception("FedRawDataInputSource") << "EvFDaqDirector not found";

  useFileBroker_ = daqDirector_->useFileBroker();
  if (useFileBroker_)
    edm::LogInfo("FedRawDataInputSource") << "EvFDaqDirector/Source configured to use file service";
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
    freeChunks_.push(new InputChunk(i, eventChunkSize_));
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
    workerThreads_.push_back(new std::thread(&FedRawDataInputSource::readWorker, this, i));
    startupCv_.wait(lk);
  }

  runAuxiliary()->setProcessHistoryID(processHistoryID_);
}

FedRawDataInputSource::~FedRawDataInputSource() {
  quit_threads_ = true;

  //delete any remaining open files
  if (!fms_ || !fms_->exceptionDetected()) {
    for (auto it = filesToDelete_.begin(); it != filesToDelete_.end(); it++)
      it->second.reset();
  } else {
    //skip deleting files with exception
    for (auto it = filesToDelete_.begin(); it != filesToDelete_.end(); it++) {
      //it->second->unsetDeleteFile();
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

void FedRawDataInputSource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("File-based Filter Farm input source for reading raw data from BU ramdisk");
  desc.addUntracked<unsigned int>("eventChunkSize", 32)->setComment("Input buffer (chunk) size");
  desc.addUntracked<unsigned int>("eventChunkBlock", 32)
      ->setComment("Block size used in a single file read call (must be smaller or equal to buffer size)");
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

edm::RawInputSource::Next FedRawDataInputSource::checkNext() {
  if (!startedSupervisorThread_) {
    //this thread opens new files and dispatches reading to worker readers
    //threadInit_.store(false,std::memory_order_release);
    std::unique_lock<std::mutex> lk(startupLock_);
    readSupervisorThread_ = std::make_unique<std::thread>(&FedRawDataInputSource::readSupervisor, this);
    startedSupervisorThread_ = true;
    startupCv_.wait(lk);
  }
  //signal hltd to start event accounting
  if (!currentLumiSection_)
    daqDirector_->createProcessingNotificationMaybe();
  setMonState(inWaitInput);
  switch (nextEvent()) {
    case evf::EvFDaqDirector::runEnded: {
      //maybe create EoL file in working directory before ending run
      struct stat buf;
      if (!useFileBroker_ && currentLumiSection_ > 0) {
        bool eolFound = (stat(daqDirector_->getEoLSFilePathOnBU(currentLumiSection_).c_str(), &buf) == 0);
        if (eolFound) {
          const std::string fuEoLS = daqDirector_->getEoLSFilePathOnFU(currentLumiSection_);
          bool found = (stat(fuEoLS.c_str(), &buf) == 0);
          if (!found) {
            daqDirector_->lockFULocal2();
            int eol_fd =
                open(fuEoLS.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH);
            close(eol_fd);
            daqDirector_->unlockFULocal2();
          }
        }
      }
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
      edm::LogInfo("FedRawDataInputSource") << "----------------RUN ENDED----------------";
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
      if (!getLSFromFilename_) {
        //get new lumi from file header
        if (event_->lumi() > currentLumiSection_) {
          reportEventsThisLumiInSource(currentLumiSection_, eventsThisLumi_);
          eventsThisLumi_ = 0;
          maybeOpenNewLumiSection(event_->lumi());
        }
      }
      if (fileListMode_ || fileListLoopMode_)
        eventRunNumber_ = runNumber_;
      else
        eventRunNumber_ = event_->run();
      L1EventID_ = event_->event();

      setEventCached();

      return Next::kEvent;
    }
  }
}

void FedRawDataInputSource::maybeOpenNewLumiSection(const uint32_t lumiSection) {
  if (!luminosityBlockAuxiliary() || luminosityBlockAuxiliary()->luminosityBlock() != lumiSection) {
    if (!useFileBroker_) {
      if (currentLumiSection_ > 0) {
        const std::string fuEoLS = daqDirector_->getEoLSFilePathOnFU(currentLumiSection_);
        struct stat buf;
        bool found = (stat(fuEoLS.c_str(), &buf) == 0);
        if (!found) {
          daqDirector_->lockFULocal2();
          int eol_fd =
              open(fuEoLS.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH);
          close(eol_fd);
          daqDirector_->createBoLSFile(lumiSection, false);
          daqDirector_->unlockFULocal2();
        }
      } else
        daqDirector_->createBoLSFile(lumiSection, true);  //needed for initial lumisection
    }

    currentLumiSection_ = lumiSection;

    resetLuminosityBlockAuxiliary();

    timeval tv;
    gettimeofday(&tv, nullptr);
    const edm::Timestamp lsopentime((unsigned long long)tv.tv_sec * 1000000 + (unsigned long long)tv.tv_usec);

    edm::LuminosityBlockAuxiliary* lumiBlockAuxiliary = new edm::LuminosityBlockAuxiliary(
        runAuxiliary()->run(), lumiSection, lsopentime, edm::Timestamp::invalidTimestamp());

    setLuminosityBlockAuxiliary(lumiBlockAuxiliary);
    luminosityBlockAuxiliary()->setProcessHistoryID(processHistoryID_);

    edm::LogInfo("FedRawDataInputSource") << "New lumi section was opened. LUMI -: " << lumiSection;
  }
}

inline evf::EvFDaqDirector::FileStatus FedRawDataInputSource::nextEvent() {
  evf::EvFDaqDirector::FileStatus status = evf::EvFDaqDirector::noFile;
  while ((status = getNextEvent()) == evf::EvFDaqDirector::noFile) {
    if (edm::shutdown_flag.load(std::memory_order_relaxed))
      break;
  }
  return status;
}

inline evf::EvFDaqDirector::FileStatus FedRawDataInputSource::getNextEvent() {
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
      throw cms::Exception("FedRawDataInputSource::getNextEvent")
          << "Run has been aborted by the input source reader thread";
    } else if (status == evf::EvFDaqDirector::newLumi) {
      setMonState(inNewLumi);
      if (getLSFromFilename_) {
        if (currentFile_->lumi_ > currentLumiSection_) {
          reportEventsThisLumiInSource(currentLumiSection_, eventsThisLumi_);
          eventsThisLumi_ = 0;
          maybeOpenNewLumiSection(currentFile_->lumi_);
        }
      } else {  //let this be picked up from next event
        status = evf::EvFDaqDirector::noFile;
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
    if (getLSFromFilename_)
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
      throw cms::Exception("FedRawDataInputSource::getNextEvent")
          << "Fully processed " << currentFile_->nProcessed_ << " from the file " << currentFile_->fileName_
          << " but according to BU JSON there should be " << currentFile_->nEvents_ << " events";
    }
    //try to wake up supervisor thread which might be sleeping waiting for the free chunk
    if (singleBufferMode_) {
      std::unique_lock<std::mutex> lkw(mWakeup_);
      cvWakeup_.notify_one();
    }
    bufferInputRead_ = 0;
    if (!daqDirector_->isSingleStreamThread() && !fileListMode_) {
      //put the file in pending delete list;
      std::unique_lock<std::mutex> lkw(fileDeleteLock_);
      filesToDelete_.push_back(std::pair<int, std::unique_ptr<InputFile>>(currentFileIndex_, std::move(currentFile_)));
    } else {
      //in single-thread and stream jobs, events are already processed
      currentFile_.reset();
    }
    return evf::EvFDaqDirector::noFile;
  }

  //handle RAW file header
  if (currentFile_->bufferPosition_ == 0 && currentFile_->rawHeaderSize_ > 0) {
    if (currentFile_->fileSize_ <= currentFile_->rawHeaderSize_) {
      if (currentFile_->fileSize_ < currentFile_->rawHeaderSize_)
        throw cms::Exception("FedRawDataInputSource::getNextEvent")
            << "Premature end of input file while reading file header";

      edm::LogWarning("FedRawDataInputSource")
          << "File with only raw header and no events received in LS " << currentFile_->lumi_;
      if (getLSFromFilename_)
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
  if (currentFile_->fileSize_ - currentFile_->bufferPosition_ < FRDHeaderVersionSize[detectedFRDversion_]) {
    throw cms::Exception("FedRawDataInputSource::getNextEvent")
        << "Premature end of input file while reading event header";
  }
  if (singleBufferMode_) {
    //should already be there
    setMonState(inWaitChunk);
    while (!currentFile_->waitForChunk(currentFile_->currentChunk_)) {
      usleep(10000);
      if (currentFile_->parent_->exceptionState() || setExceptionState_)
        currentFile_->parent_->threadError();
    }
    setMonState(inChunkReceived);

    unsigned char* dataPosition = currentFile_->chunks_[0]->buf_ + currentFile_->chunkPosition_;

    //conditions when read amount is not sufficient for the header to fit
    if (!bufferInputRead_ || bufferInputRead_ < FRDHeaderVersionSize[detectedFRDversion_] ||
        eventChunkSize_ - currentFile_->chunkPosition_ < FRDHeaderVersionSize[detectedFRDversion_]) {
      readNextChunkIntoBuffer(currentFile_.get());

      if (detectedFRDversion_ == 0) {
        detectedFRDversion_ = *((uint16_t*)dataPosition);
        if (detectedFRDversion_ > FRDHeaderMaxVersion)
          throw cms::Exception("FedRawDataInputSource::getNextEvent")
              << "Unknown FRD version -: " << detectedFRDversion_;
        assert(detectedFRDversion_ >= 1);
      }

      //recalculate chunk position
      dataPosition = currentFile_->chunks_[0]->buf_ + currentFile_->chunkPosition_;
      if (bufferInputRead_ < FRDHeaderVersionSize[detectedFRDversion_]) {
        throw cms::Exception("FedRawDataInputSource::getNextEvent")
            << "Premature end of input file while reading event header";
      }
    }

    event_ = std::make_unique<FRDEventMsgView>(dataPosition);
    if (event_->size() > eventChunkSize_) {
      throw cms::Exception("FedRawDataInputSource::getNextEvent")
          << " event id:" << event_->event() << " lumi:" << event_->lumi() << " run:" << event_->run()
          << " of size:" << event_->size() << " bytes does not fit into a chunk of size:" << eventChunkSize_
          << " bytes";
    }

    const uint32_t msgSize = event_->size() - FRDHeaderVersionSize[detectedFRDversion_];

    if (currentFile_->fileSize_ - currentFile_->bufferPosition_ < msgSize) {
      throw cms::Exception("FedRawDataInputSource::getNextEvent")
          << "Premature end of input file while reading event data";
    }
    if (eventChunkSize_ - currentFile_->chunkPosition_ < msgSize) {
      readNextChunkIntoBuffer(currentFile_.get());
      //recalculate chunk position
      dataPosition = currentFile_->chunks_[0]->buf_ + currentFile_->chunkPosition_;
      event_ = std::make_unique<FRDEventMsgView>(dataPosition);
    }
    currentFile_->bufferPosition_ += event_->size();
    currentFile_->chunkPosition_ += event_->size();
    //last chunk is released when this function is invoked next time

  }
  //multibuffer mode:
  else {
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

    //read header, copy it to a single chunk if necessary
    bool chunkEnd = currentFile_->advance(dataPosition, FRDHeaderVersionSize[detectedFRDversion_]);

    event_ = std::make_unique<FRDEventMsgView>(dataPosition);
    if (event_->size() > eventChunkSize_) {
      throw cms::Exception("FedRawDataInputSource::getNextEvent")
          << " event id:" << event_->event() << " lumi:" << event_->lumi() << " run:" << event_->run()
          << " of size:" << event_->size() << " bytes does not fit into a chunk of size:" << eventChunkSize_
          << " bytes";
    }

    const uint32_t msgSize = event_->size() - FRDHeaderVersionSize[detectedFRDversion_];

    if (currentFile_->fileSize_ - currentFile_->bufferPosition_ < msgSize) {
      throw cms::Exception("FedRawDataInputSource::getNextEvent")
          << "Premature end of input file while reading event data";
    }

    if (chunkEnd) {
      //header was at the chunk boundary, we will have to move payload as well
      currentFile_->moveToPreviousChunk(msgSize, FRDHeaderVersionSize[detectedFRDversion_]);
      chunkIsFree_ = true;
    } else {
      //header was contiguous, but check if payload fits the chunk
      if (eventChunkSize_ - currentFile_->chunkPosition_ < msgSize) {
        //rewind to header start position
        currentFile_->rewindChunk(FRDHeaderVersionSize[detectedFRDversion_]);
        //copy event to a chunk start and move pointers

        setMonState(inWaitChunk);

        chunkEnd = currentFile_->advance(dataPosition, FRDHeaderVersionSize[detectedFRDversion_] + msgSize);

        setMonState(inChunkReceived);

        assert(chunkEnd);
        chunkIsFree_ = true;
        //header is moved
        event_ = std::make_unique<FRDEventMsgView>(dataPosition);
      } else {
        //everything is in a single chunk, only move pointers forward
        chunkEnd = currentFile_->advance(dataPosition, msgSize);
        assert(!chunkEnd);
        chunkIsFree_ = false;
      }
    }
  }  //end multibuffer mode
  setMonState(inChecksumEvent);

  if (verifyChecksum_ && event_->version() >= 5) {
    uint32_t crc = 0;
    crc = crc32c(crc, (const unsigned char*)event_->payload(), event_->eventSize());
    if (crc != event_->crc32c()) {
      if (fms_)
        fms_->setExceptionDetected(currentLumiSection_);
      throw cms::Exception("FedRawDataInputSource::getNextEvent")
          << "Found a wrong crc32c checksum: expected 0x" << std::hex << event_->crc32c() << " but calculated 0x"
          << crc;
    }
  } else if (verifyChecksum_ && event_->version() >= 3) {
    uint32_t adler = adler32(0L, Z_NULL, 0);
    adler = adler32(adler, (Bytef*)event_->payload(), event_->eventSize());

    if (adler != event_->adler32()) {
      if (fms_)
        fms_->setExceptionDetected(currentLumiSection_);
      throw cms::Exception("FedRawDataInputSource::getNextEvent")
          << "Found a wrong Adler32 checksum: expected 0x" << std::hex << event_->adler32() << " but calculated 0x"
          << adler;
    }
  }
  setMonState(inCachedEvent);

  currentFile_->nProcessed_++;

  return evf::EvFDaqDirector::sameFile;
}

void FedRawDataInputSource::read(edm::EventPrincipal& eventPrincipal) {
  setMonState(inReadEvent);
  std::unique_ptr<FEDRawDataCollection> rawData(new FEDRawDataCollection);
  bool tcdsInRange;
  edm::Timestamp tstamp = fillFEDRawDataCollection(*rawData, tcdsInRange);

  if (useL1EventID_) {
    eventID_ = edm::EventID(eventRunNumber_, currentLumiSection_, L1EventID_);
    edm::EventAuxiliary aux(eventID_, processGUID(), tstamp, event_->isRealData(), edm::EventAuxiliary::PhysicsTrigger);
    aux.setProcessHistoryID(processHistoryID_);
    makeEvent(eventPrincipal, aux);
  } else if (tcds_pointer_ == nullptr) {
    if (!GTPEventID_) {
      throw cms::Exception("FedRawDataInputSource::read")
          << "No TCDS or GTP FED in event with FEDHeader EID -: " << L1EventID_;
    }
    eventID_ = edm::EventID(eventRunNumber_, currentLumiSection_, GTPEventID_);
    edm::EventAuxiliary aux(eventID_, processGUID(), tstamp, event_->isRealData(), edm::EventAuxiliary::PhysicsTrigger);
    aux.setProcessHistoryID(processHistoryID_);
    makeEvent(eventPrincipal, aux);
  } else {
    const FEDHeader fedHeader(tcds_pointer_);
    tcds::Raw_v1 const* tcds = reinterpret_cast<tcds::Raw_v1 const*>(tcds_pointer_ + FEDHeader::length);
    edm::EventAuxiliary aux =
        evf::evtn::makeEventAuxiliary(tcds,
                                      eventRunNumber_,
                                      currentLumiSection_,
                                      event_->isRealData(),
                                      static_cast<edm::EventAuxiliary::ExperimentType>(fedHeader.triggerType()),
                                      processGUID(),
                                      !fileListLoopMode_,
                                      !tcdsInRange);
    aux.setProcessHistoryID(processHistoryID_);
    makeEvent(eventPrincipal, aux);
  }

  std::unique_ptr<edm::WrapperBase> edp(new edm::Wrapper<FEDRawDataCollection>(std::move(rawData)));

  eventPrincipal.put(daqProvenanceHelper_.branchDescription(), std::move(edp), daqProvenanceHelper_.dummyProvenance());

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
      if (!fileIsBeingProcessed && (!fms_ || !fms_->isExceptionOnData(it->second->lumi_))) {
        std::string fileToDelete = it->second->fileName_;
        it = filesToDelete_.erase(it);
      } else
        it++;
    }
  }
  if (chunkIsFree_)
    freeChunks_.push(currentFile_->chunks_[currentFile_->currentChunk_ - 1]);
  chunkIsFree_ = false;
  setMonState(inNoRequest);
  return;
}

edm::Timestamp FedRawDataInputSource::fillFEDRawDataCollection(FEDRawDataCollection& rawData, bool& tcdsInRange) {
  edm::TimeValue_t time;
  timeval stv;
  gettimeofday(&stv, nullptr);
  time = stv.tv_sec;
  time = (time << 32) + stv.tv_usec;
  edm::Timestamp tstamp(time);

  uint32_t eventSize = event_->eventSize();
  unsigned char* event = (unsigned char*)event_->payload();
  GTPEventID_ = 0;
  tcds_pointer_ = nullptr;
  tcdsInRange = false;
  uint16_t selectedTCDSFed = 0;
  while (eventSize > 0) {
    assert(eventSize >= FEDTrailer::length);
    eventSize -= FEDTrailer::length;
    const FEDTrailer fedTrailer(event + eventSize);
    const uint32_t fedSize = fedTrailer.fragmentLength() << 3;  //trailer length counts in 8 bytes
    assert(eventSize >= fedSize - FEDHeader::length);
    eventSize -= (fedSize - FEDHeader::length);
    const FEDHeader fedHeader(event + eventSize);
    const uint16_t fedId = fedHeader.sourceID();
    if (fedId > FEDNumbering::MAXFEDID) {
      throw cms::Exception("FedRawDataInputSource::fillFEDRawDataCollection") << "Out of range FED ID : " << fedId;
    } else if (fedId >= MINTCDSuTCAFEDID_ && fedId <= MAXTCDSuTCAFEDID_) {
      if (!selectedTCDSFed) {
        selectedTCDSFed = fedId;
        tcds_pointer_ = event + eventSize;
        if (fedId >= FEDNumbering::MINTCDSuTCAFEDID && fedId <= FEDNumbering::MAXTCDSuTCAFEDID) {
          tcdsInRange = true;
        }
      } else
        throw cms::Exception("FedRawDataInputSource::fillFEDRawDataCollection")
            << "Second TCDS FED ID " << fedId << " found. First ID: " << selectedTCDSFed;
    }
    if (fedId == FEDNumbering::MINTriggerGTPFEDID) {
      if (evf::evtn::evm_board_sense(event + eventSize, fedSize))
        GTPEventID_ = evf::evtn::get(event + eventSize, true);
      else
        GTPEventID_ = evf::evtn::get(event + eventSize, false);
      //evf::evtn::evm_board_setformat(fedSize);
      const uint64_t gpsl = evf::evtn::getgpslow(event + eventSize);
      const uint64_t gpsh = evf::evtn::getgpshigh(event + eventSize);
      tstamp = edm::Timestamp(static_cast<edm::TimeValue_t>((gpsh << 32) + gpsl));
    }
    //take event ID from GTPE FED
    if (fedId == FEDNumbering::MINTriggerEGTPFEDID && GTPEventID_ == 0) {
      if (evf::evtn::gtpe_board_sense(event + eventSize)) {
        GTPEventID_ = evf::evtn::gtpe_get(event + eventSize);
      }
    }
    FEDRawData& fedData = rawData.FEDData(fedId);
    fedData.resize(fedSize);
    memcpy(fedData.data(), event + eventSize, fedSize);
  }
  assert(eventSize == 0);

  return tstamp;
}

void FedRawDataInputSource::rewind_() {}

void FedRawDataInputSource::readSupervisor() {
  bool stop = false;
  unsigned int currentLumiSection = 0;
  //threadInit_.exchange(true,std::memory_order_acquire);

  {
    std::unique_lock<std::mutex> lk(startupLock_);
    startupCv_.notify_one();
  }

  uint32_t ls = 0;
  uint32_t monLS = 1;
  uint32_t lockCount = 0;
  uint64_t sumLockWaitTimeUs = 0.;

  while (!stop) {
    //wait for at least one free thread and chunk
    int counter = 0;

    while ((workerPool_.empty() && !singleBufferMode_) || freeChunks_.empty() ||
           readingFilesCount_ >= maxBufferedFiles_) {
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
        //if (!(counter%50)) edm::LogInfo("FedRawDataInputSource") << "No free chunks or threads...";
        LogDebug("FedRawDataInputSource") << "No free chunks or threads...";
      } else {
        assert(!(workerPool_.empty() && !singleBufferMode_) || freeChunks_.empty());
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
    uint32_t fileSizeIndex;
    int64_t fileSizeFromMetadata;

    if (fms_) {
      setMonStateSup(inSupBusy);
      fms_->startedLookingForFile();
    }

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
            edm::LogWarning("FedRawDataInputSource") << "Source detected that the lumisection is discarded -: " << i;
            hasDiscardedLumi = true;
            break;
          }
        }
        if (hasDiscardedLumi)
          break;

        setMonStateSup(inThrottled);

        if (!(counter % 50))
          edm::LogWarning("FedRawDataInputSource") << "Input throttled detected, reading files is paused...";
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
        status = getFile(ls, nextFile, fileSizeIndex, thisLockWaitTimeUs);
        if (status == evf::EvFDaqDirector::newFile) {
          if (evf::EvFDaqDirector::parseFRDFileHeader(nextFile,
                                                      rawFd,
                                                      rawHeaderSize,
                                                      lsFromRaw,
                                                      serverEventsInNewFile,
                                                      fileSizeFromMetadata,
                                                      false,
                                                      false,
                                                      false) != 0) {
            //error
            setExceptionState_ = true;
            stop = true;
            break;
          }
          if (!getLSFromFilename_)
            ls = lsFromRaw;
        }
      } else if (!useFileBroker_)
        status = daqDirector_->updateFuLock(
            ls, nextFile, fileSizeIndex, rawHeaderSize, thisLockWaitTimeUs, setExceptionState_);
      else {
        status = daqDirector_->getNextFromFileBroker(currentLumiSection,
                                                     ls,
                                                     nextFile,
                                                     rawFd,
                                                     rawHeaderSize,
                                                     serverEventsInNewFile,
                                                     fileSizeFromMetadata,
                                                     thisLockWaitTimeUs);
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

      //check again for any remaining index/EoLS files after EoR file is seen
      if (status == evf::EvFDaqDirector::runEnded && !fileListMode_ && !useFileBroker_) {
        setMonStateSup(inRunEnd);
        usleep(100000);
        //now all files should have appeared in ramdisk, check again if any raw files were left behind
        status = daqDirector_->updateFuLock(
            ls, nextFile, fileSizeIndex, rawHeaderSize, thisLockWaitTimeUs, setExceptionState_);
        if (currentLumiSection != ls && status == evf::EvFDaqDirector::runEnded)
          status = evf::EvFDaqDirector::noFile;
      }

      if (status == evf::EvFDaqDirector::runEnded) {
        std::unique_ptr<InputFile> inf(new InputFile(evf::EvFDaqDirector::runEnded));
        fileQueue_.push(std::move(inf));
        stop = true;
        break;
      }

      //error from filelocking function
      if (status == evf::EvFDaqDirector::runAbort) {
        std::unique_ptr<InputFile> inf(new InputFile(evf::EvFDaqDirector::runAbort, 0));
        fileQueue_.push(std::move(inf));
        stop = true;
        break;
      }
      //queue new lumisection
      if (getLSFromFilename_) {
        if (ls > currentLumiSection) {
          if (!useFileBroker_) {
            //file locking
            //setMonStateSup(inSupNewLumi);
            currentLumiSection = ls;
            std::unique_ptr<InputFile> inf(new InputFile(evf::EvFDaqDirector::newLumi, currentLumiSection));
            fileQueue_.push(std::move(inf));
          } else {
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
                  std::unique_ptr<InputFile> inf(new InputFile(evf::EvFDaqDirector::newLumi, ls));
                  fileQueue_.push(std::move(inf));
                }
              } else if (ls < 100) {
                //look at last LS file on disk to start from that lumisection (only within first 100 LS)
                unsigned int lsToStart = daqDirector_->getLumisectionToStart();

                for (unsigned int nextLS = std::min(lsToStart, ls); nextLS <= ls; nextLS++) {
                  std::unique_ptr<InputFile> inf(new InputFile(evf::EvFDaqDirector::newLumi, nextLS));
                  fileQueue_.push(std::move(inf));
                }
              } else {
                //start from current LS
                std::unique_ptr<InputFile> inf(new InputFile(evf::EvFDaqDirector::newLumi, ls));
                fileQueue_.push(std::move(inf));
              }
            } else {
              //queue all lumisections after last one seen to avoid gaps
              for (unsigned int nextLS = currentLumiSection + 1; nextLS <= ls; nextLS++) {
                std::unique_ptr<InputFile> inf(new InputFile(evf::EvFDaqDirector::newLumi, nextLS));
                fileQueue_.push(std::move(inf));
              }
            }
            currentLumiSection = ls;
          }
        }
        //else
        if (currentLumiSection > 0 && ls < currentLumiSection) {
          edm::LogError("FedRawDataInputSource")
              << "Got old LS (" << ls << ") file from EvFDAQDirector! Expected LS:" << currentLumiSection
              << ". Aborting execution." << std::endl;
          if (rawFd != -1)
            close(rawFd);
          rawFd = -1;
          std::unique_ptr<InputFile> inf(new InputFile(evf::EvFDaqDirector::runAbort, 0));
          fileQueue_.push(std::move(inf));
          stop = true;
          break;
        }
      }

      int dbgcount = 0;
      if (status == evf::EvFDaqDirector::noFile) {
        setMonStateSup(inSupNoFile);
        dbgcount++;
        if (!(dbgcount % 20))
          LogDebug("FedRawDataInputSource") << "No file for me... sleep and try again...";
        if (!useFileBroker_)
          usleep(100000);
        else {
          backoff_exp = std::min(4, backoff_exp);  // max 1.6 seconds
          //backoff_exp=0; // disabled!
          int sleeptime = (int)(100000. * pow(2, backoff_exp));
          usleep(sleeptime);
          backoff_exp++;
        }
      } else
        backoff_exp = 0;
    }
    //end of file grab loop, parse result
    if (status == evf::EvFDaqDirector::newFile) {
      setMonStateSup(inSupNewFile);
      LogDebug("FedRawDataInputSource") << "The director says to grab -: " << nextFile;

      std::string rawFile;
      //file service will report raw extension
      if (useFileBroker_ || rawHeaderSize)
        rawFile = nextFile;
      else {
        std::filesystem::path rawFilePath(nextFile);
        rawFile = rawFilePath.replace_extension(".raw").string();
      }

      struct stat st;
      int stat_res = stat(rawFile.c_str(), &st);
      if (stat_res == -1) {
        edm::LogError("FedRawDataInputSource") << "Can not stat file (" << errno << "):-" << rawFile << std::endl;
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
        std::string empty;
        if (!useFileBroker_) {
          if (rawHeaderSize) {
            int rawFdEmpty = -1;
            uint16_t rawHeaderCheck;
            bool fileFound;
            eventsInNewFile = daqDirector_->grabNextJsonFromRaw(
                nextFile, rawFdEmpty, rawHeaderCheck, fileSizeFromMetadata, fileFound, 0, true);
            assert(fileFound && rawHeaderCheck == rawHeaderSize);
            daqDirector_->unlockFULocal();
          } else
            eventsInNewFile = daqDirector_->grabNextJsonFileAndUnlock(nextFile);
        } else
          eventsInNewFile = serverEventsInNewFile;
        assert(eventsInNewFile >= 0);
        assert((eventsInNewFile > 0) ==
               (fileSize > rawHeaderSize));  //file without events must be empty or contain only header
      }

      if (!singleBufferMode_) {
        //calculate number of needed chunks
        unsigned int neededChunks = fileSize / eventChunkSize_;
        if (fileSize % eventChunkSize_)
          neededChunks++;

        std::unique_ptr<InputFile> newInputFile(new InputFile(evf::EvFDaqDirector::FileStatus::newFile,
                                                              ls,
                                                              rawFile,
                                                              !fileListMode_,
                                                              rawFd,
                                                              fileSize,
                                                              rawHeaderSize,
                                                              neededChunks,
                                                              eventsInNewFile,
                                                              this));
        readingFilesCount_++;
        auto newInputFilePtr = newInputFile.get();
        fileQueue_.push(std::move(newInputFile));

        for (unsigned int i = 0; i < neededChunks; i++) {
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

          unsigned int toRead = eventChunkSize_;
          if (i == neededChunks - 1 && fileSize % eventChunkSize_)
            toRead = fileSize % eventChunkSize_;
          newChunk->reset(i * eventChunkSize_, toRead, i);

          workerJob_[newTid].first = newInputFilePtr;
          workerJob_[newTid].second = newChunk;

          //wake up the worker thread
          cvReader_[newTid]->notify_one();
        }
      } else {
        if (!eventsInNewFile) {
          if (rawFd) {
            close(rawFd);
            rawFd = -1;
          }
          //still queue file for lumi update
          std::unique_lock<std::mutex> lkw(mWakeup_);
          //TODO: also file with only file header fits in this edge case. Check if read correctly in single buffer mode
          std::unique_ptr<InputFile> newInputFile(new InputFile(evf::EvFDaqDirector::FileStatus::newFile,
                                                                ls,
                                                                rawFile,
                                                                !fileListMode_,
                                                                rawFd,
                                                                fileSize,
                                                                rawHeaderSize,
                                                                (rawHeaderSize > 0),
                                                                0,
                                                                this));
          readingFilesCount_++;
          fileQueue_.push(std::move(newInputFile));
          cvWakeup_.notify_one();
          break;
        }
        //in single-buffer mode put single chunk in the file and let the main thread read the file
        InputChunk* newChunk = nullptr;
        //should be available immediately
        while (!freeChunks_.try_pop(newChunk)) {
          usleep(100000);
          if (quit_threads_.load(std::memory_order_relaxed)) {
            stop = true;
            break;
          }
        }

        if (newChunk == nullptr) {
          stop = true;
        }

        if (stop)
          break;

        std::unique_lock<std::mutex> lkw(mWakeup_);

        unsigned int toRead = eventChunkSize_;
        if (fileSize % eventChunkSize_)
          toRead = fileSize % eventChunkSize_;
        newChunk->reset(0, toRead, 0);
        newChunk->readComplete_ = true;

        //push file and wakeup main thread
        std::unique_ptr<InputFile> newInputFile(new InputFile(evf::EvFDaqDirector::FileStatus::newFile,
                                                              ls,
                                                              rawFile,
                                                              !fileListMode_,
                                                              rawFd,
                                                              fileSize,
                                                              rawHeaderSize,
                                                              1,
                                                              eventsInNewFile,
                                                              this));
        newInputFile->chunks_[0] = newChunk;
        readingFilesCount_++;
        fileQueue_.push(std::move(newInputFile));
        cvWakeup_.notify_one();
      }
    }
  }
  setMonStateSup(inRunEnd);
  //make sure threads finish reading
  unsigned numFinishedThreads = 0;
  while (numFinishedThreads < workerThreads_.size()) {
    unsigned tid = 0;
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

void FedRawDataInputSource::readWorker(unsigned int tid) {
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

    InputFile* file;
    InputChunk* chunk;

    assert(workerJob_[tid].first != nullptr && workerJob_[tid].second != nullptr);

    file = workerJob_[tid].first;
    chunk = workerJob_[tid].second;

    //skip reading initial header size in first chunk if inheriting file descriptor (already set at appropriate position)
    unsigned int bufferLeft = (chunk->offset_ == 0 && file->rawFd_ != -1) ? file->rawHeaderSize_ : 0;

    //if only one worker thread exists, use single fd for all operations
    //if more worker threads exist, use rawFd_ for only the first read operation and then close file
    int fileDescriptor;
    bool fileOpenedHere = false;

    if (numConcurrentReads_ == 1) {
      fileDescriptor = file->rawFd_;
      if (fileDescriptor == -1) {
        fileDescriptor = open(file->fileName_.c_str(), O_RDONLY);
        fileOpenedHere = true;
        file->rawFd_ = fileDescriptor;
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

    if (fileDescriptor < 0) {
      edm::LogError("FedRawDataInputSource") << "readWorker failed to open file -: " << file->fileName_
                                             << " fd:" << fileDescriptor << " error: " << strerror(errno);
      setExceptionState_ = true;
      continue;
    }
    if (fileOpenedHere) {  //fast forward to this chunk position
      off_t pos = 0;
      pos = lseek(fileDescriptor, chunk->offset_, SEEK_SET);
      if (pos == -1) {
        edm::LogError("FedRawDataInputSource")
            << "readWorker failed to seek file -: " << file->fileName_ << " fd:" << fileDescriptor << " to offset "
            << chunk->offset_ << " error: " << strerror(errno);
        setExceptionState_ = true;
        continue;
      }
    }

    LogDebug("FedRawDataInputSource") << "Reader thread opened file -: TID: " << tid << " file: " << file->fileName_
                                      << " at offset " << lseek(fileDescriptor, 0, SEEK_CUR);

    unsigned int skipped = bufferLeft;
    auto start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < readBlocks_; i++) {
      ssize_t last;

      //protect against reading into next block
      last = ::read(
          fileDescriptor, (void*)(chunk->buf_ + bufferLeft), std::min(chunk->usedSize_ - bufferLeft, eventChunkBlock_));

      if (last < 0) {
        edm::LogError("FedRawDataInputSource") << "readWorker failed to read file -: " << file->fileName_
                                               << " fd:" << fileDescriptor << " error: " << strerror(errno);
        setExceptionState_ = true;
        break;
      }
      if (last > 0)
        bufferLeft += last;
      if (last < eventChunkBlock_) {  //last read
        //check if this is last block, then total read size must match file size
        if (!(chunk->usedSize_ - skipped == i * eventChunkBlock_ + last)) {
          edm::LogError("FedRawDataInputSource")
              << "readWorker failed to read file -: " << file->fileName_ << " fd:" << fileDescriptor << " last:" << last
              << " expectedChunkSize:" << chunk->usedSize_
              << " readChunkSize:" << (skipped + i * eventChunkBlock_ + last) << " skipped:" << skipped
              << " block:" << (i + 1) << "/" << readBlocks_ << " error: " << strerror(errno);
          setExceptionState_ = true;
        }
        break;
      }
    }
    if (setExceptionState_)
      continue;

    auto end = std::chrono::high_resolution_clock::now();
    auto diff = end - start;
    std::chrono::milliseconds msec = std::chrono::duration_cast<std::chrono::milliseconds>(diff);
    LogDebug("FedRawDataInputSource") << " finished reading block -: " << (bufferLeft >> 20) << " MB"
                                      << " in " << msec.count() << " ms (" << (bufferLeft >> 20) / double(msec.count())
                                      << " GB/s)";

    if (chunk->offset_ + bufferLeft == file->fileSize_) {  //file reading finished using same fd
      close(fileDescriptor);
      fileDescriptor = -1;
      if (numConcurrentReads_ == 1)
        file->rawFd_ = -1;
    }
    if (numConcurrentReads_ > 1 && fileDescriptor != -1)
      close(fileDescriptor);

    //detect FRD event version. Skip file Header if it exists
    if (detectedFRDversion_ == 0 && chunk->offset_ == 0) {
      detectedFRDversion_ = *((uint16_t*)(chunk->buf_ + file->rawHeaderSize_));
    }
    assert(detectedFRDversion_ <= FRDHeaderMaxVersion);
    chunk->readComplete_ =
        true;  //this is atomic to secure the sequential buffer fill before becoming available for processing)
    file->chunks_[chunk->fileIndex_] = chunk;  //put the completed chunk in the file chunk vector at predetermined index
  }
}

void FedRawDataInputSource::threadError() {
  quit_threads_ = true;
  throw cms::Exception("FedRawDataInputSource:threadError") << " file reader thread error ";
}

inline void FedRawDataInputSource::setMonState(evf::FastMonState::InputState state) {
  if (fms_)
    fms_->setInState(state);
}

inline void FedRawDataInputSource::setMonStateSup(evf::FastMonState::InputState state) {
  if (fms_)
    fms_->setInStateSup(state);
}

inline bool InputFile::advance(unsigned char*& dataPosition, const size_t size) {
  //wait for chunk

  while (!waitForChunk(currentChunk_)) {
    parent_->setMonState(inWaitChunk);
    usleep(100000);
    parent_->setMonState(inChunkReceived);
    if (parent_->exceptionState())
      parent_->threadError();
  }

  dataPosition = chunks_[currentChunk_]->buf_ + chunkPosition_;
  size_t currentLeft = chunks_[currentChunk_]->size_ - chunkPosition_;

  if (currentLeft < size) {
    //we need next chunk
    while (!waitForChunk(currentChunk_ + 1)) {
      parent_->setMonState(inWaitChunk);
      usleep(100000);
      parent_->setMonState(inChunkReceived);
      if (parent_->exceptionState())
        parent_->threadError();
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

inline void InputFile::moveToPreviousChunk(const size_t size, const size_t offset) {
  //this will fail in case of events that are too large
  assert(size < chunks_[currentChunk_]->size_ - chunkPosition_);
  assert(size - offset < chunks_[currentChunk_]->size_);
  memcpy(chunks_[currentChunk_ - 1]->buf_ + offset, chunks_[currentChunk_]->buf_ + chunkPosition_, size);
  chunkPosition_ += size;
  bufferPosition_ += size;
}

inline void InputFile::rewindChunk(const size_t size) {
  chunkPosition_ -= size;
  bufferPosition_ -= size;
}

InputFile::~InputFile() {
  if (rawFd_ != -1)
    close(rawFd_);

  if (deleteFile_ && !fileName_.empty()) {
    const std::filesystem::path filePath(fileName_);
    try {
      //sometimes this fails but file gets deleted
      LogDebug("FedRawDataInputSource:InputFile") << "Deleting input file -:" << fileName_;
      std::filesystem::remove(filePath);
      return;
    } catch (const std::filesystem::filesystem_error& ex) {
      edm::LogError("FedRawDataInputSource:InputFile")
          << " - deleteFile BOOST FILESYSTEM ERROR CAUGHT -: " << ex.what() << ". Trying again.";
    } catch (std::exception& ex) {
      edm::LogError("FedRawDataInputSource:InputFile")
          << " - deleteFile std::exception CAUGHT -: " << ex.what() << ". Trying again.";
    }
    std::filesystem::remove(filePath);
  }
}

//single-buffer mode file reading
void FedRawDataInputSource::readNextChunkIntoBuffer(InputFile* file) {
  uint32_t existingSize = 0;

  if (fileDescriptor_ < 0) {
    bufferInputRead_ = 0;
    if (file->rawFd_ == -1) {
      fileDescriptor_ = open(file->fileName_.c_str(), O_RDONLY);
      if (file->rawHeaderSize_)
        lseek(fileDescriptor_, file->rawHeaderSize_, SEEK_SET);
    } else
      fileDescriptor_ = file->rawFd_;

    //skip header size in destination buffer (chunk position was already adjusted)
    bufferInputRead_ += file->rawHeaderSize_;
    existingSize += file->rawHeaderSize_;

    if (fileDescriptor_ >= 0)
      LogDebug("FedRawDataInputSource") << "opened file -: " << std::endl << file->fileName_;
    else {
      throw cms::Exception("FedRawDataInputSource:readNextChunkIntoBuffer")
          << "failed to open file " << std::endl
          << file->fileName_ << " fd:" << fileDescriptor_;
    }
    //fill chunk (skipping file header if present)
    for (unsigned int i = 0; i < readBlocks_; i++) {
      const ssize_t last = ::read(fileDescriptor_,
                                  (void*)(file->chunks_[0]->buf_ + existingSize),
                                  eventChunkBlock_ - (i == readBlocks_ - 1 ? existingSize : 0));
      bufferInputRead_ += last;
      existingSize += last;
    }

  } else {
    //continue reading
    if (file->chunkPosition_ == 0) {  //in the rare case the last byte barely fit
      for (unsigned int i = 0; i < readBlocks_; i++) {
        const ssize_t last = ::read(fileDescriptor_, (void*)(file->chunks_[0]->buf_ + existingSize), eventChunkBlock_);
        bufferInputRead_ += last;
        existingSize += last;
      }
    } else {
      //event didn't fit in last chunk, so leftover must be moved to the beginning and completed
      uint32_t existingSizeLeft = eventChunkSize_ - file->chunkPosition_;
      memmove((void*)file->chunks_[0]->buf_, file->chunks_[0]->buf_ + file->chunkPosition_, existingSizeLeft);

      //calculate amount of data that can be added
      const uint32_t blockcount = file->chunkPosition_ / eventChunkBlock_;
      const uint32_t leftsize = file->chunkPosition_ % eventChunkBlock_;

      for (uint32_t i = 0; i < blockcount; i++) {
        const ssize_t last =
            ::read(fileDescriptor_, (void*)(file->chunks_[0]->buf_ + existingSizeLeft), eventChunkBlock_);
        bufferInputRead_ += last;
        existingSizeLeft += last;
      }
      if (leftsize) {
        const ssize_t last = ::read(fileDescriptor_, (void*)(file->chunks_[0]->buf_ + existingSizeLeft), leftsize);
        bufferInputRead_ += last;
      }
      file->chunkPosition_ = 0;  //data was moved to beginning of the chunk
    }
  }
  if (bufferInputRead_ == file->fileSize_) {  // no more data in this file
    if (fileDescriptor_ != -1) {
      LogDebug("FedRawDataInputSource") << "Closing input file -: " << std::endl << file->fileName_;
      close(fileDescriptor_);
      file->rawFd_ = fileDescriptor_ = -1;
    }
  }
}

void FedRawDataInputSource::reportEventsThisLumiInSource(unsigned int lumi, unsigned int events) {
  std::lock_guard<std::mutex> lock(monlock_);
  auto itr = sourceEventsReport_.find(lumi);
  if (itr != sourceEventsReport_.end())
    itr->second += events;
  else
    sourceEventsReport_[lumi] = events;
}

std::pair<bool, unsigned int> FedRawDataInputSource::getEventReport(unsigned int lumi, bool erase) {
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

long FedRawDataInputSource::initFileList() {
  std::sort(fileNames_.begin(), fileNames_.end(), [](std::string a, std::string b) {
    if (a.rfind('/') != std::string::npos)
      a = a.substr(a.rfind('/'));
    if (b.rfind('/') != std::string::npos)
      b = b.substr(b.rfind('/'));
    return b > a;
  });

  if (!fileNames_.empty()) {
    //get run number from first file in the vector
    std::filesystem::path fileName = fileNames_[0];
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
        edm::LogInfo("FedRawDataInputSource") << "Autodetected run number in fileListMode -: " << rval;
        return rval;
      } catch (const std::exception&) {
        edm::LogWarning("FedRawDataInputSource")
            << "Unable to autodetect run number in fileListMode from file -: " << fileName;
      }
    }
  }
  return -1;
}

evf::EvFDaqDirector::FileStatus FedRawDataInputSource::getFile(unsigned int& ls,
                                                               std::string& nextFile,
                                                               uint32_t& fsize,
                                                               uint64_t& lockWaitTime) {
  if (fileListIndex_ < fileNames_.size()) {
    nextFile = fileNames_[fileListIndex_];
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
      return getFile(ls, nextFile, fsize, lockWaitTime);
    }
  }
}
