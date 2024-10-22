#ifndef EventFilter_Utilities_FedRawDataInputSource_h
#define EventFilter_Utilities_FedRawDataInputSource_h

#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <mutex>
#include <thread>
#include <random>
#include <algorithm>

#include "oneapi/tbb/concurrent_queue.h"
#include "oneapi/tbb/concurrent_vector.h"

#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "EventFilter/Utilities/interface/EvFDaqDirector.h"
#include "FWCore/Sources/interface/RawInputSource.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Sources/interface/DaqProvenanceHelper.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "IOPool/Streamer/interface/FRDEventMessage.h"

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"

class FEDRawDataCollection;
class InputSourceDescription;
class ParameterSet;

class InputFile;
struct InputChunk;

namespace evf {
  class FastMonitoringService;
  namespace FastMonState {
    enum InputState : short;
  }
}  // namespace evf

class FedRawDataInputSource : public edm::RawInputSource {
  friend class InputFile;
  friend struct InputChunk;

public:
  explicit FedRawDataInputSource(edm::ParameterSet const&, edm::InputSourceDescription const&);
  ~FedRawDataInputSource() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  std::pair<bool, unsigned int> getEventReport(unsigned int lumi, bool erase);

protected:
  Next checkNext() override;
  void read(edm::EventPrincipal& eventPrincipal) override;
  void setMonState(evf::FastMonState::InputState state);
  void setMonStateSup(evf::FastMonState::InputState state);

private:
  void rewind_() override;

  void maybeOpenNewLumiSection(const uint32_t lumiSection);
  evf::EvFDaqDirector::FileStatus nextEvent();
  evf::EvFDaqDirector::FileStatus getNextEvent();
  edm::Timestamp fillFEDRawDataCollection(FEDRawDataCollection& rawData, bool& tcdsInRange);

  void readSupervisor();
  void readWorker(unsigned int tid);
  void threadError();
  bool exceptionState() { return setExceptionState_; }

  //functions for single buffered reader
  void readNextChunkIntoBuffer(InputFile* file);

  //monitoring
  void reportEventsThisLumiInSource(unsigned int lumi, unsigned int events);

  long initFileList();
  evf::EvFDaqDirector::FileStatus getFile(unsigned int& ls,
                                          std::string& nextFile,
                                          uint32_t& fsize,
                                          uint64_t& lockWaitTime);

  //variables
  evf::FastMonitoringService* fms_ = nullptr;
  evf::EvFDaqDirector* daqDirector_ = nullptr;

  std::string defPath_;

  unsigned int eventChunkSize_;   // for buffered read-ahead
  unsigned int eventChunkBlock_;  // how much read(2) asks at the time
  unsigned int readBlocks_;
  unsigned int numBuffers_;
  unsigned int maxBufferedFiles_;
  unsigned int numConcurrentReads_;
  std::atomic<unsigned int> readingFilesCount_;

  // get LS from filename instead of event header
  const bool getLSFromFilename_;
  const bool alwaysStartFromFirstLS_;
  const bool verifyChecksum_;
  const bool useL1EventID_;
  const std::vector<unsigned int> testTCDSFEDRange_;
  std::vector<std::string> fileNames_;
  bool useFileBroker_;
  //std::vector<std::string> fileNamesSorted_;

  const bool fileListMode_;
  unsigned int fileListIndex_ = 0;
  const bool fileListLoopMode_;
  unsigned int loopModeIterationInc_ = 0;

  edm::RunNumber_t runNumber_;
  std::string fuOutputDir_;

  const edm::DaqProvenanceHelper daqProvenanceHelper_;

  std::unique_ptr<edm::streamer::FRDEventMsgView> event_;

  edm::EventID eventID_;
  edm::ProcessHistoryID processHistoryID_;

  unsigned int currentLumiSection_;
  uint32_t eventRunNumber_ = 0;
  uint32_t GTPEventID_ = 0;
  uint32_t L1EventID_ = 0;
  unsigned char* tcds_pointer_;
  unsigned int eventsThisLumi_;
  unsigned long eventsThisRun_ = 0;

  uint16_t MINTCDSuTCAFEDID_ = FEDNumbering::MINTCDSuTCAFEDID;
  uint16_t MAXTCDSuTCAFEDID_ = FEDNumbering::MAXTCDSuTCAFEDID;

  /*
   *
   * Multithreaded file reader
   *
   **/

  typedef std::pair<InputFile*, InputChunk*> ReaderInfo;

  uint16_t detectedFRDversion_ = 0;
  std::unique_ptr<InputFile> currentFile_;
  bool chunkIsFree_ = false;

  bool startedSupervisorThread_ = false;
  std::unique_ptr<std::thread> readSupervisorThread_;
  std::vector<std::thread*> workerThreads_;

  tbb::concurrent_queue<unsigned int> workerPool_;
  std::vector<ReaderInfo> workerJob_;

  tbb::concurrent_queue<InputChunk*> freeChunks_;
  tbb::concurrent_queue<std::unique_ptr<InputFile>> fileQueue_;

  std::mutex mReader_;
  std::vector<std::unique_ptr<std::condition_variable>> cvReader_;
  std::vector<unsigned int> tid_active_;

  std::atomic<bool> quit_threads_;
  std::vector<unsigned int> thread_quit_signal;
  bool setExceptionState_ = false;
  std::mutex startupLock_;
  std::condition_variable startupCv_;

  int currentFileIndex_ = -1;
  std::list<std::pair<int, std::unique_ptr<InputFile>>> filesToDelete_;
  std::list<std::pair<int, std::string>> fileNamesToDelete_;
  std::mutex fileDeleteLock_;
  std::vector<int> streamFileTracker_;
  unsigned int checkEvery_ = 10;

  //supervisor thread wakeup
  std::mutex mWakeup_;
  std::condition_variable cvWakeup_;

  //variables for the single buffered mode
  bool singleBufferMode_;
  int fileDescriptor_ = -1;
  uint32_t bufferInputRead_ = 0;

  std::atomic<bool> threadInit_;

  std::map<unsigned int, unsigned int> sourceEventsReport_;
  std::mutex monlock_;
};

struct InputChunk {
  unsigned char* buf_;
  InputChunk* next_ = nullptr;
  uint64_t size_;
  uint64_t usedSize_ = 0;
  //unsigned int index_;
  uint64_t offset_;
  unsigned int fileIndex_;
  std::atomic<bool> readComplete_;

  InputChunk(uint64_t size) : size_(size) {
    buf_ = new unsigned char[size_];
    reset(0, 0, 0);
  }
  void reset(uint64_t newOffset, uint64_t toRead, unsigned int fileIndex) {
    offset_ = newOffset;
    usedSize_ = toRead;
    fileIndex_ = fileIndex;
    readComplete_ = false;
  }

  bool resize(uint64_t wantedSize, uint64_t maxSize) {
    if (wantedSize > maxSize)
      return false;
    if (size_ < wantedSize) {
      size_ = uint64_t(wantedSize * 1.05);
      delete[] buf_;
      buf_ = new unsigned char[size_];
    }
    return true;
  }

  ~InputChunk() { delete[] buf_; }
};

class InputFile {
public:
  FedRawDataInputSource* parent_;
  evf::EvFDaqDirector::FileStatus status_;
  unsigned int lumi_;
  std::string fileName_;
  //used by DAQSource
  std::vector<std::string> fileNames_;
  std::vector<uint64_t> diskFileSizes_;
  std::vector<uint64_t> bufferOffsets_;
  std::vector<uint64_t> fileSizes_;
  std::vector<unsigned int> fileOrder_;
  bool deleteFile_;
  int rawFd_;
  uint64_t fileSize_;
  uint16_t rawHeaderSize_;
  uint16_t nChunks_;
  uint16_t numFiles_;
  int nEvents_;
  unsigned int nProcessed_;

  tbb::concurrent_vector<InputChunk*> chunks_;

  uint32_t bufferPosition_ = 0;
  uint32_t chunkPosition_ = 0;
  unsigned int currentChunk_ = 0;

  InputFile(evf::EvFDaqDirector::FileStatus status,
            unsigned int lumi = 0,
            std::string const& name = std::string(),
            bool deleteFile = true,
            int rawFd = -1,
            uint64_t fileSize = 0,
            uint16_t rawHeaderSize = 0,
            uint16_t nChunks = 0,
            int nEvents = 0,
            FedRawDataInputSource* parent = nullptr)
      : parent_(parent),
        status_(status),
        lumi_(lumi),
        fileName_(name),
        deleteFile_(deleteFile),
        rawFd_(rawFd),
        fileSize_(fileSize),
        rawHeaderSize_(rawHeaderSize),
        nChunks_(nChunks),
        numFiles_(1),
        nEvents_(nEvents),
        nProcessed_(0) {
    fileNames_.push_back(name);
    fileOrder_.push_back(fileOrder_.size());
    diskFileSizes_.push_back(fileSize);
    fileSizes_.push_back(0);
    bufferOffsets_.push_back(0);
    chunks_.reserve(nChunks_);
    for (unsigned int i = 0; i < nChunks; i++)
      chunks_.push_back(nullptr);
  }
  virtual ~InputFile();

  void setChunks(uint16_t nChunks) {
    nChunks_ = nChunks;
    chunks_.clear();
    chunks_.reserve(nChunks_);
    for (unsigned int i = 0; i < nChunks_; i++)
      chunks_.push_back(nullptr);
  }

  void appendFile(std::string const& name, uint64_t size) {
    size_t prevOffset = bufferOffsets_.back();
    size_t prevSize = diskFileSizes_.back();
    numFiles_++;
    fileNames_.push_back(name);
    fileOrder_.push_back(fileOrder_.size());
    diskFileSizes_.push_back(size);
    fileSizes_.push_back(0);
    bufferOffsets_.push_back(prevOffset + prevSize);
  }

  bool waitForChunk(unsigned int chunkid) {
    //some atomics to make sure everything is cache synchronized for the main thread
    return chunks_[chunkid] != nullptr && chunks_[chunkid]->readComplete_;
  }
  bool advance(unsigned char*& dataPosition, const size_t size);
  void moveToPreviousChunk(const size_t size, const size_t offset);
  void rewindChunk(const size_t size);
  void unsetDeleteFile() { deleteFile_ = false; }
  void randomizeOrder(std::default_random_engine& rng) {
    std::shuffle(std::begin(fileOrder_), std::end(fileOrder_), rng);
  }
  uint64_t currentChunkSize() const { return chunks_[currentChunk_]->size_; }
  int64_t fileSizeLeft() const { return (int64_t)fileSize_ - (int64_t)bufferPosition_; }
};

#endif  // EventFilter_Utilities_FedRawDataInputSource_h

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
