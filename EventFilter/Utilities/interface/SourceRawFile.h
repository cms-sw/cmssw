#ifndef EventFilter_Utilities_SourceRawFile_h
#define EventFilter_Utilities_SourceRawFile_h

//#include <condition_variable>
//#include <cstdio>
//#include <filesystem>
//#include <memory>
//#include <mutex>
//#include <thread>
#include <memory>
#include <queue>

#include "FWCore/Framework/interface/EventPrincipal.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "EventFilter/Utilities/interface/FedRawDataInputSource.h"

//used by some models that use FEDRawDataCollection
class UnpackedRawEventWrapper {
public:
  UnpackedRawEventWrapper() {}
  ~UnpackedRawEventWrapper() {}
  void setError(std::string msg) {
    errmsg_ = msg;
    error_ = true;
  }
  void setChecksumError(std::string msg) {
    errmsg_ = msg;
    checksumError_ = true;
  }
  void setRawData(FEDRawDataCollection* rawData) { rawData_.reset(rawData); }
  void setAux(edm::EventAuxiliary* aux) { aux_.reset(aux); }
  void setRun(uint32_t run) { run_ = run; }
  FEDRawDataCollection* rawData() { return rawData_.get(); }
  std::unique_ptr<FEDRawDataCollection>& rawDataRef() { return rawData_; }
  edm::EventAuxiliary* aux() { return aux_.get(); }
  uint32_t run() const { return run_; }
  bool checksumError() const { return checksumError_; }
  bool error() const { return error_; }
  std::string const& errmsg() { return errmsg_; }

private:
  std::unique_ptr<FEDRawDataCollection> rawData_;
  std::unique_ptr<edm::EventAuxiliary> aux_;
  uint32_t run_;
  bool checksumError_ = false;
  bool error_ = false;
  std::string errmsg_;
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
  std::vector<uint64_t> bufferEnds_;
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
    bufferEnds_.push_back(fileSize);
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
    size_t prevAccumSize = diskFileSizes_.back();
    numFiles_++;
    fileNames_.push_back(name);
    fileOrder_.push_back(fileOrder_.size());
    diskFileSizes_.push_back(size);
    fileSizes_.push_back(0);
    bufferOffsets_.push_back(prevOffset + prevSize);
    bufferEnds_.push_back(prevAccumSize + size);
  }

  bool waitForChunk(unsigned int chunkid) {
    //some atomics to make sure everything is cache synchronized for the main thread
    return chunks_[chunkid] != nullptr && chunks_[chunkid]->readComplete_;
  }
  bool advance(std::mutex& m, std::condition_variable& cv, unsigned char*& dataPosition, const size_t size);
  bool advanceSimple(unsigned char*& dataPosition, const size_t size) {
    size_t currentLeft = chunks_[currentChunk_]->size_ - chunkPosition_;
    if (currentLeft < size)
      return true;
    dataPosition = chunks_[currentChunk_]->buf_ + chunkPosition_;
    chunkPosition_ += size;
    bufferPosition_ += size;
    return false;
  }
  void resetPos() {
    chunkPosition_ = 0;
    bufferPosition_ = 0;
  }
  void moveToPreviousChunk(const size_t size, const size_t offset);
  void rewindChunk(const size_t size);
  void unsetDeleteFile() { deleteFile_ = false; }
  void randomizeOrder(std::default_random_engine& rng) {
    std::shuffle(std::begin(fileOrder_), std::end(fileOrder_), rng);
  }
  uint64_t currentChunkSize() const { return chunks_[currentChunk_]->size_; }
  int64_t fileSizeLeft() const { return (int64_t)fileSize_ - (int64_t)bufferPosition_; }
  int64_t fileSizeLeft(size_t fidx) const { return (int64_t)diskFileSizes_[fidx] - (int64_t)bufferOffsets_[fidx]; }

  bool complete() const { return bufferPosition_ == fileSize_; }

  bool buffersComplete() const {
    unsigned complete = 0;
    for (size_t fidx = 0; fidx < bufferOffsets_.size(); fidx++) {
      if ((int64_t)bufferEnds_[fidx] - (int64_t)bufferOffsets_[fidx] == 0)
        complete++;
    }
    if (complete && complete < bufferOffsets_.size())
      throw cms::Exception("InputFile") << "buffers are inconsistent for input files with primary " << fileName_;
    return complete > 0;
  }
};

class DAQSource;

class RawInputFile : public InputFile {
public:
  RawInputFile(evf::EvFDaqDirector::FileStatus status,
               unsigned int lumi = 0,
               std::string const& name = std::string(),
               bool deleteFile = true,
               int rawFd = -1,
               uint64_t fileSize = 0,
               uint16_t rawHeaderSize = 0,
               uint32_t nChunks = 0,
               int nEvents = 0,
               DAQSource* parent = nullptr)
      : InputFile(status, lumi, name, deleteFile, rawFd, fileSize, rawHeaderSize, nChunks, nEvents, nullptr),
        sourceParent_(parent) {}
  bool advance(std::mutex& m, std::condition_variable& cv, unsigned char*& dataPosition, const size_t size);
  void advance(const size_t size) {
    chunkPosition_ += size;
    bufferPosition_ += size;
  }
  void advanceBuffers(const size_t size) {
    for (size_t bidx = 0; bidx < bufferOffsets_.size(); bidx++)
      bufferOffsets_[bidx] += size;
  }
  void advanceBuffer(const size_t size, const size_t bidx) { bufferOffsets_[bidx] += size; }
  void queue(UnpackedRawEventWrapper* ec) {
    if (!frdcQueue_.get())
      frdcQueue_ = std::make_unique<std::queue<std::unique_ptr<UnpackedRawEventWrapper>>>();
    std::unique_ptr<UnpackedRawEventWrapper> uptr(ec);
    frdcQueue_->push(std::move(uptr));
  }
  void popQueue(std::unique_ptr<UnpackedRawEventWrapper>& uptr) {
    uptr = std::move(frdcQueue_->front());
    frdcQueue_->pop();
  }

private:
  DAQSource* sourceParent_;
  //optional unpacked raw data queue (currently here because DAQSource controls lifetime of the RawInputfile)
  std::unique_ptr<std::queue<std::unique_ptr<UnpackedRawEventWrapper>>> frdcQueue_;
};

#endif  // EventFilter_Utilities_SourceRawFile_h
