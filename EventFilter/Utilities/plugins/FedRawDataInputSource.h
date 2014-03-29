#ifndef EventFilter_Utilities_FedRawDataInputSource_h
#define EventFilter_Utilities_FedRawDataInputSource_h

#include <memory>
#include <stdio.h>
#include <mutex>
#include <condition_variable>
#include <thread>
#include "tbb/concurrent_queue.h"
#include "tbb/concurrent_vector.h"

#include "boost/filesystem.hpp"

#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "EventFilter/Utilities/plugins/EvFDaqDirector.h"
#include "FWCore/Sources/interface/RawInputSource.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Sources/interface/DaqProvenanceHelper.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "IOPool/Streamer/interface/FRDEventMessage.h"

class FEDRawDataCollection;
class InputSourceDescription;
class ParameterSet;

namespace evf {
class FastMonitoringService;
}

namespace jsoncollector {
class DataPointDefinition;
}


class FedRawDataInputSource: public edm::RawInputSource {

class InputFile;

public:
  explicit FedRawDataInputSource(edm::ParameterSet const&,edm::InputSourceDescription const&);
  virtual ~FedRawDataInputSource();

protected:
  virtual bool checkNextEvent() override;
  virtual void read(edm::EventPrincipal& eventPrincipal) override;

private:
  virtual void preForkReleaseResources() override;
  virtual void postForkReacquireResources(boost::shared_ptr<edm::multicore::MessageReceiverForSource>) override;
  virtual void rewind_() override;

  void maybeOpenNewLumiSection(const uint32_t lumiSection);
  evf::EvFDaqDirector::FileStatus nextEvent();
  evf::EvFDaqDirector::FileStatus getNextEvent();
  edm::Timestamp fillFEDRawDataCollection(std::auto_ptr<FEDRawDataCollection>&) const;
  void deleteFile(std::string const&);
  int grabNextJsonFile(boost::filesystem::path const&);
  void renameToNextFree(std::string const& fileName) const;

  void readSupervisor();
  void readWorker(unsigned int tid);
  void threadError();
  bool exceptionState() {return setExceptionState_;}

  //functions for single buffered reader
  void readNextChunkIntoBuffer(InputFile *file);

  //variables
  evf::FastMonitoringService* fms_=nullptr;
  evf::EvFDaqDirector* daqDirector_=nullptr;

  std::string defPath_;

  unsigned int eventChunkSize_; // for buffered read-ahead
  unsigned int eventChunkBlock_; // how much read(2) asks at the time
  unsigned int readBlocks_;
  unsigned int numBuffers_;
  unsigned int numConcurrentReads_;

  // get LS from filename instead of event header
  const bool getLSFromFilename_;
  const bool verifyAdler32_;
  const bool testModeNoBuilderUnit_;

  const edm::RunNumber_t runNumber_;

  const std::string fuOutputDir_;

  const edm::DaqProvenanceHelper daqProvenanceHelper_;

  std::unique_ptr<FRDEventMsgView> event_;

  edm::EventID eventID_;
  edm::ProcessHistoryID processHistoryID_;

  unsigned int currentLumiSection_;
  unsigned int eventsThisLumi_;
  unsigned long eventsThisRun_ = 0;

  jsoncollector::DataPointDefinition *dpd_;

  /*
   *
   * Multithreaded file reader
   *
   **/

  struct InputChunk {
    unsigned char * buf_;
    InputChunk *next_ = nullptr;
    uint32_t size_;
    uint32_t usedSize_ = 0;
    unsigned int index_;
    unsigned int offset_;
    unsigned int fileIndex_;
    std::atomic<bool> readComplete_;

    InputChunk(unsigned int index, uint32_t size): size_(size),index_(index) {
      buf_ = new unsigned char[size_];
      reset(0,0,0);
    }
    void reset(unsigned int newOffset, unsigned int toRead, unsigned int fileIndex) {
      offset_=newOffset;
      usedSize_=toRead;
      fileIndex_=fileIndex;
      readComplete_=false;
    }

    ~InputChunk() {delete[] buf_;}
  };

  struct InputFile {
    FedRawDataInputSource *parent_;
    evf::EvFDaqDirector::FileStatus status_;
    unsigned int lumi_;
    std::string fileName_;
    uint32_t fileSize_;
    uint32_t nChunks_;
    unsigned int nEvents_;
    unsigned int nProcessed_;

    tbb::concurrent_vector<InputChunk*> chunks_;

    uint32_t  bufferPosition_ = 0;
    uint32_t  chunkPosition_ = 0;
    unsigned int currentChunk_ = 0;

    InputFile(evf::EvFDaqDirector::FileStatus status, unsigned int lumi = 0, std::string const& name = std::string(), 
	uint32_t fileSize =0, uint32_t nChunks=0, unsigned int nEvents=0, FedRawDataInputSource *parent = nullptr):
      parent_(parent),
      status_(status),
      lumi_(lumi),
      fileName_(name),
      fileSize_(fileSize),
      nChunks_(nChunks),
      nEvents_(nEvents),
      nProcessed_(0)
    {
      for (unsigned int i=0;i<nChunks;i++)
	chunks_.push_back(nullptr);
    }

    InputFile(std::string & name):fileName_(name) {}

    bool waitForChunk(unsigned int chunkid) {
      //some atomics to make sure everything is cache synchronized for the main thread
      return chunks_[chunkid]!=nullptr && chunks_[chunkid]->readComplete_;
    }
    bool advance(unsigned char* & dataPosition, const size_t size);
    void moveToPreviousChunk(const size_t size, const size_t offset);
    void rewindChunk(const size_t size);
  };

  typedef std::pair<InputFile*,InputChunk*> ReaderInfo;

  InputFile *currentFile_ = nullptr;
  bool chunkIsFree_=false;

  bool startedSupervisorThread_ = false;
  std::unique_ptr<std::thread> readSupervisorThread_;
  std::vector<std::thread*> workerThreads_;

  tbb::concurrent_queue<unsigned int> workerPool_;
  std::vector<ReaderInfo> workerJob_;

  tbb::concurrent_queue<InputChunk*> freeChunks_;
  tbb::concurrent_queue<InputFile*> fileQueue_;

  std::mutex mReader_;
  std::vector<std::condition_variable*> cvReader_;

  bool quit_threads_=false;
  std::vector<bool> thread_quit_signal;
  bool setExceptionState_ = false;
  std::mutex startupLock_;
  std::condition_variable startupCv_;

  int currentFileIndex_ = -1;
  std::list<std::pair<int,InputFile*>> filesToDelete_;
  std::list<std::pair<int,std::string>> fileNamesToDelete_;
  std::vector<int> *streamFileTrackerPtr_ = nullptr;
  unsigned int nStreams_ = 0;
  unsigned int checkEvery_ = 10;

  //supervisor thread wakeup
  std::mutex mWakeup_;
  std::condition_variable cvWakeup_;

  //variables for the single buffered mode
  bool singleBufferMode_;
  int fileDescriptor_ = -1;
  uint32_t bufferInputRead_ = 0;

  std::atomic<bool> threadInit_;

};

#endif // EventFilter_Utilities_FedRawDataInputSource_h

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
