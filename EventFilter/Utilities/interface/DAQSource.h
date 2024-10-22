#ifndef EventFilter_Utilities_DAQSource_h
#define EventFilter_Utilities_DAQSource_h

#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <mutex>
#include <thread>

#include "oneapi/tbb/concurrent_queue.h"
#include "oneapi/tbb/concurrent_vector.h"

#include "FWCore/Sources/interface/RawInputSource.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"

#include "EventFilter/Utilities/interface/EvFDaqDirector.h"

//import InputChunk
#include "EventFilter/Utilities/interface/FedRawDataInputSource.h"

class FEDRawDataCollection;
class InputSourceDescription;
class ParameterSet;

class RawInputFile;
class DataMode;

class DataModeFRD;

namespace evf {
  class FastMonitoringService;
  namespace FastMonState {
    enum InputState : short;
  }
}  // namespace evf

class DAQSource : public edm::RawInputSource {
  friend class RawInputFile;
  friend struct InputChunk;

public:
  explicit DAQSource(edm::ParameterSet const&, edm::InputSourceDescription const&);
  ~DAQSource() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  std::pair<bool, unsigned int> getEventReport(unsigned int lumi, bool erase);
  bool useL1EventID() const { return useL1EventID_; }
  int currentLumiSection() const { return currentLumiSection_; }
  int eventRunNumber() const { return eventRunNumber_; }
  void makeEventWrapper(edm::EventPrincipal& eventPrincipal, edm::EventAuxiliary& aux) {
    makeEvent(eventPrincipal, aux);
  }
  bool fileListLoopMode() { return fileListLoopMode_; }

  edm::ProcessHistoryID& processHistoryID() { return processHistoryID_; }

protected:
  Next checkNext() override;
  void read(edm::EventPrincipal& eventPrincipal) override;
  void setMonState(evf::FastMonState::InputState state);
  void setMonStateSup(evf::FastMonState::InputState state);

private:
  void rewind_() override;
  inline evf::EvFDaqDirector::FileStatus getNextEventFromDataBlock();
  inline evf::EvFDaqDirector::FileStatus getNextDataBlock();

  void maybeOpenNewLumiSection(const uint32_t lumiSection);

  void readSupervisor();
  void dataArranger();
  void readWorker(unsigned int tid);
  void threadError();
  bool exceptionState() { return setExceptionState_; }

  //monitoring
  void reportEventsThisLumiInSource(unsigned int lumi, unsigned int events);

  long initFileList();
  evf::EvFDaqDirector::FileStatus getFile(unsigned int& ls, std::string& nextFile, uint64_t& lockWaitTime);

  //variables
  evf::FastMonitoringService* fms_ = nullptr;
  evf::EvFDaqDirector* daqDirector_ = nullptr;

  const std::string dataModeConfig_;
  uint64_t eventChunkSize_;   // for buffered read-ahead
  uint64_t maxChunkSize_;     // for buffered read-ahead
  uint64_t eventChunkBlock_;  // how much read(2) asks at the time
  unsigned int readBlocks_;
  unsigned int numBuffers_;
  unsigned int maxBufferedFiles_;
  unsigned int numConcurrentReads_;
  std::atomic<unsigned int> readingFilesCount_;

  // get LS from filename instead of event header
  const bool alwaysStartFromFirstLS_;
  const bool verifyChecksum_;
  const bool useL1EventID_;
  const std::vector<unsigned int> testTCDSFEDRange_;
  std::vector<std::string> listFileNames_;
  bool useFileBroker_;
  //std::vector<std::string> fileNamesSorted_;

  const bool fileListMode_;
  unsigned int fileListIndex_ = 0;
  const bool fileListLoopMode_;
  unsigned int loopModeIterationInc_ = 0;

  edm::RunNumber_t runNumber_;
  std::string fuOutputDir_;

  edm::ProcessHistoryID processHistoryID_;

  unsigned int currentLumiSection_;
  uint32_t eventRunNumber_ = 0;
  uint32_t GTPEventID_ = 0;
  unsigned int eventsThisLumi_;
  unsigned long eventsThisRun_ = 0;
  std::default_random_engine rng_;

  /*
   *
   * Multithreaded file reader
   *
   **/

  typedef std::pair<RawInputFile*, InputChunk*> ReaderInfo;

  std::unique_ptr<RawInputFile> currentFile_;
  bool chunkIsFree_ = false;

  bool startedSupervisorThread_ = false;
  std::unique_ptr<std::thread> readSupervisorThread_;
  std::unique_ptr<std::thread> dataArrangerThread_;
  std::vector<std::thread*> workerThreads_;

  tbb::concurrent_queue<unsigned int> workerPool_;
  std::vector<ReaderInfo> workerJob_;

  tbb::concurrent_queue<InputChunk*> freeChunks_;
  tbb::concurrent_queue<std::unique_ptr<RawInputFile>> fileQueue_;

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
  std::mutex fileDeleteLock_;
  std::vector<int> streamFileTracker_;
  unsigned int checkEvery_ = 10;

  //supervisor thread wakeup
  std::mutex mWakeup_;
  std::condition_variable cvWakeup_;

  //variables for the single buffered mode
  int fileDescriptor_ = -1;

  std::atomic<bool> threadInit_;

  std::map<unsigned int, unsigned int> sourceEventsReport_;
  std::mutex monlock_;

  std::shared_ptr<DataMode> dataMode_;
};

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
  bool advance(unsigned char*& dataPosition, const size_t size);
  void advance(const size_t size) {
    chunkPosition_ += size;
    bufferPosition_ += size;
  }

private:
  DAQSource* sourceParent_;
};

#endif  // EventFilter_Utilities_DAQSource_h
