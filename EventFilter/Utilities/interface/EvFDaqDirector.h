#ifndef EVENTFILTER_UTILTIES_PLUGINS_EVFDAQDIRECTOR
#define EVENTFILTER_UTILTIES_PLUGINS_EVFDAQDIRECTOR

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "DataFormats/Provenance/interface/RunID.h"

#include "EventFilter/Utilities/interface/FFFNamingSchema.h"
#include "EventFilter/Utilities/interface/DirManager.h"

//std headers
#include <filesystem>
#include <iomanip>
#include <list>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

//system headers
#include <sys/stat.h>
#include <sys/file.h>
#include <fcntl.h>
#include <cerrno>
#include <cstring>
#include <cstdio>

#include <boost/asio.hpp>
#include <oneapi/tbb/concurrent_hash_map.h>

class SystemBounds;
class GlobalContext;
class StreamID;

namespace edm {
  class PathsAndConsumesOfModulesBase;
  class ProcessContext;
}  // namespace edm

namespace jsoncollector {
  class DataPointDefinition;

  namespace Json {
    class Value;
  }
}  // namespace jsoncollector

namespace edm {
  class ConfigurationDescriptions;
}

namespace evf {

  enum MergeType { MergeTypeNULL = 0, MergeTypeDAT = 1, MergeTypePB = 2, MergeTypeJSNDATA = 3 };

  class FastMonitoringService;

  class EvFDaqDirector {
  public:
    enum FileStatus { noFile, sameFile, newFile, newLumi, runEnded, runAbort };

    explicit EvFDaqDirector(const edm::ParameterSet& pset, edm::ActivityRegistry& reg);
    ~EvFDaqDirector();
    void initRun();
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
    void preallocate(edm::service::SystemBounds const& bounds);
    void preBeginRun(edm::GlobalContext const& globalContext);
    void postEndRun(edm::GlobalContext const& globalContext);
    void preGlobalEndLumi(edm::GlobalContext const& globalContext);
    void updateRunParams();
    void overrideRunNumber(unsigned int run) {
      run_ = run;
      updateRunParams();
    }
    std::string const& runString() const { return run_string_; }
    std::string& baseRunDir() { return run_dir_; }
    std::string& buBaseRunDir() { return bu_run_dir_; }
    std::string& buBaseRunOpenDir() { return bu_run_open_dir_; }
    bool useFileBroker() const { return useFileBroker_; }

    std::string findCurrentRunDir() { return dirManager_.findRunDir(run_); }
    std::string getInputJsonFilePath(const unsigned int ls, const unsigned int index) const;
    std::string getRawFilePath(const unsigned int ls, const unsigned int index) const;
    std::string getOpenRawFilePath(const unsigned int ls, const unsigned int index) const;
    std::string getOpenInputJsonFilePath(const unsigned int ls, const unsigned int index) const;
    std::string getDatFilePath(const unsigned int ls, std::string const& stream) const;
    std::string getOpenDatFilePath(const unsigned int ls, std::string const& stream) const;
    std::string getOpenOutputJsonFilePath(const unsigned int ls, std::string const& stream) const;
    std::string getOutputJsonFilePath(const unsigned int ls, std::string const& stream) const;
    std::string getMergedDatFilePath(const unsigned int ls, std::string const& stream) const;
    std::string getMergedDatChecksumFilePath(const unsigned int ls, std::string const& stream) const;
    std::string getOpenInitFilePath(std::string const& stream) const;
    std::string getInitFilePath(std::string const& stream) const;
    std::string getInitTempFilePath(std::string const& stream) const;
    std::string getOpenProtocolBufferHistogramFilePath(const unsigned int ls, std::string const& stream) const;
    std::string getProtocolBufferHistogramFilePath(const unsigned int ls, std::string const& stream) const;
    std::string getMergedProtocolBufferHistogramFilePath(const unsigned int ls, std::string const& stream) const;
    std::string getOpenRootHistogramFilePath(const unsigned int ls, std::string const& stream) const;
    std::string getRootHistogramFilePath(const unsigned int ls, std::string const& stream) const;
    std::string getMergedRootHistogramFilePath(const unsigned int ls, std::string const& stream) const;
    std::string getEoLSFilePathOnBU(const unsigned int ls) const;
    std::string getEoLSFilePathOnFU(const unsigned int ls) const;
    std::string getBoLSFilePathOnFU(const unsigned int ls) const;
    std::string getEoRFilePath() const;
    std::string getEoRFileName() const;
    std::string getEoRFilePathOnFU() const;
    std::string getFFFParamsFilePathOnBU() const;
    std::string getRunOpenDirPath() const { return run_dir_ + "/open"; }
    bool outputAdler32Recheck() const { return outputAdler32Recheck_; }
    void removeFile(std::string);

    FileStatus updateFuLock(unsigned int& ls,
                            std::string& nextFile,
                            uint32_t& fsize,
                            uint16_t& rawHeaderSize,
                            uint64_t& lockWaitTime,
                            bool& setExceptionState);
    void tryInitializeFuLockFile();
    unsigned int getRunNumber() const { return run_; }
    void lockInitLock();
    void unlockInitLock();
    void setFMS(evf::FastMonitoringService* fms) { fms_ = fms; }
    bool isSingleStreamThread() { return nStreams_ == 1 && nThreads_ == 1; }
    unsigned int numConcurrentLumis() const { return nConcurrentLumis_; }
    void lockFULocal();
    void unlockFULocal();
    void lockFULocal2();
    void unlockFULocal2();
    void createBoLSFile(const uint32_t lumiSection, bool checkIfExists) const;
    void createLumiSectionFiles(const uint32_t lumiSection,
                                const uint32_t currentLumiSection,
                                bool doCreateBoLS,
                                bool doCreateEoLS);
    static int parseFRDFileHeader(std::string const& rawSourcePath,
                                  int& rawFd,
                                  uint16_t& rawHeaderSize,
                                  uint16_t& rawDataType,
                                  uint32_t& lsFromHeader,
                                  int32_t& eventsFromHeader,
                                  int64_t& fileSizeFromHeader,
                                  bool requireHeader,
                                  bool retry,
                                  bool closeFile);
    bool rawFileHasHeader(std::string const& rawSourcePath, uint16_t& rawHeaderSize);
    int grabNextJsonFromRaw(std::string const& rawSourcePath,
                            int& rawFd,
                            uint16_t& rawHeaderSize,
                            int64_t& fileSizeFromHeader,
                            bool& fileFound,
                            uint32_t serverLS,
                            bool closeFile,
                            bool requireHeader = true);
    int grabNextJsonFile(std::string const& jsonSourcePath,
                         std::string const& rawSourcePath,
                         int64_t& fileSizeFromJson,
                         bool& fileFound);
    int grabNextJsonFileAndUnlock(std::filesystem::path const& jsonSourcePath);

    EvFDaqDirector::FileStatus contactFileBroker(unsigned int& serverHttpStatus,
                                                 bool& serverState,
                                                 uint32_t& serverLS,
                                                 uint32_t& closedServerLS,
                                                 std::string& nextFileJson,
                                                 std::string& nextFileRaw,
                                                 bool& rawHeader,
                                                 int maxLS);

    FileStatus getNextFromFileBroker(const unsigned int currentLumiSection,
                                     unsigned int& ls,
                                     std::string& nextFile,
                                     int& rawFd,
                                     uint16_t& rawHeaderSize,
                                     int32_t& serverEventsInNewFile_,
                                     int64_t& fileSize,
                                     uint64_t& thisLockWaitTimeUs,
                                     bool requireHeader = true);
    void createRunOpendirMaybe();
    void createProcessingNotificationMaybe() const;
    int readLastLSEntry(std::string const& file);
    unsigned int getLumisectionToStart() const;
    unsigned int getStartLumisectionFromEnv() const { return startFromLS_; }

    std::string getStreamDestinations(std::string const&) const { return std::string(""); }
    std::string getStreamMergeType(std::string const&, MergeType defaultType) const {
      return MergeTypeNames_[defaultType];
    }
    static struct flock make_flock(short type, short whence, off_t start, off_t len, pid_t pid);
    bool inputThrottled();
    bool lumisectionDiscarded(unsigned int ls);
    std::vector<std::string> const& getBUBaseDirs() const { return bu_base_dirs_all_; }
    std::vector<int> const& getBUBaseDirsNSources() const { return bu_base_dirs_nSources_; }
    void setFileListMode() { fileListMode_ = true; }
    bool fileListMode() const { return fileListMode_; }
    unsigned int lsWithFilesOpen(unsigned int ls) const;

  private:
    bool bumpFile(unsigned int& ls,
                  unsigned int& index,
                  std::string& nextFile,
                  uint32_t& fsize,
                  uint16_t& rawHeaderSize,
                  int maxLS,
                  bool& setExceptionState);
    void openFULockfileStream(bool create);
    static bool checkFileRead(char* buf, int infile, std::size_t buf_sz, std::string const& path);
    std::string inputFileNameStem(const unsigned int ls, const unsigned int index) const;
    std::string outputFileNameStem(const unsigned int ls, std::string const& stream) const;
    std::string mergedFileNameStem(const unsigned int ls, std::string const& stream) const;
    std::string initFileName(std::string const& stream) const;
    std::string eolsFileName(const unsigned int ls) const;
    std::string eorFileName() const;
    int getNFilesFromEoLS(std::string BUEoLSFile);

    std::string base_dir_;
    std::string bu_base_dir_;
    std::vector<std::string> bu_base_dirs_all_;
    std::vector<int> bu_base_dirs_nSources_;
    unsigned int run_;
    bool useFileBroker_;
    bool fileBrokerHostFromCfg_;
    std::string fileBrokerHost_;
    std::string fileBrokerPort_;
    bool fileBrokerKeepAlive_;
    bool fileBrokerUseLocalLock_;
    unsigned int fuLockPollInterval_;
    bool outputAdler32Recheck_;
    bool directorBU_;
    std::string hltSourceDirectory_;

    unsigned int startFromLS_ = 1;
    oneapi::tbb::concurrent_hash_map<unsigned int, unsigned int> lsWithFilesMap_;

    std::string hostname_;
    std::string run_string_;
    std::string run_nstring_;
    std::string pid_;
    std::string run_dir_;
    std::string bu_run_dir_;
    std::string bu_run_open_dir_;
    std::string fulockfile_;

    int bu_readlock_fd_;
    int bu_writelock_fd_;
    int fu_readwritelock_fd_;
    int fulocal_rwlock_fd_;
    int fulocal_rwlock_fd2_;

    FILE* bu_w_lock_stream;
    FILE* bu_r_lock_stream;
    FILE* fu_rw_lock_stream;
    FILE* bu_w_monitor_stream;
    FILE* bu_t_monitor_stream;

    DirManager dirManager_;

    unsigned long previousFileSize_;

    struct flock bu_w_flk;
    struct flock bu_r_flk;
    struct flock bu_w_fulk;
    struct flock bu_r_fulk;
    struct flock fu_rw_flk;
    struct flock fu_rw_fulk;

    evf::FastMonitoringService* fms_ = nullptr;

    pthread_mutex_t init_lock_ = PTHREAD_MUTEX_INITIALIZER;

    unsigned int nStreams_ = 0;
    unsigned int nThreads_ = 0;
    unsigned int nConcurrentLumis_ = 0;

    bool readEolsDefinition_ = true;
    unsigned int eolsNFilesIndex_ = 1;
    std::string stopFilePath_;
    std::string stopFilePathPid_;
    unsigned int stop_ls_override_ = 0;

    //values initialized in .cc file
    static const std::vector<std::string> MergeTypeNames_;

    //json parser
    jsoncollector::DataPointDefinition* dpd_;

    boost::asio::io_context io_service_;
    std::unique_ptr<boost::asio::ip::tcp::resolver> resolver_;
    std::unique_ptr<boost::asio::ip::tcp::resolver::results_type> endpoint_iterator_;
    std::unique_ptr<boost::asio::ip::tcp::socket> socket_;

    std::string input_throttled_file_;
    std::string discard_ls_filestem_;
    bool fileListMode_ = false;
  };
}  // namespace evf

#endif
