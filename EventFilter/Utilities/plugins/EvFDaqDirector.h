#ifndef EVENTFILTER_UTILTIES_PLUGINS_EVFDAQDIRECTOR
#define EVENTFILTER_UTILTIES_PLUGINS_EVFDAQDIRECTOR

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "DataFormats/Provenance/interface/RunID.h"

#include "EventFilter/Utilities/interface/FFFNamingSchema.h"
#include "DirManager.h"

//std headers
#include <string>
#include <sstream>
#include <iomanip>
#include <vector>

//system headers
//#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>

class SystemBounds;
class GlobalContext;
class StreamID;

namespace evf{

  class FastMonitoringService;

  class EvFDaqDirector
    {
    public:

      enum FileStatus { noFile, sameFile, newFile, newLumi, runEnded };

      explicit EvFDaqDirector( const edm::ParameterSet &pset, edm::ActivityRegistry& reg );
      ~EvFDaqDirector(){}
      void preallocate(edm::service::SystemBounds const& bounds);
      void preBeginRun(edm::GlobalContext const& globalContext);
      void postEndRun(edm::GlobalContext const& globalContext);
      void preSourceEvent(edm::StreamID const& streamID);
      //void preBeginRun(edm::RunID const& id, edm::Timestamp const& ts);
      //void postEndRun(edm::Run const& run, edm::EventSetup const& es);
      std::string &baseDir(){return base_dir_;}
      std::string &fuBaseDir(){return run_dir_;}
      std::string &smBaseDir(){return sm_base_dir_;}
      std::string &buBaseDir(){return bu_run_dir_;}
      std::string &buBaseOpenDir(){return bu_run_open_dir_;}
      std::string &monitorBaseDir(){return monitor_base_dir_;}

      std::string findHighestRunDir(){ return dirManager_.findHighestRunDir();}
      std::string findCurrentRunDir(){ return dirManager_.findRunDir(run_);}
      std::string findHighestRunDirStem();
      unsigned int findHighestRun(){return dirManager_.findHighestRun();}
      std::string getRawFilePath(const unsigned int ls, const unsigned int index) const;
      std::string getOpenRawFilePath(const unsigned int ls, const unsigned int index) const;
      std::string getOpenDatFilePath(const unsigned int ls, std::string const& stream) const;
      std::string getOutputJsonFilePath(const unsigned int ls, std::string const& stream) const;
      std::string getMergedDatFilePath(const unsigned int ls, std::string const& stream) const;
      std::string getInitFilePath(std::string const& stream) const;
      std::string getEoLSFilePathOnBU(const unsigned int ls) const;
      std::string getEoLSFilePathOnFU(const unsigned int ls) const;
      std::string getEoRFilePath() const;
      std::string getEoRFilePathOnFU() const;
      std::string getPathForFU() const;
      void removeFile(unsigned int ls, unsigned int index);
      void removeFile(std::string );
      void updateBuLock(unsigned int ls);
      int readBuLock();
      // DEPRECATED
      //int updateFuLock(unsigned int &ls);
      FileStatus updateFuLock(unsigned int& ls, std::string& nextFile, uint32_t& fsize);
      void writeLsStatisticsBU(unsigned int, unsigned int, unsigned long long, long long);
      void writeLsStatisticsFU(unsigned int ls, unsigned int events, timeval completion_time){}
      void writeDiskAndThrottleStat(double, int, int);
      void tryInitializeFuLockFile();
      unsigned int getRunNumber() const { return run_; }
      unsigned int getJumpLS() const { return jumpLS_; }
      unsigned int getJumpIndex() const { return jumpIndex_; }
      std::string getJumpFilePath() const { return bu_run_dir_ + "/" + fffnaming::inputRawFileName(getRunNumber(),jumpLS_,jumpIndex_); }
      bool getTestModeNoBuilderUnit() { return testModeNoBuilderUnit_;}
      FILE * maybeCreateAndLockFileHeadForStream(unsigned int ls, std::string &stream);
      void unlockAndCloseMergeStream();
      void lockInitLock();
      void unlockInitLock();
      void setFMS(evf::FastMonitoringService* fms) {fms_=fms;}
      void updateFileIndex(int const& fileIndex) {currentFileIndex_=fileIndex;}
      std::vector<int>* getStreamFileTracker() {return &streamFileTracker_;}
      bool isSingleStreamThread() {return nStreams_==1 && nThreads_==1;}


    private:
      bool bulock();
      bool fulock();
      // DEPRECATED
      // bool copyRunDirToSlaves();
      // This functionality is for emulator running only
      bool mkFuRunDir();
      // This functionality is for emulator running only
      bool createOutputDirectory();
      bool bumpFile(unsigned int& ls, unsigned int& index, std::string& nextFile, uint32_t& fsize);
      bool findHighestActiveLS(unsigned int& startingLS) const;
      void openFULockfileStream(std::string& fuLockFilePath, bool create);
      std::string inputFileNameStem(const unsigned int ls, const unsigned int index) const;
      std::string outputFileNameStem(const unsigned int ls, std::string const& stream) const;
      std::string mergedFileNameStem(const unsigned int ls, std::string const& stream) const;
      std::string initFileName(std::string const& stream) const;
      std::string eolsFileName(const unsigned int ls) const;
      std::string eorFileName() const;

      bool testModeNoBuilderUnit_;
      std::string base_dir_;
      std::string bu_base_dir_;
      std::string sm_base_dir_;
      std::string monitor_base_dir_;
      bool directorBu_;
      unsigned int run_;

      std::string hostname_;
      std::string run_string_;
      std::string run_dir_;
      std::string bu_run_dir_;
      std::string bu_run_open_dir_;

      int bu_readlock_fd_;
      int bu_writelock_fd_;
      int fu_readwritelock_fd_;
      int data_readwrite_fd_;

      FILE * bu_w_lock_stream;
      FILE * bu_r_lock_stream;
      FILE * fu_rw_lock_stream;
      FILE * bu_w_monitor_stream;
      FILE * bu_t_monitor_stream;
      FILE * data_rw_stream;

      DirManager dirManager_;

      unsigned long previousFileSize_;
      unsigned int jumpLS_, jumpIndex_;

      struct flock bu_w_flk;
      struct flock bu_r_flk;
      struct flock bu_w_fulk;
      struct flock bu_r_fulk;
      struct flock fu_rw_flk;
      struct flock fu_rw_fulk;
      struct flock data_rw_flk;
      struct flock data_rw_fulk;

      evf::FastMonitoringService * fms_ = nullptr;
      std::vector<int> streamFileTracker_;
      int currentFileIndex_ = -1;

      pthread_mutex_t init_lock_ = PTHREAD_MUTEX_INITIALIZER;

      unsigned int nStreams_=0;
      unsigned int nThreads_=0;

  };
}

#endif

