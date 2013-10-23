#ifndef EVENTFILTER_UTILTIES_PLUGINS_EVFDAQDIRECTOR
#define EVENTFILTER_UTILTIES_PLUGINS_EVFDAQDIRECTOR

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "DataFormats/Provenance/interface/RunID.h"

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

namespace evf{
  class EvFDaqDirector 
    {
    public:
      
      explicit EvFDaqDirector( const edm::ParameterSet &pset, edm::ActivityRegistry& reg ); 
      ~EvFDaqDirector(){}
      void preBeginRun(edm::RunID const& id, edm::Timestamp const& ts);
      void postEndRun(edm::Run const& run, edm::EventSetup const& es);
      std::string &baseDir(){return base_dir_;}
      std::string &smBaseDir(){return sm_base_dir_;}
      std::string &buBaseDir(){return bu_run_dir_;}
      std::string &buBaseOpenDir(){return bu_run_open_dir_;}
      std::string &monitorBaseDir(){return monitor_base_dir_;}

      std::string findHighestRunDir(){ return dirManager_.findHighestRunDir();}
      std::string findCurrentRunDir(){ return dirManager_.findRunDir(run_);}
      std::string findHighestRunDirStem();
      unsigned int findHighestRun(){return dirManager_.findHighestRun();}
      std::string getRawFilePath(unsigned int ls, unsigned int index);
      std::string getOpenRawFilePath(unsigned int ls, unsigned int index);
      std::string getPathForFU();
      void removeFile(unsigned int ls, unsigned int index);
      void removeFile(std::string );
      void updateBuLock(unsigned int ls);
      int readBuLock();
      // DEPRECATED
      //int updateFuLock(unsigned int &ls);
      bool updateFuLock(unsigned int& ls, unsigned int& index, bool& eorSeen);
      void writeLsStatisticsBU(unsigned int, unsigned int, unsigned long long, long long);
      void writeLsStatisticsFU(unsigned int ls, unsigned int events, timeval completion_time){}
      void writeDiskAndThrottleStat(double, int, int);
      void tryInitializeFuLockFile();
      unsigned int getJumpLS() const { return jumpLS_; }
      unsigned int getJumpIndex() const { return jumpIndex_; }
      bool getTestModeNoBuilderUnit() { return testModeNoBuilderUnit_;}
      FILE * maybeCreateAndLockFileHeadForStream(unsigned int ls, std::string &stream);
      void unlockAndCloseMergeStream();
      std::string formatRawFilePath(unsigned int ls, unsigned int index);
      std::string formatOpenRawFilePath(unsigned int ls, unsigned int index);
      std::string formatMergeFilePath(unsigned int ls, std::string &stream);
      std::string formatEndOfLS(unsigned int ls);

    private:
      bool bulock();
      bool fulock();
      // DEPRECATED
      // bool copyRunDirToSlaves();
      // This functionality is for emulator running only
      bool mkFuRunDir();
      // This functionality is for emulator running only
      bool createOutputDirectory();
      bool bumpFile(unsigned int& ls, unsigned int& index);
      bool findHighestActiveLS(unsigned int& startingLS) const;
      void openFULockfileStream(std::string& fuLockFilePath, bool create);

      bool testModeNoBuilderUnit_;
      std::string base_dir_;
      std::string bu_base_dir_;
      std::string sm_base_dir_;
      std::string monitor_base_dir_;
      bool directorBu_;
      unsigned int run_;

      std::string hostname_;
      std::string run_dir_name_;
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
  };
}

#endif

