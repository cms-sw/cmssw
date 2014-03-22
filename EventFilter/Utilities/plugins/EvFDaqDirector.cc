#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/SystemBounds.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "EventFilter/Utilities/plugins/EvFDaqDirector.h"
#include "EventFilter/Utilities/plugins/FastMonitoringService.h"


#include <iostream>
#include <sstream>
#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>

//#define DEBUG

namespace evf {

  namespace {
    struct flock make_flock(short type, short whence, off_t start, off_t len, pid_t pid)
    {
#ifdef __APPLE__
      return {start, len, pid, type, whence};
#else
      return {type, whence, start, len, pid};
#endif
    }
  }

  EvFDaqDirector::EvFDaqDirector(const edm::ParameterSet &pset,
				 edm::ActivityRegistry& reg) :
    testModeNoBuilderUnit_(
			   pset.getUntrackedParameter<bool> ("testModeNoBuilderUnit",
							     false)
			   ),
    base_dir_(
	      pset.getUntrackedParameter<std::string> ("baseDir", "/data")
	      ),
    bu_base_dir_(
		 pset.getUntrackedParameter<std::string> ("buBaseDir", "/data")
		 ),
    sm_base_dir_(
		 pset.getUntrackedParameter<std::string> ("smBaseDir", "/sm")
		 ),
    monitor_base_dir_(
		      pset.getUntrackedParameter<std::string> ("monBaseDir",
							       "MON")
		      ),
    directorBu_(
		pset.getUntrackedParameter<bool> ("directorIsBu", false)
		),
    run_(pset.getUntrackedParameter<unsigned int> ("runNumber",0)),
    hostname_(""),
    bu_readlock_fd_(-1),
    bu_writelock_fd_(-1),
    fu_readwritelock_fd_(-1),
    data_readwrite_fd_(-1),

    bu_w_lock_stream(0),
    bu_r_lock_stream(0),
    fu_rw_lock_stream(0),
    bu_w_monitor_stream(0),
    bu_t_monitor_stream(0),
    data_rw_stream(0),

    dirManager_(base_dir_),

    previousFileSize_(0),
    jumpLS_(0),
    jumpIndex_(0),

    bu_w_flk( make_flock( F_WRLCK, SEEK_SET, 0, 0, 0 )),
    bu_r_flk( make_flock( F_RDLCK, SEEK_SET, 0, 0, 0 )),
    bu_w_fulk( make_flock( F_UNLCK, SEEK_SET, 0, 0, 0 )),
    bu_r_fulk( make_flock( F_UNLCK, SEEK_SET, 0, 0, 0 )),
    fu_rw_flk( make_flock ( F_WRLCK, SEEK_SET, 0, 0, getpid() )),
    fu_rw_fulk( make_flock( F_UNLCK, SEEK_SET, 0, 0, getpid() )),
    data_rw_flk( make_flock ( F_WRLCK, SEEK_SET, 0, 0, getpid() )),
    data_rw_fulk( make_flock( F_UNLCK, SEEK_SET, 0, 0, getpid() ))
  {

    reg.watchPreallocate(this, &EvFDaqDirector::preallocate);
    reg.watchPreGlobalBeginRun(this, &EvFDaqDirector::preBeginRun);
    reg.watchPostGlobalEndRun(this, &EvFDaqDirector::postEndRun);
    reg.watchPreSourceEvent(this, &EvFDaqDirector::preSourceEvent);

    std::stringstream ss;
    ss << "run" << std::setfill('0') << std::setw(6) << run_;
    run_string_ = ss.str();
    run_dir_ = base_dir_+"/"+run_string_;

    //save hostname for later
    char hostname[33];
    gethostname(hostname,32);
    hostname_ = hostname;
    // check if base dir exists or create it accordingly
    int retval = mkdir(base_dir_.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (retval != 0 && errno != EEXIST) {
      throw cms::Exception("DaqDirector") << " Error checking for base dir "
    					  << base_dir_ << " mkdir error:" << strerror(errno) << "\n";
    }

    //bu_run_dir: for FU, for which the base dir is local and the BU is remote, it is expected to be there
    //for BU, it is created at this point


    if (directorBu_)
      {
	bu_run_dir_ = bu_base_dir_ + "/" + run_string_;
	std::string bulockfile = bu_run_dir_ + "/bu.lock";
	std::string fulockfile = bu_run_dir_ + "/fu.lock";

	//make or find bu run dir
	retval = mkdir(bu_run_dir_.c_str(),
		       S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	if (retval != 0 && errno != EEXIST) {
	  throw cms::Exception("DaqDirector")
	    << " Error creating bu run dir " << bu_run_dir_
	    << " mkdir error:" << strerror(errno) << "\n";
	}
	bu_run_open_dir_ = bu_run_dir_ + "/open";
	retval = mkdir(bu_run_open_dir_.c_str(),
		       S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	if (retval != 0 && errno != EEXIST) {
	  throw cms::Exception("DaqDirector") << " Error creating bu run open dir "
					      << bu_run_open_dir_ << " mkdir error:" << strerror(errno)
					      << "\n";
	}

	//make or find monitor base dir
	//@@EM make sure this is still needed

	std::stringstream ost;
	ost << bu_run_dir_ << "/" << monitor_base_dir_;
	monitor_base_dir_ = ost.str() + "_OLD";
	retval = mkdir(monitor_base_dir_.c_str(),
		       S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	if (retval != 0 && errno != EEXIST) {
	  throw cms::Exception("DaqDirector")
	    << " Error creating monitor dir " << monitor_base_dir_
	    << " mkdir error:" << strerror(errno) << "\n";
	}

	// the BU director does not need to know about the fu lock
	bu_writelock_fd_ = open(bulockfile.c_str(),
				O_WRONLY | O_CREAT | O_TRUNC, S_IRWXU);
	if (bu_writelock_fd_ == -1)
	  edm::LogWarning("EvFDaqDirector") << "problem with creating filedesc for buwritelock "
					    << strerror(errno);
	else
	  edm::LogInfo("EvFDaqDirector") << "creating filedesc for buwritelock "
					 << bu_writelock_fd_;
	bu_w_lock_stream = fdopen(bu_writelock_fd_, "w");
	if (bu_w_lock_stream == 0)
	  edm::LogWarning("EvFDaqDirector")<< "Error creating write lock stream " << strerror(errno);
	std::string filename = monitor_base_dir_ + "/bumonitor.txt";
	bu_w_monitor_stream = fopen(filename.c_str(), "w+");
	filename = monitor_base_dir_ + "/diskmonitor.txt";
	bu_t_monitor_stream = fopen(filename.c_str(), "w+");
	if (bu_t_monitor_stream == 0)
	  edm::LogWarning("EvFDaqDirector") << "Error creating bu write lock stream " << strerror(errno);

	// BU INITIALIZES LOCK FILE
	// FU LOCK FILE OPEN
	openFULockfileStream(fulockfile, true);
	tryInitializeFuLockFile();
	fflush(fu_rw_lock_stream);
	close(fu_readwritelock_fd_);
	//createOutputDirectory(); // this should act not on the bu base dir but on the output disk
      }
    else
      {
	// for FU, check if bu base dir exists

	retval = mkdir(bu_base_dir_.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	if (retval != 0 && errno != EEXIST) {
	  throw cms::Exception("DaqDirector") << " Error checking for bu base dir "
					      << base_dir_ << " mkdir error:" << strerror(errno) << "\n";
	}

	bu_run_dir_ = bu_base_dir_ + "/" + run_string_;
	std::string fulockfile = bu_run_dir_ + "/fu.lock";
	openFULockfileStream(fulockfile, false);
      }

    pthread_mutex_init(&init_lock_,NULL);

  }

//  void EvFDaqDirector::postEndRun(edm::Run const& run, edm::EventSetup const& es) {
  void EvFDaqDirector::postEndRun(edm::GlobalContext const& globalContext) {
    close(bu_readlock_fd_);
    close(bu_writelock_fd_);
    if (directorBu_) {
      std::string filename = bu_base_dir_ + "/bu.lock";
      removeFile(filename);
    }
  }

  void EvFDaqDirector::preallocate(edm::service::SystemBounds const& bounds) {

    for (unsigned int i=0;i<bounds.maxNumberOfStreams();i++){
      streamFileTracker_.push_back(-1);
    }
    nThreads_=bounds.maxNumberOfStreams();
    nStreams_=bounds.maxNumberOfThreads();
  }

  void EvFDaqDirector::preBeginRun(edm::GlobalContext const& globalContext) {

//    assert(run_ == id.run());

    // check if run dir exists or make it.
    umask(0);
    int retval = mkdir(run_dir_.c_str(),
		       S_IRWXU | S_IRWXG | S_IROTH | S_IRWXO | S_IXOTH);
    if (retval != 0 && errno != EEXIST) {
      throw cms::Exception("DaqDirector") << " Error creating run dir "
					  << run_dir_ << " mkdir error:" << strerror(errno) << "\n";
    }

    // check if the requested run is the latest one - issue a warning if it isn't
    if (dirManager_.findHighestRunDir() != run_dir_) {
      edm::LogWarning("EvFDaqDirector") << "DaqDirector Warning checking run dir "
					<< run_dir_ << " this is not the highest run "
					<< dirManager_.findHighestRunDir();
    }
  }

  inline void EvFDaqDirector::preSourceEvent(edm::StreamID const& streamID) {
    streamFileTracker_[streamID]=currentFileIndex_;
  }

  bool EvFDaqDirector::createOutputDirectory() {
    int retval = mkdir(sm_base_dir_.c_str(),
		       S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (retval != 0 && errno != EEXIST) {
      throw cms::Exception("DaqDirector") << " Error creating output dir "
					  << sm_base_dir_ << " mkdir error:" << strerror(errno) << "\n";
      return false;
    }
    std::string mergedRunDir = sm_base_dir_ + "/" + run_string_;
    std::string mergedDataDir = mergedRunDir + "/data";
    std::string mergedMonDir = mergedRunDir + "/mon";
    retval = mkdir(mergedRunDir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (retval != 0 && errno != EEXIST) {
      throw cms::Exception("DaqDirector")
	<< " Error creating merged Run dir " << mergedDataDir
	<< " mkdir error:" << strerror(errno) << "\n";
      return false;
    }
    retval
      = mkdir(mergedDataDir.c_str(),
	      S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (retval != 0 && errno != EEXIST) {
      throw cms::Exception("DaqDirector")
	<< " Error creating merged data dir " << mergedDataDir
	<< " mkdir error:" << strerror(errno) << "\n";
      return false;
    }
    retval = mkdir(mergedMonDir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (retval != 0 && errno != EEXIST) {
      throw cms::Exception("DaqDirector")
	<< " Error creating merged mon dir " << mergedMonDir
	<< " mkdir error:" << strerror(errno) << "\n";
      return false;
    }
    return true;
  }

  std::string EvFDaqDirector::getRawFilePath(const unsigned int ls, const unsigned int index) const {
    return bu_run_dir_ + "/" + fffnaming::inputRawFileName(run_,ls,index);
  }

  std::string EvFDaqDirector::getOpenRawFilePath(const unsigned int ls, const unsigned int index) const {
    return bu_run_dir_ + "/open/" + fffnaming::inputRawFileName(run_,ls,index);
  }

  std::string EvFDaqDirector::getOpenDatFilePath(const unsigned int ls, std::string const& stream) const {
    return run_dir_ + "/open/" + fffnaming::streamerDataFileNameWithPid(run_,ls,stream);
  }

  std::string EvFDaqDirector::getOutputJsonFilePath(const unsigned int ls, std::string const& stream) const {
    return run_dir_ + "/" + fffnaming::streamerJsonFileNameWithPid(run_,ls,stream);
  }

  std::string EvFDaqDirector::getMergedDatFilePath(const unsigned int ls, std::string const& stream) const {
    return run_dir_ + "/" + fffnaming::streamerDataFileNameWithInstance(run_,ls,stream,hostname_);
  }

  std::string EvFDaqDirector::getInitFilePath(std::string const& stream) const {
    return run_dir_ + "/" + fffnaming::initFileNameWithPid(run_,0,stream);
  }

  std::string EvFDaqDirector::getEoLSFilePathOnBU(const unsigned int ls) const {
    return bu_run_dir_ + "/" + fffnaming::eolsFileName(run_,ls);
  }

  std::string EvFDaqDirector::getEoLSFilePathOnFU(const unsigned int ls) const {
    return run_dir_ + "/" + fffnaming::eolsFileName(run_,ls);
  }

  std::string EvFDaqDirector::getEoRFilePath() const {
    return bu_run_dir_ + "/" + fffnaming::eorFileName(run_);
  }


  std::string EvFDaqDirector::getEoRFilePathOnFU() const {
    return run_dir_ + "/" + fffnaming::eorFileName(run_);
  }


  std::string EvFDaqDirector::getPathForFU() const {
    return sm_base_dir_;
  }

  void EvFDaqDirector::removeFile(std::string filename) {
    int retval = remove(filename.c_str());
    if (retval != 0)
      edm::LogError("EvFDaqDirector") << "Could not remove used file " << filename << " error "
		                      << strerror(errno);
  }

  void EvFDaqDirector::removeFile(unsigned int ls, unsigned int index) {
    removeFile(getRawFilePath(ls,index));
  }

  void EvFDaqDirector::updateBuLock(unsigned int ls) {
    int check = 0;
    fcntl(bu_writelock_fd_, F_SETLKW, &bu_w_flk);
    if (bu_w_lock_stream != 0) {
      check = fseek(bu_w_lock_stream, 0, SEEK_SET);
      if (check == 0)
	fprintf(bu_w_lock_stream, "%u", ls);
      else
	edm::LogError("EvFDaqDirector")
	  << "seek on bu write lock for updating failed with error "
	  << strerror(errno);
    } else
      edm::LogError("EvFDaqDirector") << "bu write lock stream is invalid " << strerror(errno);

    fcntl(bu_writelock_fd_, F_SETLKW, &bu_w_fulk);
  }

  EvFDaqDirector::FileStatus EvFDaqDirector::updateFuLock(unsigned int& ls, std::string& nextFile, uint32_t& fsize) {
    EvFDaqDirector::FileStatus fileStatus = noFile;

    int retval = -1;
    while (retval==-1) {
      retval = fcntl(fu_readwritelock_fd_, F_SETLK, &fu_rw_flk);
      if (retval==-1) usleep(50000);
    }
    if(retval!=0) return fileStatus;

#ifdef DEBUG
    timeval ts_lockend;
    gettimeofday(&ts_lockend,0);
#endif

    // if the stream is readable
    if (fu_rw_lock_stream != 0) {
      unsigned int readLs, readIndex, jumpLs, jumpIndex;
      int check = 0;
      // rewind the stream
      check = fseek(fu_rw_lock_stream, 0, SEEK_SET);
      // if rewinded ok
      if (check == 0) {
	// read its' values
	if (testModeNoBuilderUnit_)
	  fscanf(fu_rw_lock_stream, "%u %u %u %u", &readLs, &readIndex,
		 &jumpLs, &jumpIndex);
	else
	  fscanf(fu_rw_lock_stream, "%u %u", &readLs, &readIndex);

	// try to bump
	bool bumpedOk = bumpFile(readLs, readIndex, nextFile, fsize);
	ls = readLs;
	// there is a new file to grab or lumisection ended
	if (bumpedOk) {
	  // rewind and clear
	  check = fseek(fu_rw_lock_stream, 0, SEEK_SET);
	  if (check == 0) {
	    ftruncate(fu_readwritelock_fd_, 0);
	    fflush(fu_rw_lock_stream); //this should not be needed ???
	  } else
	      edm::LogError("EvFDaqDirector") << "seek on fu read/write lock for updating failed with error "
	                                      << strerror(errno);
	  // write new data
	  check = fseek(fu_rw_lock_stream, 0, SEEK_SET);
	  if (check == 0) {
	    // write next index in the file, which is the file the next process should take
	    if (testModeNoBuilderUnit_) {
	      fprintf(fu_rw_lock_stream, "%u %u %u %u", readLs,
		      readIndex + 1, readLs + 2, readIndex + 1);
	      jumpLS_ = readLs + 2;
	      jumpIndex_ = readIndex;
	    } else {
	      fprintf(fu_rw_lock_stream, "%u %u", readLs,
		      readIndex + 1);
	    }
	    fflush(fu_rw_lock_stream);
	    fsync(fu_readwritelock_fd_);

	    fileStatus = newFile;

	    if (testModeNoBuilderUnit_)
	      edm::LogInfo("EvFDaqDirector") << "Written to file: " << readLs << ":"
			                     << readIndex + 1 << " --> " << readLs + 2
			                     << ":" << readIndex + 1;
	    else
	      edm::LogInfo("EvFDaqDirector") << "Written to file: " << readLs << ":"
			                     << readIndex + 1;

	  } else
	      edm::LogError("EvFDaqDirector") << "seek on fu read/write lock for updating failed with error "
	                                      << strerror(errno);
	}
      } else
	edm::LogError("EvFDaqDirector") << "seek on fu read/write lock for reading failed with error "
					<< strerror(errno);
    } else
      edm::LogError("EvFDaqDirector") << "fu read/write lock stream is invalid " << strerror(errno);

#ifdef DEBUG
    timeval ts_preunlock;
    gettimeofday(&ts_preunlock,0);
    int locked_period_int = ts_preunlock.tv_sec - ts_lockend.tv_sec;
    double locked_period=locked_period_int+double(ts_preunlock.tv_usec - ts_lockend.tv_usec)/1000000;
#endif

    //release lock at this point
    int retvalu=-1;
    retvalu=fcntl(fu_readwritelock_fd_, F_SETLKW, &fu_rw_fulk);
    if (retvalu==-1) edm::LogError("EvFDaqDirector") << "Error unlocking the fu.lock " << strerror(errno);

#ifdef DEBUG
    edm::LogInfo("EvFDaqDirector") << "Waited during lock:" << locked_period;
#endif

    if ( fileStatus == noFile ) {
      struct stat buf;
      edm::LogInfo("EvFDaqDirector") << " looking for EoR file: " << getEoRFilePath().c_str();
      if ( stat(getEoRFilePath().c_str(), &buf) == 0 )
        fileStatus = runEnded;
    }
    return fileStatus;
  }

  int EvFDaqDirector::readBuLock() {
    int retval = -1;
    // check if lock file has disappeared and if so whether directory is empty (signal end of run)
    if (!bulock() && dirManager_.checkDirEmpty(bu_base_dir_))
      return retval;
    if (fcntl(bu_readlock_fd_, F_SETLKW, &bu_r_flk) != 0)
      retval = 0;
    if (bu_r_lock_stream) {
      unsigned int readval;
      int check = 0;
      unsigned int *p = &readval;
      check = fseek(bu_r_lock_stream, 0, SEEK_SET);
      if (check == 0) {
	fscanf(bu_r_lock_stream, "%u", p);
	retval = readval;
      }
    } else {
      edm::LogError("EvFDaqDirector") << "error reading bu lock file " << strerror(errno);
      retval = -1;
    }
    fcntl(bu_readlock_fd_, F_SETLKW, &bu_r_fulk);
     edm::LogInfo("EvFDaqDirector") << "readbulock returning " << retval;
    return retval;
  }



  bool EvFDaqDirector::bulock() {
    struct stat buf;
    std::string lockfile = bu_base_dir_;
    lockfile += "/bu.lock";
    bool retval = (stat(lockfile.c_str(), &buf) == 0);
    if (!retval) {
      close(bu_readlock_fd_);
      close(bu_writelock_fd_);
    }
    edm::LogInfo("EvFDaqDirector") << "stat of lockfile returned " << retval;
    return retval;
  }

  bool EvFDaqDirector::fulock() {
    struct stat buf;
    std::string lockfile = bu_base_dir_;
    lockfile += "/fu.lock";
    bool retval = (stat(lockfile.c_str(), &buf) == 0);
    if (!retval) {
      close(fu_readwritelock_fd_);
      close(fu_readwritelock_fd_); // why the second close ?
    }
    edm::LogInfo("EvFDaqDirector") << "stat of lockfile returned " << retval;
    return retval;
  }

  void EvFDaqDirector::writeLsStatisticsBU(unsigned int ls, unsigned int events,
					   unsigned long long totsize, long long lsusec) {
    if (bu_w_monitor_stream != 0) {
      int check = fseek(bu_w_monitor_stream, 0, SEEK_SET);
      if (check == 0) {
	fprintf(bu_w_monitor_stream, "%u %u %llu %f %f %012lld", ls,
		events, totsize,
		double(totsize) / double(events) / 1000000.,
		double(totsize) / double(lsusec), lsusec);
	fflush(bu_w_monitor_stream);
      } else
	edm::LogError("EvFDaqDirector") << "seek on bu write monitor for updating failed with error "
	                                << strerror(errno);
    } else
      edm::LogError("EvFDaqDirector") << "bu write monitor stream is invalid " << strerror(errno);

  }
  void EvFDaqDirector::writeDiskAndThrottleStat(double fraction, int highWater,
						int lowWater) {
    if (bu_t_monitor_stream != 0) {
      int check = fseek(bu_t_monitor_stream, 0, SEEK_SET);
      if (check == 0)
	fprintf(bu_t_monitor_stream, "%f %d %d", fraction, highWater,
		lowWater);
      else
	edm::LogError("EvFDaqDirector") << "seek on disk write monitor for updating failed with error "
	                                << strerror(errno);
    } else
      edm::LogError("EvFDaqDirector") << "disk write monitor stream is invalid " << strerror(errno);
  }

  bool EvFDaqDirector::bumpFile(unsigned int& ls, unsigned int& index, std::string& nextFile, uint32_t& fsize) {

    if (previousFileSize_ != 0) {
      if (!fms_) {
        try {
          fms_ = (FastMonitoringService *) (edm::Service<evf::MicroStateService>().operator->());
        } catch (...) {
	        edm::LogError("EvFDaqDirector") <<" FastMonitoringService not found ";
        }
      }
      if (fms_) fms_->accumulateFileSize(ls, previousFileSize_);
      previousFileSize_ = 0;
    }

    struct stat buf;
    std::stringstream ss;
    unsigned int nextIndex = index;
    nextIndex++;

    // 1. Check suggested file
    nextFile = getRawFilePath(ls,index);
    if (stat(nextFile.c_str(), &buf) == 0) {
     
      previousFileSize_ = buf.st_size;
      fsize = buf.st_size;
      return true;
    }
    // 2. No file -> lumi ended? (and how many?)
    else {
      bool eolFound = (stat(getEoLSFilePathOnBU(ls).c_str(), &buf) == 0);
      unsigned int startingLumi = ls;
      while (eolFound) {
        // recheck that no raw file appeared in the meantime
        if (stat(nextFile.c_str(), &buf) == 0) {
          previousFileSize_ = buf.st_size;
          fsize = buf.st_size;
          return true;
        }
	// this lumi ended, check for files
	++ls;
	nextFile = getRawFilePath(ls,0);
	if (stat(nextFile.c_str(), &buf) == 0) {
	  // a new file was found at new lumisection, index 0
	  index = 0;
	  previousFileSize_ = buf.st_size;
          fsize = buf.st_size;

	  if (testModeNoBuilderUnit_) {
	    // rename ended lumi to + 2
            std::string sourceEol = getEoLSFilePathOnBU(startingLumi);

	    std::string destEol = getEoLSFilePathOnBU(startingLumi+2);

	    std::string cpCmd = "cp " + sourceEol + " " + destEol;
	    edm::LogInfo("EvFDaqDirector") << " testmode: Running copy cmd = " << cpCmd;
	    int rc = system(cpCmd.c_str());
	    if (rc != 0) {
	      edm::LogError("EvFDaqDirector") << " testmode: COPY EOL FAILED!!!!!: " << cpCmd;
	    }
	  }

	  return true;
	}
	eolFound = (stat(getEoLSFilePathOnBU(ls).c_str(), &buf) == 0);
      }
    }
    // no new file found
    return false;
  }

  std::string EvFDaqDirector::findHighestRunDirStem() {
    boost::filesystem::path highestRunDirPath (findHighestRunDir());
    return highestRunDirPath.filename().string();
  }

  void EvFDaqDirector::tryInitializeFuLockFile() {
    if (fu_rw_lock_stream == 0)
      edm::LogError("EvFDaqDirector") << "Error creating fu read/write lock stream "
				      << strerror(errno);
    else {
      edm::LogInfo("EvFDaqDirector") << "Initializing FU LOCK FILE";
      unsigned int readLs = 1, readIndex = 0, jumpLs = 3, jumpIndex = 0;
      if (testModeNoBuilderUnit_)
	fprintf(fu_rw_lock_stream, "%u %u %u %u", readLs, readIndex,
		jumpLs, jumpIndex);
      else
	fprintf(fu_rw_lock_stream, "%u %u", readLs, readIndex);
    }
  }

  void EvFDaqDirector::openFULockfileStream(std::string& fulockfile, bool create) {
    if (create) {
      fu_readwritelock_fd_ = open(fulockfile.c_str(), O_RDWR | O_CREAT,
				  S_IRWXU | S_IRWXG | S_IROTH | S_IRWXO | S_IXOTH);
    } else {
      fu_readwritelock_fd_ = open(fulockfile.c_str(), O_RDWR, S_IRWXU);
    }
    if (fu_readwritelock_fd_ == -1)
      edm::LogError("EvFDaqDirector") << "problem with creating filedesc for fuwritelock "
		<< strerror(errno);
    else
      edm::LogInfo("EvFDaqDirector") << "creating filedesc for fureadwritelock "
		<< fu_readwritelock_fd_;

    fu_rw_lock_stream = fdopen(fu_readwritelock_fd_, "r+");
  }

  //create if does not exist then lock the merge destination file
  FILE *EvFDaqDirector::maybeCreateAndLockFileHeadForStream(unsigned int ls, std::string &stream) {
    data_rw_stream = fopen(getMergedDatFilePath(ls,stream).c_str(), "a"); //open stream for appending
    data_readwrite_fd_ = fileno(data_rw_stream);
    if (data_readwrite_fd_ == -1)
      edm::LogError("EvFDaqDirector") << "problem with creating filedesc for datamerge "
		<< strerror(errno);
    else
      edm::LogInfo("EvFDaqDirector") << "creating filedesc for datamerge "
		<< data_readwrite_fd_;
    fcntl(data_readwrite_fd_, F_SETLKW, &data_rw_flk);

    return data_rw_stream;
  }

  void EvFDaqDirector::unlockAndCloseMergeStream() {
    fflush(data_rw_stream);
    fcntl(data_readwrite_fd_, F_SETLKW, &data_rw_fulk);
    fclose(data_rw_stream);
  }

  void EvFDaqDirector::lockInitLock() {
    pthread_mutex_lock(&init_lock_);
  }

  void EvFDaqDirector::unlockInitLock() {
    pthread_mutex_unlock(&init_lock_);
  }

}
