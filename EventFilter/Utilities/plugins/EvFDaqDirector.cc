#include "EvFDaqDirector.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FastMonitoringService.h"
#include <iostream>
#include <sstream>
#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>

using std::stringstream;

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
    reg.watchPreBeginRun(this, &EvFDaqDirector::preBeginRun);
    reg.watchPostEndRun(this, &EvFDaqDirector::postEndRun);

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

	stringstream ost;
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
	createOutputDirectory(); // this should act not on the bu base dir but on the output disk
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


  }

  void EvFDaqDirector::postEndRun(edm::Run const& run, edm::EventSetup const& es) {
    close(bu_readlock_fd_);
    close(bu_writelock_fd_);
    if (directorBu_) {
      std::string filename = bu_base_dir_ + "/bu.lock";
      removeFile(filename);
    }
  }

  void EvFDaqDirector::preBeginRun(edm::RunID const& id, edm::Timestamp const& ts) {

    assert(run_ == id.run());

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
    return bu_run_dir_ + "/" + inputFileNameStem(ls, index) + ".raw";
  }

  std::string EvFDaqDirector::getOpenRawFilePath(const unsigned int ls, const unsigned int index) const {
    return bu_run_dir_ + "/open/" + inputFileNameStem(ls, index) + ".raw";
  }

  std::string EvFDaqDirector::getOpenDatFilePath(const unsigned int ls, std::string const& stream) const {
    return run_dir_ + "/open/" + outputFileNameStem(ls,stream) + ".dat";
  }

  std::string EvFDaqDirector::getOutputJsonFilePath(const unsigned int ls, std::string const& stream) const {
    return run_dir_ + "/" + outputFileNameStem(ls,stream) + ".jsn";
  }

  std::string EvFDaqDirector::getMergedDatFilePath(const unsigned int ls, std::string const& stream) const {
    return run_dir_ + "/" + mergedFileNameStem(ls,stream) + ".dat";
  }

  std::string EvFDaqDirector::getInitFilePath(std::string const& stream) const {
    return run_dir_ + "/" + initFileName(stream);
  }

  std::string EvFDaqDirector::getEoLSFilePathOnBU(const unsigned int ls) const {
    return bu_run_dir_ + "/" + eolsFileName(ls);
  }

  std::string EvFDaqDirector::getEoLSFilePathOnFU(const unsigned int ls) const {
    return run_dir_ + "/" + eolsFileName(ls);
  }

  std::string EvFDaqDirector::getEoRFilePath() const {
    return bu_run_dir_ + "/" + eorFileName();
  }

  std::string EvFDaqDirector::getPathForFU() const {
    return sm_base_dir_;
  }

  void EvFDaqDirector::removeFile(std::string filename) {
    int retval = remove(filename.c_str());
    if (retval != 0)
      std::cout << "Could not remove used file " << filename << " error "
		<< strerror(errno) << "\n";
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

  bool EvFDaqDirector::updateFuLock(unsigned int& ls, std::string& nextFile,
				    bool& eorSeen) {
    // close and reopen the stream / fd
    close(fu_readwritelock_fd_);
    std::string fulockfile = bu_run_dir_ + "/fu.lock";
    fu_readwritelock_fd_ = open(fulockfile.c_str(), O_RDWR | O_NONBLOCK, S_IRWXU);
    if (fu_readwritelock_fd_ == -1)
      edm::LogError("EvFDaqDirector") << "problem with creating filedesc for fuwritelock "
		<< strerror(errno);
    else
      edm::LogInfo("EvFDaqDirector") << "created filedesc for fureadwritelock "
		<< fu_readwritelock_fd_;
    fu_rw_lock_stream = fdopen(fu_readwritelock_fd_, "r+");
    edm::LogInfo("EvFDaqDirector") << "Reopened the fw FD & STREAM";

    // obtain lock on the fulock file - this call will block if the lock is held by another process
    int retval = fcntl(fu_readwritelock_fd_, F_SETLKW, &fu_rw_flk);
    //if locking fails just return here 
    if(retval!=0) return false;

    eorSeen = false;
    struct stat buf;
    eorSeen = (stat(getEoRFilePath().c_str(), &buf) == 0);
    bool valid = false;

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
	bool bumpedOk = bumpFile(readLs, readIndex, nextFile);

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

	    valid = true;

	    if (testModeNoBuilderUnit_)
	      edm::LogInfo("EvFDaqDirector")<< "Written to file: " << readLs << ":"
			<< readIndex + 1 << " --> " << readLs + 2
			<< ":" << readIndex + 1;
	    else
	      edm::LogInfo("EvFDaqDirector")<< "Written to file: " << readLs << ":"
			<< readIndex + 1;

	  } else
	      edm::LogError("EvFDaqDirector") << "seek on fu read/write lock for updating failed with error "
	      << strerror(errno) << std::endl;
	}
      } else
	edm::LogError("EvFDaqDirector") << "seek on fu read/write lock for reading failed with error "
					<< strerror(errno) << std::endl;
    } else
      edm::LogError("EvFDaqDirector") << "fu read/write lock stream is invalid " << strerror(errno);
    //release flock at this point
    fcntl(fu_readwritelock_fd_, F_SETLKW, &fu_rw_fulk);

    return valid;
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
    std::cout << "readbulock returning " << retval << std::endl;
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
    std::cout << "stat of lockfile returned " << retval << std::endl;
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
    std::cout << "stat of lockfile returned " << retval << std::endl;
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
	std::cout
	  << "seek on bu write monitor for updating failed with error "
	  << strerror(errno) << std::endl;
    } else
      std::cout << "bu write monitor stream is invalid " << strerror(errno)
		<< std::endl;

  }
  void EvFDaqDirector::writeDiskAndThrottleStat(double fraction, int highWater,
						int lowWater) {
    if (bu_t_monitor_stream != 0) {
      int check = fseek(bu_t_monitor_stream, 0, SEEK_SET);
      if (check == 0)
	fprintf(bu_t_monitor_stream, "%f %d %d", fraction, highWater,
		lowWater);
      else
	std::cout
	  << "seek on disk write monitor for updating failed with error "
	  << strerror(errno) << std::endl;
    } else
      std::cout << "disk write monitor stream is invalid " << strerror(errno)
		<< std::endl;

  }

  bool EvFDaqDirector::bumpFile(unsigned int& ls, unsigned int& index, std::string& nextFile) {

    if (previousFileSize_ != 0) {
      FastMonitoringService *mss = (FastMonitoringService *) (edm::Service<
							      evf::MicroStateService>().operator->());
      mss->accummulateFileSize(previousFileSize_);
      previousFileSize_ = 0;
    }

    struct stat buf;
    std::stringstream ss;
    unsigned int nextIndex = index;
    nextIndex++;

    // 1. Check suggested file
    nextFile = getRawFilePath(ls,index);
    bool found = (stat(nextFile.c_str(), &buf) == 0);
    // if found
    if (found) {
      //grabbedFileSize = buf.st_size;
      previousFileSize_ = buf.st_size;
      return true;
    }
    // 2. No file -> lumi ended? (and how many?)
    else {
      bool eolFound = (stat(getEoLSFilePathOnBU(ls).c_str(), &buf) == 0);
      unsigned int startingLumi = ls;
      while (eolFound) {
	// this lumi ended, check for files
	++ls;
	nextFile = getRawFilePath(ls,0);
	found = (stat(nextFile.c_str(), &buf) == 0);
	// update highest ls even if there is no file
	// input source can now end its' LS when an EoL jsn file is seen
	if (found) {
	  // a new file was found at new lumisection, index 0
	  index = 0;
	  //grabbedFileSize = buf.st_size;
	  previousFileSize_ = buf.st_size;
	  
	  if (testModeNoBuilderUnit_) {
	    // rename ended lumi to + 2
	    string sourceEol = getEoLSFilePathOnBU(startingLumi);
	    
	    string destEol = getEoLSFilePathOnBU(startingLumi+2);
	    
	    string cpCmd = "cp " + sourceEol + " " + destEol;
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

  std::string EvFDaqDirector::inputFileNameStem(const unsigned int ls, const unsigned int index) const {
    std::stringstream ss;
    ss << run_string_
       << "_ls" << std::setfill('0') << std::setw(4) << ls
       << "_index" << std::setfill('0') << std::setw(6) << index;
    return ss.str();
  }

  std::string EvFDaqDirector::outputFileNameStem(const unsigned int ls, std::string const& stream) const {
    std::stringstream ss;
    ss << run_string_
       << "_ls" << std::setfill('0') << std::setw(4) << ls
       << "_" << stream
       << "_pid" << std::setfill('0') << std::setw(5) << getpid();
    return ss.str();
  }

  std::string EvFDaqDirector::mergedFileNameStem(const unsigned int ls, std::string const& stream) const {
    std::stringstream ss;
    ss << run_string_
       << "_ls" << std::setfill('0') << std::setw(4) << ls
       << "_" << stream
       << "_" << hostname_;
    return ss.str();
  }

  std::string EvFDaqDirector::initFileName(std::string const& stream) const {
    std::stringstream ss;
    ss << run_string_
       << "_" << stream
       << "_pid" << std::setfill('0') << std::setw(5) << getpid()
       << ".ini";
    return ss.str();
  }

  std::string EvFDaqDirector::eolsFileName(const unsigned int ls) const {
    std::stringstream ss;
    ss << "EoLS_" << std::setfill('0') << std::setw(4) << ls << ".jsn";
    return ss.str();
  }

  std::string EvFDaqDirector::eorFileName() const {
    std::stringstream ss;
    ss << "EoR_" << std::setfill('0') << std::setw(6) << run_ << ".jsn";
    return ss.str();
  }
}
