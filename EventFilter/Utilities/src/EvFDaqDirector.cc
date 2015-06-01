#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/SystemBounds.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "EventFilter/Utilities/interface/EvFDaqDirector.h"
#include "EventFilter/Utilities/interface/FastMonitoringService.h"
#include "EventFilter/Utilities/plugins/FedRawDataInputSource.h"
#include "EventFilter/Utilities/interface/DataPointDefinition.h"
#include "EventFilter/Utilities/interface/DataPoint.h"

#include <iostream>
#include <sstream>
#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>
#include <sys/file.h>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem/fstream.hpp>

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
    directorBu_(
		pset.getUntrackedParameter<bool> ("directorIsBu", false)
		),
    run_(pset.getUntrackedParameter<unsigned int> ("runNumber",0)),
    outputAdler32Recheck_(pset.getUntrackedParameter<bool>("outputAdler32Recheck",false)),
    requireTSPSet_(pset.getUntrackedParameter<bool>("requireTransfersPSet",false)),
    selectedTransferMode_(pset.getUntrackedParameter<std::string>("selectedTransferMode","")),
    hltSourceDirectory_(pset.getUntrackedParameter<std::string>("hltSourceDirectory","")),
    fuLockPollInterval_(pset.getUntrackedParameter<unsigned int>("fuLockPollInterval",2000)),
    hostname_(""),
    bu_readlock_fd_(-1),
    bu_writelock_fd_(-1),
    fu_readwritelock_fd_(-1),
    data_readwrite_fd_(-1),
    fulocal_rwlock_fd_(-1),
    fulocal_rwlock_fd2_(-1),

    bu_w_lock_stream(0),
    bu_r_lock_stream(0),
    fu_rw_lock_stream(0),
    //bu_w_monitor_stream(0),
    //bu_t_monitor_stream(0),
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
    //fulocal_rw_flk( make_flock( F_WRLCK, SEEK_SET, 0, 0, getpid() )),
    //fulocal_rw_fulk( make_flock( F_UNLCK, SEEK_SET, 0, 0, getpid() )),
    //fulocal_rw_flk2( make_flock( F_WRLCK, SEEK_SET, 0, 0, getpid() )),
    //fulocal_rw_fulk2( make_flock( F_UNLCK, SEEK_SET, 0, 0, getpid() ))
  {

    reg.watchPreallocate(this, &EvFDaqDirector::preallocate);
    reg.watchPreGlobalBeginRun(this, &EvFDaqDirector::preBeginRun);
    reg.watchPostGlobalEndRun(this, &EvFDaqDirector::postEndRun);
    reg.watchPreSourceEvent(this, &EvFDaqDirector::preSourceEvent);
    reg.watchPreGlobalEndLumi(this,&EvFDaqDirector::preGlobalEndLumi);

    std::stringstream ss;
    ss << "run" << std::setfill('0') << std::setw(6) << run_;
    run_string_ = ss.str();
    run_dir_ = base_dir_+"/"+run_string_;

    //save hostname for later
    char hostname[33];
    gethostname(hostname,32);
    hostname_ = hostname;

    char * fuLockPollIntervalPtr = getenv("FFF_LOCKPOLLINTERVAL");
    if (fuLockPollIntervalPtr) {
      try {
        fuLockPollInterval_=boost::lexical_cast<unsigned int>(std::string(fuLockPollIntervalPtr));
        edm::LogInfo("Setting fu lock poll interval by environment string: ") << fuLockPollInterval_ << " us";
      }
      catch (...) {edm::LogWarning("Unable to parse environment string: ") << std::string(fuLockPollIntervalPtr);}
    }

    // check if base dir exists or create it accordingly
    int retval = mkdir(base_dir_.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (retval != 0 && errno != EEXIST) {
      throw cms::Exception("DaqDirector") << " Error checking for base dir -: "
    					  << base_dir_ << " mkdir error:" << strerror(errno);
    }

    //create run dir in base dir
    umask(0);
    retval = mkdir(run_dir_.c_str(),
		       S_IRWXU | S_IRWXG | S_IROTH | S_IRWXO | S_IXOTH);
    if (retval != 0 && errno != EEXIST) {
      throw cms::Exception("DaqDirector") << " Error creating run dir -: "
					  << run_dir_ << " mkdir error:" << strerror(errno);
    }

    //create fu-local.lock in run open dir
    if (!directorBu_) {

      createRunOpendirMaybe();
      std::string fulocal_lock_ = getRunOpenDirPath() +"/fu-local.lock";
      fulocal_rwlock_fd_ = open(fulocal_lock_.c_str(), O_RDWR | O_CREAT, S_IRWXU | S_IWGRP | S_IRGRP | S_IWOTH | S_IROTH);//O_RDWR?
      if (fulocal_rwlock_fd_==-1)
        throw cms::Exception("DaqDirector") << " Error creating/opening a local lock file -: " << fulocal_lock_.c_str() << " : " << strerror(errno);
      chmod(fulocal_lock_.c_str(),0777);
      fsync(fulocal_rwlock_fd_);
      //open second fd for another input source thread
      fulocal_rwlock_fd2_ = open(fulocal_lock_.c_str(), O_RDWR, S_IRWXU | S_IWGRP | S_IRGRP | S_IWOTH | S_IROTH);//O_RDWR?
      if (fulocal_rwlock_fd2_==-1)
        throw cms::Exception("DaqDirector") << " Error opening a local lock file -: " << fulocal_lock_.c_str() << " : " << strerror(errno);
    }

    //bu_run_dir: for FU, for which the base dir is local and the BU is remote, it is expected to be there
    //for BU, it is created at this point
    if (directorBu_)
      {
	bu_run_dir_ = base_dir_ + "/" + run_string_;
	std::string bulockfile = bu_run_dir_ + "/bu.lock";
	std::string fulockfile = bu_run_dir_ + "/fu.lock";

	//make or find bu run dir
	retval = mkdir(bu_run_dir_.c_str(),
		       S_IRWXU | S_IRWXG | S_IRWXO);
	if (retval != 0 && errno != EEXIST) {
	  throw cms::Exception("DaqDirector")
	    << " Error creating bu run dir -: " << bu_run_dir_
	    << " mkdir error:" << strerror(errno) << "\n";
	}
	bu_run_open_dir_ = bu_run_dir_ + "/open";
	retval = mkdir(bu_run_open_dir_.c_str(),
		       S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	if (retval != 0 && errno != EEXIST) {
	  throw cms::Exception("DaqDirector") << " Error creating bu run open dir -: "
					      << bu_run_open_dir_ << " mkdir error:" << strerror(errno)
					      << "\n";
	}

	// the BU director does not need to know about the fu lock
	bu_writelock_fd_ = open(bulockfile.c_str(),
				O_WRONLY | O_CREAT | O_TRUNC, S_IRWXU);
	if (bu_writelock_fd_ == -1)
	  edm::LogWarning("EvFDaqDirector") << "problem with creating filedesc for buwritelock -: "
					    << strerror(errno);
	else
	  edm::LogInfo("EvFDaqDirector") << "creating filedesc for buwritelock -: "
					 << bu_writelock_fd_;
	bu_w_lock_stream = fdopen(bu_writelock_fd_, "w");
	if (bu_w_lock_stream == 0)
	  edm::LogWarning("EvFDaqDirector")<< "Error creating write lock stream -: " << strerror(errno);

	// BU INITIALIZES LOCK FILE
	// FU LOCK FILE OPEN
	openFULockfileStream(fulockfile, true);
	tryInitializeFuLockFile();
	fflush(fu_rw_lock_stream);
	close(fu_readwritelock_fd_);

        if (hltSourceDirectory_.size())
	{
	  struct stat buf;
	  if (stat(hltSourceDirectory_.c_str(),&buf)==0) {
	    std::string hltdir=bu_run_dir_+"/hlt";
	    std::string tmphltdir=bu_run_open_dir_+"/hlt";
	    retval = mkdir(tmphltdir.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

            boost::filesystem::copy_file(hltSourceDirectory_+"/HltConfig.py",tmphltdir+"/HltConfig.py");
            try {
              boost::filesystem::copy_file(hltSourceDirectory_+"/CMSSW_VERSION",tmphltdir+"/CMSSW_VERSION");
              boost::filesystem::copy_file(hltSourceDirectory_+"/SCRAM_ARCH",tmphltdir+"/SCRAM_ARCH");
            } catch (...) {}

            boost::filesystem::copy_file(hltSourceDirectory_+"/fffParameters.jsn",tmphltdir+"/fffParameters.jsn");

            boost::filesystem::rename(tmphltdir,hltdir);
	  }
          else
	    throw cms::Exception("DaqDirector") << " Error looking for HLT configuration -: " << hltSourceDirectory_;
	}
        //else{}//no configuration specified
      }
    else
    {
      // for FU, check if bu base dir exists

	retval = mkdir(bu_base_dir_.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	if (retval != 0 && errno != EEXIST) {
	  throw cms::Exception("DaqDirector") << " Error checking for bu base dir -: "
					      << bu_base_dir_ << " mkdir error:" << strerror(errno) << "\n";
	}

	bu_run_dir_ = bu_base_dir_ + "/" + run_string_;
	std::string fulockfile = bu_run_dir_ + "/fu.lock";
	openFULockfileStream(fulockfile, false);
      }

    pthread_mutex_init(&init_lock_,NULL);

    stopFilePath_ = run_dir_+"/CMSSW_STOP";
  }

  EvFDaqDirector::~EvFDaqDirector()
  {
    if (fulocal_rwlock_fd_!=-1) {
      unlockFULocal();
      close(fulocal_rwlock_fd_);
    }

    if (fulocal_rwlock_fd2_!=-1) {
      unlockFULocal2();
      close(fulocal_rwlock_fd2_);
    }

  }

  void EvFDaqDirector::postEndRun(edm::GlobalContext const& globalContext) {
    close(bu_readlock_fd_);
    close(bu_writelock_fd_);
    if (directorBu_) {
      std::string filename = bu_run_dir_ + "/bu.lock";
      removeFile(filename);
    }
  }

  void EvFDaqDirector::preallocate(edm::service::SystemBounds const& bounds) {

    for (unsigned int i=0;i<bounds.maxNumberOfStreams();i++){
      streamFileTracker_.push_back(-1);
    }
    nThreads_=bounds.maxNumberOfStreams();
    nStreams_=bounds.maxNumberOfThreads();

    checkTransferSystemPSet();
  }

  void EvFDaqDirector::preBeginRun(edm::GlobalContext const& globalContext) {

    //assert(run_ == id.run());

    // check if the requested run is the latest one - issue a warning if it isn't
    if (dirManager_.findHighestRunDir() != run_dir_) {
      edm::LogWarning("EvFDaqDirector") << "WARNING - checking run dir -: "
					<< run_dir_ << ". This is not the highest run "
					<< dirManager_.findHighestRunDir();
    }
  }

  void EvFDaqDirector::preGlobalEndLumi(edm::GlobalContext const& globalContext)
  {
    //delete all files belonging to just closed lumi
    unsigned int ls = globalContext.luminosityBlockID().luminosityBlock();
    if (!fileDeleteLockPtr_ || !filesToDeletePtr_) {
      edm::LogWarning("EvFDaqDirector") << " Handles to check for files to delete were not set by the input source...";
      return;
    }

    std::unique_lock<std::mutex> lkw(*fileDeleteLockPtr_);
    auto it = filesToDeletePtr_->begin();
    while (it!=filesToDeletePtr_->end()) {
      if (it->second->lumi_ ==  ls) {
        const boost::filesystem::path filePath(it->second->fileName_);
        LogDebug("EvFDaqDirector") << "Deleting input file -:" << it->second->fileName_;
        try {
          //rarely this fails but file gets deleted
          boost::filesystem::remove(filePath);
        }
        catch (const boost::filesystem::filesystem_error& ex)
        {
          edm::LogError("EvFDaqDirector") << " - deleteFile BOOST FILESYSTEM ERROR CAUGHT -: " << ex.what() << ". Trying again.";
          usleep(10000);
          try {
            boost::filesystem::remove(filePath);
          }
            catch (...) {/*file gets deleted first time but exception is still thrown*/}
        }
        catch (std::exception& ex)
        {
          edm::LogError("EvFDaqDirector") << " - deleteFile std::exception CAUGHT -: " << ex.what() << ". Trying again.";
          usleep(10000);
          try {
	    boost::filesystem::remove(filePath);
          } catch (...) {/*file gets deleted first time but exception is still thrown*/}
        }
        
        delete it->second;
	it = filesToDeletePtr_->erase(it);
      }
      else it++;
    }
  }

  inline void EvFDaqDirector::preSourceEvent(edm::StreamID const& streamID) {
    streamFileTracker_[streamID]=currentFileIndex_;
  }


  std::string EvFDaqDirector::getInputJsonFilePath(const unsigned int ls, const unsigned int index) const {
    return bu_run_dir_ + "/" + fffnaming::inputJsonFileName(run_,ls,index);
  }


  std::string EvFDaqDirector::getRawFilePath(const unsigned int ls, const unsigned int index) const {
    return bu_run_dir_ + "/" + fffnaming::inputRawFileName(run_,ls,index);
  }

  std::string EvFDaqDirector::getOpenRawFilePath(const unsigned int ls, const unsigned int index) const {
    return bu_run_dir_ + "/open/" + fffnaming::inputRawFileName(run_,ls,index);
  }

  std::string EvFDaqDirector::getOpenInputJsonFilePath(const unsigned int ls, const unsigned int index) const {
    return bu_run_dir_ + "/open/" + fffnaming::inputJsonFileName(run_,ls,index);
  }

  std::string EvFDaqDirector::getOpenDatFilePath(const unsigned int ls, std::string const& stream) const {
    return run_dir_ + "/open/" + fffnaming::streamerDataFileNameWithPid(run_,ls,stream);
  }

  std::string EvFDaqDirector::getOpenOutputJsonFilePath(const unsigned int ls, std::string const& stream) const {
    return run_dir_ + "/open/" + fffnaming::streamerJsonFileNameWithPid(run_,ls,stream);
  }

  std::string EvFDaqDirector::getOutputJsonFilePath(const unsigned int ls, std::string const& stream) const {
    return run_dir_ + "/" + fffnaming::streamerJsonFileNameWithPid(run_,ls,stream);
  }

  std::string EvFDaqDirector::getMergedDatFilePath(const unsigned int ls, std::string const& stream) const {
    return run_dir_ + "/" + fffnaming::streamerDataFileNameWithInstance(run_,ls,stream,hostname_);
  }

  std::string EvFDaqDirector::getMergedDatChecksumFilePath(const unsigned int ls, std::string const& stream) const {
    return run_dir_ + "/" + fffnaming::streamerDataChecksumFileNameWithInstance(run_,ls,stream,hostname_);
  }

  std::string EvFDaqDirector::getOpenInitFilePath(std::string const& stream) const {
    return run_dir_ + "/open/" + fffnaming::initFileNameWithPid(run_,0,stream);
  }

  std::string EvFDaqDirector::getInitFilePath(std::string const& stream) const {
    return run_dir_ + "/" + fffnaming::initFileNameWithPid(run_,0,stream);
  }

  std::string EvFDaqDirector::getOpenProtocolBufferHistogramFilePath(const unsigned int ls, std::string const& stream) const {
    return run_dir_ + "/open/" + fffnaming::protocolBufferHistogramFileNameWithPid(run_,ls,stream);
  }

  std::string EvFDaqDirector::getProtocolBufferHistogramFilePath(const unsigned int ls, std::string const& stream) const {
    return run_dir_ + "/" + fffnaming::protocolBufferHistogramFileNameWithPid(run_,ls,stream);
  }

  std::string EvFDaqDirector::getMergedProtocolBufferHistogramFilePath(const unsigned int ls, std::string const& stream) const {
    return run_dir_ + "/" + fffnaming:: protocolBufferHistogramFileNameWithInstance(run_,ls,stream,hostname_);
  }

  std::string EvFDaqDirector::getOpenRootHistogramFilePath(const unsigned int ls, std::string const& stream) const {
    return run_dir_ + "/open/" + fffnaming::rootHistogramFileNameWithPid(run_,ls,stream);
  }

  std::string EvFDaqDirector::getRootHistogramFilePath(const unsigned int ls, std::string const& stream) const {
    return run_dir_ + "/" + fffnaming::rootHistogramFileNameWithPid(run_,ls,stream);
  }

  std::string EvFDaqDirector::getMergedRootHistogramFilePath(const unsigned int ls, std::string const& stream) const {
    return run_dir_ + "/" + fffnaming:: rootHistogramFileNameWithInstance(run_,ls,stream,hostname_);
  }

  std::string EvFDaqDirector::getEoLSFilePathOnBU(const unsigned int ls) const {
    return bu_run_dir_ + "/" + fffnaming::eolsFileName(run_,ls);
  }

  std::string EvFDaqDirector::getEoLSFilePathOnFU(const unsigned int ls) const {
    return run_dir_ + "/" + fffnaming::eolsFileName(run_,ls);
  }

  std::string EvFDaqDirector::getBoLSFilePathOnFU(const unsigned int ls) const {
    return run_dir_ + "/" + fffnaming::bolsFileName(run_,ls);
  }

  std::string EvFDaqDirector::getEoRFilePath() const {
    return bu_run_dir_ + "/" + fffnaming::eorFileName(run_);
  }


  std::string EvFDaqDirector::getEoRFilePathOnFU() const {
    return run_dir_ + "/" + fffnaming::eorFileName(run_);
  }

  void EvFDaqDirector::removeFile(std::string filename) {
    int retval = remove(filename.c_str());
    if (retval != 0)
      edm::LogError("EvFDaqDirector") << "Could not remove used file -: " << filename << ". error = "
		                      << strerror(errno);
  }

  void EvFDaqDirector::removeFile(unsigned int ls, unsigned int index) {
    removeFile(getRawFilePath(ls,index));
  }

  EvFDaqDirector::FileStatus EvFDaqDirector::updateFuLock(unsigned int& ls, std::string& nextFile, uint32_t& fsize, uint64_t& lockWaitTime) {
    EvFDaqDirector::FileStatus fileStatus = noFile;

    int retval = -1;
    int lock_attempts = 0;

    struct stat buf;
    int stopFileLS = -1;
    if (stat(stopFilePath_.c_str(),&buf)==0) {
        stopFileLS = readLastLSEntry(stopFilePath_);
        edm::LogWarning("EvFDaqDirector") << "Detected stop request from hltd. Ending run for this process after LS -: " << stopFileLS;
        //return runEnded;
    }

    timeval ts_lockbegin;
    gettimeofday(&ts_lockbegin,0);

    while (retval==-1) {
      retval = fcntl(fu_readwritelock_fd_, F_SETLK, &fu_rw_flk);
      if (retval==-1) usleep(fuLockPollInterval_);
      else continue;

      lock_attempts+=fuLockPollInterval_;
      if (lock_attempts>5000000 ||  errno==116) {
        if (errno==116)
          edm::LogWarning("EvFDaqDirector") << "Stale lock file handle. Checking if run directory and fu.lock file are present" << std::endl;
        else
          edm::LogWarning("EvFDaqDirector") << "Unable to obtain a lock for 5 seconds. Checking if run directory and fu.lock file are present -: errno "
                                            << errno <<":"<< strerror(errno) << std::endl;


        if (stat(getEoLSFilePathOnFU(ls).c_str(),&buf)==0) {
          edm::LogWarning("EvFDaqDirector") << "Detected local EoLS for lumisection "<< ls ; 
          ls++;
          return noFile;
        }

        if (stat(bu_run_dir_.c_str(), &buf)!=0) return runEnded;
        if (stat((bu_run_dir_+"/fu.lock").c_str(), &buf)!=0) return runEnded;
        lock_attempts=0;
      }
    }

    timeval ts_lockend;
    gettimeofday(&ts_lockend,0);
    long deltat = (ts_lockend.tv_usec-ts_lockbegin.tv_usec) + (ts_lockend.tv_sec-ts_lockbegin.tv_sec)*1000000;
    if (deltat>0.) lockWaitTime=deltat;

    
 
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
	else {
	  fscanf(fu_rw_lock_stream, "%u %u", &readLs, &readIndex);
	  edm::LogInfo("EvFDaqDirector") << "Read fu.lock file file -: " << readLs << ":" << readIndex;
        }

	// try to bump
	bool bumpedOk = bumpFile(readLs, readIndex, nextFile, fsize, stopFileLS);
	ls = readLs;
	// there is a new file to grab or lumisection ended
	if (bumpedOk) {
	  // write new data
	  check = fseek(fu_rw_lock_stream, 0, SEEK_SET);
	  if (check == 0) {
	    ftruncate(fu_readwritelock_fd_, 0);
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
	      edm::LogInfo("EvFDaqDirector") << "Written to file -: " << readLs << ":"
			                     << readIndex + 1 << " --> " << readLs + 2
			                     << ":" << readIndex + 1;
	    else
	      LogDebug("EvFDaqDirector") << "Written to file -: " << readLs << ":"
			                     << readIndex + 1;

	  } else
	      throw cms::Exception("EvFDaqDirector") << "seek on fu read/write lock for updating failed with error "
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

    //if new json is present, lock file which FedRawDataInputSource will later unlock
    if (fileStatus==newFile) lockFULocal();

    //release lock at this point
    int retvalu=-1;
    retvalu=fcntl(fu_readwritelock_fd_, F_SETLKW, &fu_rw_fulk);
    if (retvalu==-1) edm::LogError("EvFDaqDirector") << "Error unlocking the fu.lock " << strerror(errno);

#ifdef DEBUG
    edm::LogDebug("EvFDaqDirector") << "Waited during lock -: " << locked_period << " seconds";
#endif

    if ( fileStatus == noFile ) {
      struct stat buf;
      //edm::LogInfo("EvFDaqDirector") << " looking for EoR file: " << getEoRFilePath().c_str();
      if ( stat(getEoRFilePath().c_str(), &buf) == 0 || stat(bu_run_dir_.c_str(), &buf)!=0)
        fileStatus = runEnded;
      if (stopFileLS>=0 && (int)ls > stopFileLS) {
         edm::LogInfo("EvFDaqDirector") << "Reached maximum lumisection set by hltd";
         fileStatus = runEnded;
      }
    }
    return fileStatus;
  }

  int EvFDaqDirector::getNFilesFromEoLS(std::string BUEoLSFile) {

    boost::filesystem::ifstream ij(BUEoLSFile);
    Json::Value deserializeRoot;
    Json::Reader reader;

    if (!reader.parse(ij, deserializeRoot)) {
      edm::LogError("EvFDaqDirector") << "Cannot deserialize input JSON file -:" << BUEoLSFile;
      return -1;
    }

    std::string data;
    DataPoint dp;
    dp.deserialize(deserializeRoot);

    //read definition 
    if (readEolsDefinition_) {
      //std::string def = boost::algorithm::trim(dp.getDefinition());
      std::string def = dp.getDefinition();
      if (!def.size()) readEolsDefinition_=false;
      while (def.size()) {
        std::string fullpath;
        if (def.find('/')==0)
          fullpath = def;
        else
          fullpath = bu_run_dir_+'/'+def; 
        struct stat buf;
        if (stat(fullpath.c_str(), &buf) == 0) {
          DataPointDefinition eolsDpd;
          std::string defLabel = "legend";
          DataPointDefinition::getDataPointDefinitionFor(fullpath, &eolsDpd,&defLabel);
          if (eolsDpd.getNames().size()==0) {
             //try with "data" label if "legend" format is not used
             eolsDpd = DataPointDefinition();
             defLabel="data";
             DataPointDefinition::getDataPointDefinitionFor(fullpath, &eolsDpd,&defLabel);
          }
          for (unsigned int i=0;i<eolsDpd.getNames().size();i++)
            if (eolsDpd.getNames().at(i)=="NFiles")
              eolsNFilesIndex_ = i;
          readEolsDefinition_=false;
          break;
        }
        //check if we can still find definition
        if (def.size()<=1 || def.find('/')==std::string::npos) {
          readEolsDefinition_=false;
          break;
        }
        def = def.substr(def.find('/')+1);
      }
    }

    if (dp.getData().size()>eolsNFilesIndex_)
      data = dp.getData()[eolsNFilesIndex_];
    else {
      edm::LogError("EvFDaqDirector") << " error reading number of files from BU JSON -: " << BUEoLSFile;
      return -1;
    }
    return boost::lexical_cast<int>(data);
  }

  bool EvFDaqDirector::bumpFile(unsigned int& ls, unsigned int& index, std::string& nextFile, uint32_t& fsize, int maxLS) {

    if (previousFileSize_ != 0) {
      if (!fms_) {
        try {
          fms_ = (FastMonitoringService *) (edm::Service<evf::MicroStateService>().operator->());
        } catch (...) {
	        edm::LogError("EvFDaqDirector") <<" FastMonitoringService not found";
        }
      }
      if (fms_) fms_->accumulateFileSize(ls, previousFileSize_);
      previousFileSize_ = 0;
    }

    //reached limit
    if (maxLS>=0 && ls > (unsigned int)maxLS) return false; 

    struct stat buf;
    std::stringstream ss;
    unsigned int nextIndex = index;
    nextIndex++;

    // 1. Check suggested file
    nextFile = getInputJsonFilePath(ls,index);
    if (stat(nextFile.c_str(), &buf) == 0) {
     
      previousFileSize_ = buf.st_size;
      fsize = buf.st_size;
      return true;
    }
    // 2. No file -> lumi ended? (and how many?)
    else {
      std::string BUEoLSFile = getEoLSFilePathOnBU(ls);
      bool eolFound = (stat(BUEoLSFile.c_str(), &buf) == 0);
      unsigned int startingLumi = ls;
      while (eolFound) {

        // recheck that no raw file appeared in the meantime
        if (stat(nextFile.c_str(), &buf) == 0) {
          previousFileSize_ = buf.st_size;
          fsize = buf.st_size;
          return true;
        }

        int indexFilesInLS = getNFilesFromEoLS(BUEoLSFile);
        if (indexFilesInLS < 0)
          //parsing failed
          return false;
        else {
          //check index
          if ((int)index<indexFilesInLS) {
            //we have 2 files, and check for 1 failed... retry (2 will never be here)
            edm::LogError("EvFDaqDirector") << "Potential miss of index file in LS -: " << ls << ". Missing "
                         << nextFile << " because " << indexFilesInLS-1 << " is the highest index expected. Will not update fu.lock file";
            return false;
          }
        }
	// this lumi ended, check for files
	++ls;
        index = 0;

        //reached limit
        if (maxLS>=0 && ls > (unsigned int)maxLS) return false; 

	nextFile = getInputJsonFilePath(ls,0);
	if (stat(nextFile.c_str(), &buf) == 0) {
	  // a new file was found at new lumisection, index 0
	  previousFileSize_ = buf.st_size;
          fsize = buf.st_size;

	  if (testModeNoBuilderUnit_) {
	    // rename ended lumi to + 2
            std::string sourceEol = getEoLSFilePathOnBU(startingLumi);

	    std::string destEol = getEoLSFilePathOnBU(startingLumi+2);

	    std::string cpCmd = "cp " + sourceEol + " " + destEol;
	    edm::LogInfo("EvFDaqDirector") << " testmode: Running copy cmd -: " << cpCmd;
	    int rc = system(cpCmd.c_str());
	    if (rc != 0) {
	      edm::LogError("EvFDaqDirector") << " testmode: COPY EOL FAILED!!!!! -: " << cpCmd;
	    }
	  }

	  return true;
	}
        BUEoLSFile = getEoLSFilePathOnBU(ls);
        eolFound = (stat(BUEoLSFile.c_str(), &buf) == 0);
      }
    }
    // no new file found
    return false;
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
				  S_IRWXU | S_IWGRP | S_IRGRP | S_IWOTH | S_IROTH);
      chmod(fulockfile.c_str(),0766);
    } else {
      fu_readwritelock_fd_ = open(fulockfile.c_str(), O_RDWR, S_IRWXU);
    }
    if (fu_readwritelock_fd_ == -1)
      edm::LogError("EvFDaqDirector") << "problem with creating filedesc for fuwritelock -: " << fulockfile.c_str()
                                      << " create:" << create << " error:" << strerror(errno);
    else
      LogDebug("EvFDaqDirector") << "creating filedesc for fureadwritelock -: "
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
      LogDebug("EvFDaqDirector") << "creating filedesc for datamerge -: "
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

  void EvFDaqDirector::lockFULocal() {
    //fcntl(fulocal_rwlock_fd_, F_SETLKW, &fulocal_rw_flk);
    flock(fulocal_rwlock_fd_,LOCK_SH);
  }

  void EvFDaqDirector::unlockFULocal() {
    //fcntl(fulocal_rwlock_fd_, F_SETLKW, &fulocal_rw_fulk);
    flock(fulocal_rwlock_fd_,LOCK_UN);
  }


  void EvFDaqDirector::lockFULocal2() {
    //fcntl(fulocal_rwlock_fd2_, F_SETLKW, &fulocal_rw_flk2);
    flock(fulocal_rwlock_fd2_,LOCK_EX);
  }

  void EvFDaqDirector::unlockFULocal2() {
    //fcntl(fulocal_rwlock_fd2_, F_SETLKW, &fulocal_rw_fulk2);
    flock(fulocal_rwlock_fd2_,LOCK_UN);
  }


  void EvFDaqDirector::createRunOpendirMaybe() {
    // create open dir if not already there

    boost::filesystem::path openPath = getRunOpenDirPath();
    if (!boost::filesystem::is_directory(openPath)) {
      LogDebug("EvFDaqDirector") << "<open> FU dir not found. Creating... -:" << openPath.string();
      boost::filesystem::create_directories(openPath);
    }
  }


  int EvFDaqDirector::readLastLSEntry(std::string const& file) {

    boost::filesystem::ifstream ij(file);
    Json::Value deserializeRoot;
    Json::Reader reader;

    if (!reader.parse(ij, deserializeRoot)) {
      edm::LogError("EvFDaqDirector") << "Cannot deserialize input JSON file -:" << file;
      return -1;
    }

    int ret = deserializeRoot.get("lastLS","").asInt();
    return ret;

  }

  //if transferSystem PSet is present in the menu, we require it to be complete and consistent for all specified streams
  void EvFDaqDirector::checkTransferSystemPSet()
  {
    transferSystemJson_.reset(new Json::Value);
    if (edm::getProcessParameterSet().existsAs<edm::ParameterSet>("transferSystem",true))
    {
      const edm::ParameterSet& tsPset(edm::getProcessParameterSet().getParameterSet("transferSystem"));

      Json::Value destinationsVal(Json::arrayValue);
      std::vector<std::string> destinations = tsPset.getParameter<std::vector<std::string>>("destinations");
      for (auto & dest: destinations) destinationsVal.append(dest);
      (*transferSystemJson_)["destinations"]=destinationsVal;

      Json::Value modesVal(Json::arrayValue);
      std::vector<std::string> modes = tsPset.getParameter< std::vector<std::string> >("transferModes");
      for (auto & mode: modes) modesVal.append(mode);
      (*transferSystemJson_)["transferModes"]=modesVal;

      for (auto psKeyItr =tsPset.psetTable().begin();psKeyItr!=tsPset.psetTable().end(); ++ psKeyItr) {
        if (psKeyItr->first!="destinations" && psKeyItr->first!="transferModes") {
          const edm::ParameterSet & streamDef = tsPset.getParameterSet(psKeyItr->first);
          Json::Value streamVal;
          for (auto & mode : modes) {
            //validation
            if (!streamDef.existsAs<std::vector<std::string>>(mode,true))
              throw cms::Exception("EvFDaqDirector") << " Missing transfer system specification for -:" << psKeyItr->first << " (transferMode " << mode << ")";
            std::vector<std::string> streamDestinations =  streamDef.getParameter<std::vector<std::string>>(mode);

            Json::Value sDestsValue(Json::arrayValue);

            if (!streamDestinations.size())
              throw cms::Exception("EvFDaqDirector") << " Missing transter system destination(s) for -: "<< psKeyItr->first << ", mode:" << mode;

            for (auto & sdest:streamDestinations) {
              bool sDestValid=false;
              sDestsValue.append(sdest);
              for (auto & dest: destinations) {
                if (dest==sdest) sDestValid=true;
              }
              if (!sDestValid)
                throw cms::Exception("EvFDaqDirector") << " Invalid transter system destination specified for -: "<< psKeyItr->first << ", mode:" << mode << ", dest:"<<sdest;
            }
            streamVal[mode]=sDestsValue;
          }
          (*transferSystemJson_)[psKeyItr->first] = streamVal;
        }
      }
    }
    else {
      if (requireTSPSet_)
        throw cms::Exception("EvFDaqDirector") << "transferSystem PSet not found";
    }
  }

  std::string EvFDaqDirector::getStreamDestinations(std::string const& stream) const
  {
    std::string streamRequestName;
    if (transferSystemJson_->isMember(stream.c_str()))
      streamRequestName = stream;
    else {
      std::stringstream msg;
      msg << "Transfer system mode definitions missing for -: " << stream;
      if (requireTSPSet_)
        throw cms::Exception("EvFDaqDirector") << msg.str();
      else {
        edm::LogWarning("EvFDaqDirector") << msg.str() << " (permissive mode)";
        return std::string("Failsafe");
      }
    }
    //return empty if strict check parameter is not on
    if (!requireTSPSet_ && (selectedTransferMode_=="" || selectedTransferMode_=="null")) {
      edm::LogWarning("EvFDaqDirector") << "Selected mode string is not provided as DaqDirector parameter."
                                        << "Switch on requireTSPSet parameter to enforce this requirement. Setting mode to empty string.";
      return std::string("Failsafe");
    }
    if (requireTSPSet_ && (selectedTransferMode_=="" || selectedTransferMode_=="null")) {
      throw cms::Exception("EvFDaqDirector") << "Selected mode string is not provided as DaqDirector parameter.";
    }
    //check if stream has properly listed transfer stream
    if  (!transferSystemJson_->get(streamRequestName, "").isMember(selectedTransferMode_.c_str()))
    {
         std::stringstream msg;
         msg << "Selected transfer mode " << selectedTransferMode_ << " is not specified for stream " << streamRequestName;
         if (requireTSPSet_)
           throw cms::Exception("EvFDaqDirector") << msg.str();
         else
           edm::LogWarning("EvFDaqDirector") << msg.str() << " (permissive mode)"; 
           return std::string("Failsafe");
    }
    Json::Value destsVec = transferSystemJson_->get(streamRequestName, "").get(selectedTransferMode_,"");

    //flatten string json::Array into CSV std::string
    std::string ret;
    for (Json::Value::iterator it = destsVec.begin(); it!=destsVec.end(); it++)
    {
      if (ret!="") ret +=",";
      ret+=(*it).asString();
    }
    return ret;
  }

}
