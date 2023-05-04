#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/SystemBounds.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/ProcessContext.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/UnixSignalHandlers.h"

#include "EventFilter/Utilities/interface/EvFDaqDirector.h"
#include "EventFilter/Utilities/interface/FastMonitoringService.h"
#include "EventFilter/Utilities/interface/FedRawDataInputSource.h"
#include "EventFilter/Utilities/interface/DataPointDefinition.h"
#include "EventFilter/Utilities/interface/DataPoint.h"
#include "IOPool/Streamer/interface/FRDEventMessage.h"
#include "IOPool/Streamer/interface/FRDFileHeader.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <sys/time.h>
#include <unistd.h>
#include <cstdio>
#include <boost/algorithm/string.hpp>

//using boost::asio::ip::tcp;

//#define DEBUG

using namespace jsoncollector;

namespace evf {

  //for enum MergeType
  const std::vector<std::string> EvFDaqDirector::MergeTypeNames_ = {"", "DAT", "PB", "JSNDATA"};

  EvFDaqDirector::EvFDaqDirector(const edm::ParameterSet& pset, edm::ActivityRegistry& reg)
      : base_dir_(pset.getUntrackedParameter<std::string>("baseDir")),
        bu_base_dir_(pset.getUntrackedParameter<std::string>("buBaseDir")),
        bu_base_dirs_all_(pset.getUntrackedParameter<std::vector<std::string>>("buBaseDirsAll")),
        run_(pset.getUntrackedParameter<unsigned int>("runNumber")),
        useFileBroker_(pset.getUntrackedParameter<bool>("useFileBroker")),
        fileBrokerHostFromCfg_(pset.getUntrackedParameter<bool>("fileBrokerHostFromCfg", true)),
        fileBrokerHost_(pset.getUntrackedParameter<std::string>("fileBrokerHost", "InValid")),
        fileBrokerPort_(pset.getUntrackedParameter<std::string>("fileBrokerPort", "8080")),
        fileBrokerKeepAlive_(pset.getUntrackedParameter<bool>("fileBrokerKeepAlive", true)),
        fileBrokerUseLocalLock_(pset.getUntrackedParameter<bool>("fileBrokerUseLocalLock", true)),
        fuLockPollInterval_(pset.getUntrackedParameter<unsigned int>("fuLockPollInterval", 2000)),
        outputAdler32Recheck_(pset.getUntrackedParameter<bool>("outputAdler32Recheck", false)),
        requireTSPSet_(pset.getUntrackedParameter<bool>("requireTransfersPSet", false)),
        selectedTransferMode_(pset.getUntrackedParameter<std::string>("selectedTransferMode", "")),
        mergeTypePset_(pset.getUntrackedParameter<std::string>("mergingPset", "")),
        directorBU_(pset.getUntrackedParameter<bool>("directorIsBU", false)),
        hltSourceDirectory_(pset.getUntrackedParameter<std::string>("hltSourceDirectory", "")),
        hostname_(""),
        bu_readlock_fd_(-1),
        bu_writelock_fd_(-1),
        fu_readwritelock_fd_(-1),
        fulocal_rwlock_fd_(-1),
        fulocal_rwlock_fd2_(-1),
        bu_w_lock_stream(nullptr),
        bu_r_lock_stream(nullptr),
        fu_rw_lock_stream(nullptr),
        dirManager_(base_dir_),
        previousFileSize_(0),
        bu_w_flk(make_flock(F_WRLCK, SEEK_SET, 0, 0, 0)),
        bu_r_flk(make_flock(F_RDLCK, SEEK_SET, 0, 0, 0)),
        bu_w_fulk(make_flock(F_UNLCK, SEEK_SET, 0, 0, 0)),
        bu_r_fulk(make_flock(F_UNLCK, SEEK_SET, 0, 0, 0)),
        fu_rw_flk(make_flock(F_WRLCK, SEEK_SET, 0, 0, getpid())),
        fu_rw_fulk(make_flock(F_UNLCK, SEEK_SET, 0, 0, getpid())) {
    reg.watchPreallocate(this, &EvFDaqDirector::preallocate);
    reg.watchPreBeginJob(this, &EvFDaqDirector::preBeginJob);
    reg.watchPreGlobalBeginRun(this, &EvFDaqDirector::preBeginRun);
    reg.watchPostGlobalEndRun(this, &EvFDaqDirector::postEndRun);
    reg.watchPreGlobalEndLumi(this, &EvFDaqDirector::preGlobalEndLumi);

    //save hostname for later
    char hostname[33];
    gethostname(hostname, 32);
    hostname_ = hostname;

    char* fuLockPollIntervalPtr = std::getenv("FFF_LOCKPOLLINTERVAL");
    if (fuLockPollIntervalPtr) {
      try {
        fuLockPollInterval_ = std::stoul(std::string(fuLockPollIntervalPtr));
        edm::LogInfo("EvFDaqDirector") << "Setting fu lock poll interval by environment string: " << fuLockPollInterval_
                                       << " us";
      } catch (const std::exception&) {
        edm::LogWarning("EvFDaqDirector") << "Bad lexical cast in parsing: " << std::string(fuLockPollIntervalPtr);
      }
    }

    //override file service parameter if specified by environment
    char* fileBrokerParamPtr = std::getenv("FFF_USEFILEBROKER");
    if (fileBrokerParamPtr) {
      try {
        useFileBroker_ = (std::stoul(std::string(fileBrokerParamPtr))) > 0;
        edm::LogInfo("EvFDaqDirector") << "Setting useFileBroker parameter by environment string: " << useFileBroker_;
      } catch (const std::exception&) {
        edm::LogWarning("EvFDaqDirector") << "Bad lexical cast in parsing: " << std::string(fileBrokerParamPtr);
      }
    }
    if (useFileBroker_) {
      if (fileBrokerHostFromCfg_) {
        //find BU data address from hltd configuration
        fileBrokerHost_ = std::string();
        struct stat buf;
        if (stat("/etc/appliance/bus.config", &buf) == 0) {
          std::ifstream busconfig("/etc/appliance/bus.config", std::ifstream::in);
          std::getline(busconfig, fileBrokerHost_);
        }
        if (fileBrokerHost_.empty())
          throw cms::Exception("EvFDaqDirector") << "No file service or BU data address information";
      } else if (fileBrokerHost_.empty() || fileBrokerHost_ == "InValid")
        throw cms::Exception("EvFDaqDirector")
            << "fileBrokerHostFromCfg must be set to true if fileBrokerHost parameter is not valid or empty";

      resolver_ = std::make_unique<boost::asio::ip::tcp::resolver>(io_service_);
      query_ = std::make_unique<boost::asio::ip::tcp::resolver::query>(fileBrokerHost_, fileBrokerPort_);
      endpoint_iterator_ = std::make_unique<boost::asio::ip::tcp::resolver::iterator>(resolver_->resolve(*query_));
      socket_ = std::make_unique<boost::asio::ip::tcp::socket>(io_service_);
    }

    char* startFromLSPtr = std::getenv("FFF_START_LUMISECTION");
    if (startFromLSPtr) {
      try {
        startFromLS_ = std::stoul(std::string(startFromLSPtr));
        edm::LogInfo("EvFDaqDirector") << "Setting start from LS by environment string: " << startFromLS_;
      } catch (const std::exception&) {
        edm::LogWarning("EvFDaqDirector") << "Bad lexical cast in parsing: " << std::string(startFromLSPtr);
      }
    }

    //override file service parameter if specified by environment
    char* fileBrokerUseLockParamPtr = std::getenv("FFF_FILEBROKERUSELOCALLOCK");
    if (fileBrokerUseLockParamPtr) {
      try {
        fileBrokerUseLocalLock_ = (std::stoul(std::string(fileBrokerUseLockParamPtr))) > 0;
        edm::LogInfo("EvFDaqDirector") << "Setting fileBrokerUseLocalLock parameter by environment string: "
                                       << fileBrokerUseLocalLock_;
      } catch (const std::exception&) {
        edm::LogWarning("EvFDaqDirector") << "Bad lexical cast in parsing: " << std::string(fileBrokerUseLockParamPtr);
      }
    }

    std::stringstream ss;
    ss << "run" << std::setfill('0') << std::setw(6) << run_;
    run_string_ = ss.str();
    ss = std::stringstream();
    ss << run_;
    run_nstring_ = ss.str();
    run_dir_ = base_dir_ + "/" + run_string_;
    input_throttled_file_ = run_dir_ + "/input_throttle";
    discard_ls_filestem_ = run_dir_ + "/discard_ls";
    ss = std::stringstream();
    ss << getpid();
    pid_ = ss.str();
  }

  void EvFDaqDirector::initRun() {
    // check if base dir exists or create it accordingly
    int retval = mkdir(base_dir_.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (retval != 0 && errno != EEXIST) {
      throw cms::Exception("DaqDirector")
          << " Error checking for base dir -: " << base_dir_ << " mkdir error:" << strerror(errno);
    }

    //create run dir in base dir
    umask(0);
    retval = mkdir(run_dir_.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IRWXO | S_IXOTH);
    if (retval != 0 && errno != EEXIST) {
      throw cms::Exception("DaqDirector")
          << " Error creating run dir -: " << run_dir_ << " mkdir error:" << strerror(errno);
    }

    //create fu-local.lock in run open dir
    if (!directorBU_) {
      createRunOpendirMaybe();
      std::string fulocal_lock_ = getRunOpenDirPath() + "/fu-local.lock";
      fulocal_rwlock_fd_ =
          open(fulocal_lock_.c_str(), O_RDWR | O_CREAT, S_IRWXU | S_IWGRP | S_IRGRP | S_IWOTH | S_IROTH);  //O_RDWR?
      if (fulocal_rwlock_fd_ == -1)
        throw cms::Exception("DaqDirector")
            << " Error creating/opening a local lock file -: " << fulocal_lock_.c_str() << " : " << strerror(errno);
      chmod(fulocal_lock_.c_str(), 0777);
      fsync(fulocal_rwlock_fd_);
      //open second fd for another input source thread
      fulocal_rwlock_fd2_ =
          open(fulocal_lock_.c_str(), O_RDWR, S_IRWXU | S_IWGRP | S_IRGRP | S_IWOTH | S_IROTH);  //O_RDWR?
      if (fulocal_rwlock_fd2_ == -1)
        throw cms::Exception("DaqDirector")
            << " Error opening a local lock file -: " << fulocal_lock_.c_str() << " : " << strerror(errno);
    }

    //bu_run_dir: for FU, for which the base dir is local and the BU is remote, it is expected to be there
    //for BU, it is created at this point
    if (directorBU_) {
      bu_run_dir_ = base_dir_ + "/" + run_string_;
      std::string bulockfile = bu_run_dir_ + "/bu.lock";
      fulockfile_ = bu_run_dir_ + "/fu.lock";

      //make or find bu run dir
      retval = mkdir(bu_run_dir_.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
      if (retval != 0 && errno != EEXIST) {
        throw cms::Exception("DaqDirector")
            << " Error creating bu run dir -: " << bu_run_dir_ << " mkdir error:" << strerror(errno);
      }
      bu_run_open_dir_ = bu_run_dir_ + "/open";
      retval = mkdir(bu_run_open_dir_.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
      if (retval != 0 && errno != EEXIST) {
        throw cms::Exception("DaqDirector")
            << " Error creating bu run open dir -: " << bu_run_open_dir_ << " mkdir error:" << strerror(errno);
      }

      // the BU director does not need to know about the fu lock
      bu_writelock_fd_ = open(bulockfile.c_str(), O_WRONLY | O_CREAT | O_TRUNC, S_IRWXU);
      if (bu_writelock_fd_ == -1)
        edm::LogWarning("EvFDaqDirector") << "problem with creating filedesc for buwritelock -: " << strerror(errno);
      else
        edm::LogInfo("EvFDaqDirector") << "creating filedesc for buwritelock -: " << bu_writelock_fd_;
      bu_w_lock_stream = fdopen(bu_writelock_fd_, "w");
      if (bu_w_lock_stream == nullptr)
        edm::LogWarning("EvFDaqDirector") << "Error creating write lock stream -: " << strerror(errno);

      // BU INITIALIZES LOCK FILE
      // FU LOCK FILE OPEN
      openFULockfileStream(true);
      tryInitializeFuLockFile();
      fflush(fu_rw_lock_stream);
      close(fu_readwritelock_fd_);

      if (!hltSourceDirectory_.empty()) {
        struct stat buf;
        if (stat(hltSourceDirectory_.c_str(), &buf) == 0) {
          std::string hltdir = bu_run_dir_ + "/hlt";
          std::string tmphltdir = bu_run_open_dir_ + "/hlt";
          retval = mkdir(tmphltdir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
          if (retval != 0 && errno != EEXIST)
            throw cms::Exception("DaqDirector")
                << " Error creating bu run dir -: " << hltdir << " mkdir error:" << strerror(errno);

          std::filesystem::copy_file(hltSourceDirectory_ + "/HltConfig.py", tmphltdir + "/HltConfig.py");
          std::filesystem::copy_file(hltSourceDirectory_ + "/fffParameters.jsn", tmphltdir + "/fffParameters.jsn");

          std::string optfiles[3] = {"hltinfo", "blacklist", "whitelist"};
          for (auto& optfile : optfiles) {
            try {
              std::filesystem::copy_file(hltSourceDirectory_ + "/" + optfile, tmphltdir + "/" + optfile);
            } catch (...) {
            }
          }

          std::filesystem::rename(tmphltdir, hltdir);
        } else
          throw cms::Exception("DaqDirector") << " Error looking for HLT configuration -: " << hltSourceDirectory_;
      }
      //else{}//no configuration specified
    } else {
      // for FU, check if bu base dir exists

      auto checkExists = [=](std::string const& bu_base_dir) -> void {
        int retval = mkdir(bu_base_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (retval != 0 && errno != EEXIST) {
          throw cms::Exception("DaqDirector")
              << " Error checking for bu base dir -: " << bu_base_dir << " mkdir error:" << strerror(errno);
        }
      };

      auto waitForDir = [=](std::string const& bu_base_dir) -> void {
        int cnt = 0;
        while (!edm::shutdown_flag.load(std::memory_order_relaxed)) {
          int retval = mkdir(bu_base_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
          if (retval != 0 && errno != EEXIST) {
            usleep(500000);
            cnt++;
            if (cnt % 20 == 0)
              edm::LogWarning("DaqDirector") << "waiting for " << bu_base_dir;
            if (cnt > 120)
              throw cms::Exception("DaqDirector") << " Error checking for bu base dir after 1 minute -: " << bu_base_dir
                                                  << " mkdir error:" << strerror(errno);
          }
          break;
        }
      };

      if (!bu_base_dirs_all_.empty()) {
        checkExists(bu_base_dirs_all_[0]);
        bu_run_dir_ = bu_base_dirs_all_[0] + "/" + run_string_;
        for (unsigned int i = 1; i < bu_base_dirs_all_.size(); i++)
          waitForDir(bu_base_dirs_all_[i]);
      } else {
        checkExists(bu_base_dir_);
        bu_run_dir_ = bu_base_dir_ + "/" + run_string_;
      }

      fulockfile_ = bu_run_dir_ + "/fu.lock";
      if (!useFileBroker_)
        openFULockfileStream(false);
    }

    pthread_mutex_init(&init_lock_, nullptr);

    stopFilePath_ = run_dir_ + "/CMSSW_STOP";
    std::stringstream sstp;
    sstp << stopFilePath_ << "_pid" << pid_;
    stopFilePathPid_ = sstp.str();

    if (!directorBU_) {
      std::string defPath = bu_run_dir_ + "/jsd/rawData.jsd";
      struct stat statbuf;
      if (!stat(defPath.c_str(), &statbuf))
        edm::LogInfo("EvFDaqDirector") << "found JSD file in ramdisk -: " << defPath;
      else {
        //look in source directory if not present in ramdisk
        std::string defPathSuffix = "src/EventFilter/Utilities/plugins/budef.jsd";
        defPath = std::string(std::getenv("CMSSW_BASE")) + "/" + defPathSuffix;
        if (stat(defPath.c_str(), &statbuf)) {
          defPath = std::string(std::getenv("CMSSW_RELEASE_BASE")) + "/" + defPathSuffix;
          if (stat(defPath.c_str(), &statbuf)) {
            defPath = defPathSuffix;
          }
        }
      }
      dpd_ = new DataPointDefinition();
      std::string defLabel = "data";
      DataPointDefinition::getDataPointDefinitionFor(defPath, dpd_, &defLabel);
    }
  }

  EvFDaqDirector::~EvFDaqDirector() {
    //close server connection
    if (socket_.get() && socket_->is_open()) {
      boost::system::error_code ec;
      socket_->shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
      socket_->close(ec);
    }

    if (fulocal_rwlock_fd_ != -1) {
      unlockFULocal();
      close(fulocal_rwlock_fd_);
    }

    if (fulocal_rwlock_fd2_ != -1) {
      unlockFULocal2();
      close(fulocal_rwlock_fd2_);
    }
  }

  void EvFDaqDirector::preallocate(edm::service::SystemBounds const& bounds) {
    initRun();

    nThreads_ = bounds.maxNumberOfStreams();
    nStreams_ = bounds.maxNumberOfThreads();
    nConcurrentLumis_ = bounds.maxNumberOfConcurrentLuminosityBlocks();
  }

  void EvFDaqDirector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.setComment(
        "Service used for file locking arbitration and for propagating information between other EvF components");
    desc.addUntracked<std::string>("baseDir", ".")->setComment("Local base directory for run output");
    desc.addUntracked<std::string>("buBaseDir", ".")->setComment("BU base ramdisk directory ");
    desc.addUntracked<std::vector<std::string>>("buBaseDirsAll", std::vector<std::string>())
        ->setComment("BU base ramdisk directories for multi-file DAQSource models");
    desc.addUntracked<unsigned int>("runNumber", 0)->setComment("Run Number in ramdisk to open");
    desc.addUntracked<bool>("useFileBroker", false)
        ->setComment("Use BU file service to grab input data instead of NFS file locking");
    desc.addUntracked<bool>("fileBrokerHostFromCfg", true)
        ->setComment("Allow service to discover BU address from hltd configuration");
    desc.addUntracked<std::string>("fileBrokerHost", "InValid")->setComment("BU file service host.");
    desc.addUntracked<std::string>("fileBrokerPort", "8080")->setComment("BU file service port");
    desc.addUntracked<bool>("fileBrokerKeepAlive", true)
        ->setComment("Use keep alive to avoid using large number of sockets");
    desc.addUntracked<bool>("fileBrokerUseLocalLock", true)
        ->setComment("Use local lock file to synchronize appearance of index and EoLS file markers for hltd");
    desc.addUntracked<unsigned int>("fuLockPollInterval", 2000)
        ->setComment("Lock polling interval in microseconds for the input directory file lock");
    desc.addUntracked<bool>("outputAdler32Recheck", false)
        ->setComment("Check Adler32 of per-process output files while micro-merging");
    desc.addUntracked<bool>("requireTransfersPSet", false)
        ->setComment("Require complete transferSystem PSet in the process configuration");
    desc.addUntracked<std::string>("selectedTransferMode", "")
        ->setComment("Selected transfer mode (choice in Lvl0 propagated as Python parameter");
    desc.addUntracked<bool>("directorIsBU", false)->setComment("BU director mode used for testing");
    desc.addUntracked<std::string>("hltSourceDirectory", "")->setComment("BU director mode source directory");
    desc.addUntracked<std::string>("mergingPset", "")
        ->setComment("Name of merging PSet to look for merging type definitions for streams");
    descriptions.add("EvFDaqDirector", desc);
  }

  void EvFDaqDirector::preBeginJob(edm::PathsAndConsumesOfModulesBase const&, edm::ProcessContext const& pc) {
    checkTransferSystemPSet(pc);
    checkMergeTypePSet(pc);
  }

  void EvFDaqDirector::preBeginRun(edm::GlobalContext const& globalContext) {
    //assert(run_ == id.run());

    // check if the requested run is the latest one - issue a warning if it isn't
    if (dirManager_.findHighestRunDir() != run_dir_) {
      edm::LogWarning("EvFDaqDirector") << "WARNING - checking run dir -: " << run_dir_
                                        << ". This is not the highest run " << dirManager_.findHighestRunDir();
    }
  }

  void EvFDaqDirector::postEndRun(edm::GlobalContext const& globalContext) {
    close(bu_readlock_fd_);
    close(bu_writelock_fd_);
    if (directorBU_) {
      std::string filename = bu_run_dir_ + "/bu.lock";
      removeFile(filename);
    }
  }

  void EvFDaqDirector::preGlobalEndLumi(edm::GlobalContext const& globalContext) {
    //delete all files belonging to just closed lumi
    unsigned int ls = globalContext.luminosityBlockID().luminosityBlock();
    if (!fileDeleteLockPtr_ || !filesToDeletePtr_) {
      edm::LogWarning("EvFDaqDirector") << " Handles to check for files to delete were not set by the input source...";
      return;
    }

    std::unique_lock<std::mutex> lkw(*fileDeleteLockPtr_);
    auto it = filesToDeletePtr_->begin();
    while (it != filesToDeletePtr_->end()) {
      if (it->second->lumi_ == ls && (!fms_ || !fms_->isExceptionOnData(it->second->lumi_))) {
        it = filesToDeletePtr_->erase(it);
      } else
        it++;
    }
  }

  std::string EvFDaqDirector::getInputJsonFilePath(const unsigned int ls, const unsigned int index) const {
    return bu_run_dir_ + "/" + fffnaming::inputJsonFileName(run_, ls, index);
  }

  std::string EvFDaqDirector::getRawFilePath(const unsigned int ls, const unsigned int index) const {
    return bu_run_dir_ + "/" + fffnaming::inputRawFileName(run_, ls, index);
  }

  std::string EvFDaqDirector::getOpenRawFilePath(const unsigned int ls, const unsigned int index) const {
    return bu_run_dir_ + "/open/" + fffnaming::inputRawFileName(run_, ls, index);
  }

  std::string EvFDaqDirector::getOpenInputJsonFilePath(const unsigned int ls, const unsigned int index) const {
    return bu_run_dir_ + "/open/" + fffnaming::inputJsonFileName(run_, ls, index);
  }

  std::string EvFDaqDirector::getDatFilePath(const unsigned int ls, std::string const& stream) const {
    return run_dir_ + "/" + fffnaming::streamerDataFileNameWithPid(run_, ls, stream);
  }

  std::string EvFDaqDirector::getOpenDatFilePath(const unsigned int ls, std::string const& stream) const {
    return run_dir_ + "/open/" + fffnaming::streamerDataFileNameWithPid(run_, ls, stream);
  }

  std::string EvFDaqDirector::getOpenOutputJsonFilePath(const unsigned int ls, std::string const& stream) const {
    return run_dir_ + "/open/" + fffnaming::streamerJsonFileNameWithPid(run_, ls, stream);
  }

  std::string EvFDaqDirector::getOutputJsonFilePath(const unsigned int ls, std::string const& stream) const {
    return run_dir_ + "/" + fffnaming::streamerJsonFileNameWithPid(run_, ls, stream);
  }

  std::string EvFDaqDirector::getMergedDatFilePath(const unsigned int ls, std::string const& stream) const {
    return run_dir_ + "/" + fffnaming::streamerDataFileNameWithInstance(run_, ls, stream, hostname_);
  }

  std::string EvFDaqDirector::getMergedDatChecksumFilePath(const unsigned int ls, std::string const& stream) const {
    return run_dir_ + "/" + fffnaming::streamerDataChecksumFileNameWithInstance(run_, ls, stream, hostname_);
  }

  std::string EvFDaqDirector::getOpenInitFilePath(std::string const& stream) const {
    return run_dir_ + "/open/" + fffnaming::initFileNameWithPid(run_, 0, stream);
  }

  std::string EvFDaqDirector::getInitFilePath(std::string const& stream) const {
    return run_dir_ + "/" + fffnaming::initFileNameWithPid(run_, 0, stream);
  }

  std::string EvFDaqDirector::getInitTempFilePath(std::string const& stream) const {
    return run_dir_ + "/" + fffnaming::initTempFileNameWithPid(run_, 0, stream);
  }

  std::string EvFDaqDirector::getOpenProtocolBufferHistogramFilePath(const unsigned int ls,
                                                                     std::string const& stream) const {
    return run_dir_ + "/open/" + fffnaming::protocolBufferHistogramFileNameWithPid(run_, ls, stream);
  }

  std::string EvFDaqDirector::getProtocolBufferHistogramFilePath(const unsigned int ls,
                                                                 std::string const& stream) const {
    return run_dir_ + "/" + fffnaming::protocolBufferHistogramFileNameWithPid(run_, ls, stream);
  }

  std::string EvFDaqDirector::getMergedProtocolBufferHistogramFilePath(const unsigned int ls,
                                                                       std::string const& stream) const {
    return run_dir_ + "/" + fffnaming::protocolBufferHistogramFileNameWithInstance(run_, ls, stream, hostname_);
  }

  std::string EvFDaqDirector::getOpenRootHistogramFilePath(const unsigned int ls, std::string const& stream) const {
    return run_dir_ + "/open/" + fffnaming::rootHistogramFileNameWithPid(run_, ls, stream);
  }

  std::string EvFDaqDirector::getRootHistogramFilePath(const unsigned int ls, std::string const& stream) const {
    return run_dir_ + "/" + fffnaming::rootHistogramFileNameWithPid(run_, ls, stream);
  }

  std::string EvFDaqDirector::getMergedRootHistogramFilePath(const unsigned int ls, std::string const& stream) const {
    return run_dir_ + "/" + fffnaming::rootHistogramFileNameWithInstance(run_, ls, stream, hostname_);
  }

  std::string EvFDaqDirector::getEoLSFilePathOnBU(const unsigned int ls) const {
    return bu_run_dir_ + "/" + fffnaming::eolsFileName(run_, ls);
  }

  std::string EvFDaqDirector::getEoLSFilePathOnFU(const unsigned int ls) const {
    return run_dir_ + "/" + fffnaming::eolsFileName(run_, ls);
  }

  std::string EvFDaqDirector::getBoLSFilePathOnFU(const unsigned int ls) const {
    return run_dir_ + "/" + fffnaming::bolsFileName(run_, ls);
  }

  std::string EvFDaqDirector::getEoRFilePath() const { return bu_run_dir_ + "/" + fffnaming::eorFileName(run_); }

  std::string EvFDaqDirector::getEoRFileName() const { return fffnaming::eorFileName(run_); }

  std::string EvFDaqDirector::getEoRFilePathOnFU() const { return run_dir_ + "/" + fffnaming::eorFileName(run_); }

  std::string EvFDaqDirector::getFFFParamsFilePathOnBU() const { return bu_run_dir_ + "/hlt/fffParameters.jsn"; }

  void EvFDaqDirector::removeFile(std::string filename) {
    int retval = remove(filename.c_str());
    if (retval != 0)
      edm::LogError("EvFDaqDirector") << "Could not remove used file -: " << filename
                                      << ". error = " << strerror(errno);
  }

  EvFDaqDirector::FileStatus EvFDaqDirector::updateFuLock(unsigned int& ls,
                                                          std::string& nextFile,
                                                          uint32_t& fsize,
                                                          uint16_t& rawHeaderSize,
                                                          uint64_t& lockWaitTime,
                                                          bool& setExceptionState) {
    EvFDaqDirector::FileStatus fileStatus = noFile;
    rawHeaderSize = 0;

    int retval = -1;
    int lock_attempts = 0;
    long total_lock_attempts = 0;

    struct stat buf;
    int stopFileLS = -1;
    int stopFileCheck = stat(stopFilePath_.c_str(), &buf);
    int stopFilePidCheck = stat(stopFilePathPid_.c_str(), &buf);
    if (stopFileCheck == 0 || stopFilePidCheck == 0) {
      if (stopFileCheck == 0)
        stopFileLS = readLastLSEntry(stopFilePath_);
      else
        stopFileLS = 1;  //stop without drain if only pid is stopped
      if (!stop_ls_override_) {
        //if lumisection is higher than in stop file, should quit at next from current
        if (stopFileLS >= 0 && (int)ls >= stopFileLS)
          stopFileLS = stop_ls_override_ = ls;
      } else
        stopFileLS = stop_ls_override_;
      edm::LogWarning("EvFDaqDirector") << "Detected stop request from hltd. Ending run for this process after LS -: "
                                        << stopFileLS;
      //return runEnded;
    } else  //if file was removed before reaching stop condition, reset this
      stop_ls_override_ = 0;

    timeval ts_lockbegin;
    gettimeofday(&ts_lockbegin, nullptr);

    while (retval == -1) {
      retval = fcntl(fu_readwritelock_fd_, F_SETLK, &fu_rw_flk);
      if (retval == -1)
        usleep(fuLockPollInterval_);
      else
        continue;

      lock_attempts += fuLockPollInterval_;
      total_lock_attempts += fuLockPollInterval_;
      if (lock_attempts > 5000000 || errno == 116) {
        if (errno == 116)
          edm::LogWarning("EvFDaqDirector")
              << "Stale lock file handle. Checking if run directory and fu.lock file are present" << std::endl;
        else
          edm::LogWarning("EvFDaqDirector") << "Unable to obtain a lock for 5 seconds. Checking if run directory and "
                                               "fu.lock file are present -: errno "
                                            << errno << ":" << strerror(errno) << std::endl;

        if (stat(getEoLSFilePathOnFU(ls).c_str(), &buf) == 0) {
          edm::LogWarning("EvFDaqDirector") << "Detected local EoLS for lumisection " << ls;
          ls++;
          return noFile;
        }

        if (stat(bu_run_dir_.c_str(), &buf) != 0)
          return runEnded;
        if (stat(fulockfile_.c_str(), &buf) != 0)
          return runEnded;

        lock_attempts = 0;
      }
      if (total_lock_attempts > 5 * 60000000) {
        edm::LogError("EvFDaqDirector") << "Unable to obtain a lock for 5 minutes. Stopping polling activity.";
        return runAbort;
      }
    }

    timeval ts_lockend;
    gettimeofday(&ts_lockend, nullptr);
    long deltat = (ts_lockend.tv_usec - ts_lockbegin.tv_usec) + (ts_lockend.tv_sec - ts_lockbegin.tv_sec) * 1000000;
    if (deltat > 0.)
      lockWaitTime = deltat;

    if (retval != 0)
      return fileStatus;

#ifdef DEBUG
    timeval ts_lockend;
    gettimeofday(&ts_lockend, 0);
#endif

    //open another lock file FD after the lock using main fd has been acquired
    int fu_readwritelock_fd2 = open(fulockfile_.c_str(), O_RDWR, S_IRWXU);
    if (fu_readwritelock_fd2 == -1)
      edm::LogError("EvFDaqDirector") << "problem with creating filedesc for fuwritelock -: " << fulockfile_
                                      << " create. error:" << strerror(errno);

    FILE* fu_rw_lock_stream2 = fdopen(fu_readwritelock_fd2, "r+");

    // if the stream is readable
    if (fu_rw_lock_stream2 != nullptr) {
      unsigned int readLs, readIndex;
      int check = 0;
      // rewind the stream
      check = fseek(fu_rw_lock_stream2, 0, SEEK_SET);
      // if rewinded ok
      if (check == 0) {
        // read its' values
        fscanf(fu_rw_lock_stream2, "%u %u", &readLs, &readIndex);
        edm::LogInfo("EvFDaqDirector") << "Read fu.lock file file -: " << readLs << ":" << readIndex;

        unsigned int currentLs = readLs;
        bool bumpedOk = false;
        //if next lumisection in a lock file is not +1 wrt. source, cycle through the next empty one, unless initial lumi not yet set
        //no lock file write in this case
        if (ls && ls + 1 < currentLs)
          ls++;
        else {
          // try to bump (look for new index or EoLS file)
          bumpedOk = bumpFile(readLs, readIndex, nextFile, fsize, rawHeaderSize, stopFileLS, setExceptionState);
          //avoid 2 lumisections jump
          if (ls && readLs > currentLs && currentLs > ls) {
            ls++;
            readLs = currentLs = ls;
            readIndex = 0;
            bumpedOk = false;
            //no write to lock file
          } else {
            if (ls == 0 && readLs > currentLs) {
              //make sure to intialize always with LS found in the lock file, with possibility of grabbing index file immediately
              //in this case there is no new file in the same LS
              //this covers case where run has empty first lumisections and CMSSW are late to the lock file. always one process will start with LS 1,... and create empty files for them
              readLs = currentLs;
              readIndex = 0;
              bumpedOk = false;
              //no write to lock file
            }
            //update return LS value
            ls = readLs;
          }
        }
        if (bumpedOk) {
          // there is a new index file to grab, lock file needs to be updated
          check = fseek(fu_rw_lock_stream2, 0, SEEK_SET);
          if (check == 0) {
            ftruncate(fu_readwritelock_fd2, 0);
            // write next index in the file, which is the file the next process should take
            fprintf(fu_rw_lock_stream2, "%u %u", readLs, readIndex + 1);
            fflush(fu_rw_lock_stream2);
            fsync(fu_readwritelock_fd2);
            fileStatus = newFile;
            LogDebug("EvFDaqDirector") << "Written to file -: " << readLs << ":" << readIndex + 1;
          } else {
            edm::LogError("EvFDaqDirector")
                << "seek on fu read/write lock for updating failed with error " << strerror(errno);
            setExceptionState = true;
            return noFile;
          }
        } else if (currentLs < readLs) {
          //there is no new file in next LS (yet), but lock file can be updated to the next LS
          check = fseek(fu_rw_lock_stream2, 0, SEEK_SET);
          if (check == 0) {
            ftruncate(fu_readwritelock_fd2, 0);
            // in this case LS was bumped, but no new file. Thus readIndex is 0 (set by bumpFile)
            fprintf(fu_rw_lock_stream2, "%u %u", readLs, readIndex);
            fflush(fu_rw_lock_stream2);
            fsync(fu_readwritelock_fd2);
            LogDebug("EvFDaqDirector") << "Written to file -: " << readLs << ":" << readIndex;
          } else {
            edm::LogError("EvFDaqDirector")
                << "seek on fu read/write lock for updating failed with error " << strerror(errno);
            setExceptionState = true;
            return noFile;
          }
        }
      } else {
        edm::LogError("EvFDaqDirector") << "seek on fu read/write lock for reading failed with error "
                                        << strerror(errno);
      }
    } else {
      edm::LogError("EvFDaqDirector") << "fu read/write lock stream is invalid " << strerror(errno);
    }
    fclose(fu_rw_lock_stream2);  // = fdopen(fu_readwritelock_fd2, "r+");

#ifdef DEBUG
    timeval ts_preunlock;
    gettimeofday(&ts_preunlock, 0);
    int locked_period_int = ts_preunlock.tv_sec - ts_lockend.tv_sec;
    double locked_period = locked_period_int + double(ts_preunlock.tv_usec - ts_lockend.tv_usec) / 1000000;
#endif

    //if new json is present, lock file which FedRawDataInputSource will later unlock
    if (fileStatus == newFile)
      lockFULocal();

    //release lock at this point
    int retvalu = -1;
    retvalu = fcntl(fu_readwritelock_fd_, F_SETLKW, &fu_rw_fulk);
    if (retvalu == -1)
      edm::LogError("EvFDaqDirector") << "Error unlocking the fu.lock " << strerror(errno);

#ifdef DEBUG
    edm::LogDebug("EvFDaqDirector") << "Waited during lock -: " << locked_period << " seconds";
#endif

    if (fileStatus == noFile) {
      struct stat buf;
      //edm::LogInfo("EvFDaqDirector") << " looking for EoR file: " << getEoRFilePath().c_str();
      if (stat(getEoRFilePath().c_str(), &buf) == 0 || stat(bu_run_dir_.c_str(), &buf) != 0)
        fileStatus = runEnded;
      if (stopFileLS >= 0 && (int)ls > stopFileLS) {
        edm::LogInfo("EvFDaqDirector") << "Reached maximum lumisection set by hltd";
        fileStatus = runEnded;
      }
    }
    return fileStatus;
  }

  int EvFDaqDirector::getNFilesFromEoLS(std::string BUEoLSFile) {
    std::ifstream ij(BUEoLSFile);
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
      if (def.empty())
        readEolsDefinition_ = false;
      while (!def.empty()) {
        std::string fullpath;
        if (def.find('/') == 0)
          fullpath = def;
        else
          fullpath = bu_run_dir_ + '/' + def;
        struct stat buf;
        if (stat(fullpath.c_str(), &buf) == 0) {
          DataPointDefinition eolsDpd;
          std::string defLabel = "legend";
          DataPointDefinition::getDataPointDefinitionFor(fullpath, &eolsDpd, &defLabel);
          if (eolsDpd.getNames().empty()) {
            //try with "data" label if "legend" format is not used
            eolsDpd = DataPointDefinition();
            defLabel = "data";
            DataPointDefinition::getDataPointDefinitionFor(fullpath, &eolsDpd, &defLabel);
          }
          for (unsigned int i = 0; i < eolsDpd.getNames().size(); i++)
            if (eolsDpd.getNames().at(i) == "NFiles")
              eolsNFilesIndex_ = i;
          readEolsDefinition_ = false;
          break;
        }
        //check if we can still find definition
        if (def.size() <= 1 || def.find('/') == std::string::npos) {
          readEolsDefinition_ = false;
          break;
        }
        def = def.substr(def.find('/') + 1);
      }
    }

    if (dp.getData().size() > eolsNFilesIndex_)
      data = dp.getData()[eolsNFilesIndex_];
    else {
      edm::LogError("EvFDaqDirector") << " error reading number of files from BU JSON -: " << BUEoLSFile;
      return -1;
    }
    return std::stoi(data);
  }

  bool EvFDaqDirector::bumpFile(unsigned int& ls,
                                unsigned int& index,
                                std::string& nextFile,
                                uint32_t& fsize,
                                uint16_t& rawHeaderSize,
                                int maxLS,
                                bool& setExceptionState) {
    if (previousFileSize_ != 0) {
      if (!fms_) {
        fms_ = (FastMonitoringService*)(edm::Service<evf::MicroStateService>().operator->());
      }
      if (fms_)
        fms_->accumulateFileSize(ls, previousFileSize_);
      previousFileSize_ = 0;
    }
    nextFile = "";

    //reached limit
    if (maxLS >= 0 && ls > (unsigned int)maxLS)
      return false;

    struct stat buf;
    std::stringstream ss;

    // 1. Check suggested file
    std::string nextFileJson = getInputJsonFilePath(ls, index);
    if (stat(nextFileJson.c_str(), &buf) == 0) {
      fsize = previousFileSize_ = buf.st_size;
      nextFile = nextFileJson;
      return true;
    }
    // 2. No file -> lumi ended? (and how many?)
    else {
      // 3. No file -> check for standalone raw file
      std::string nextFileRaw = getRawFilePath(ls, index);
      if (stat(nextFileRaw.c_str(), &buf) == 0 && rawFileHasHeader(nextFileRaw, rawHeaderSize)) {
        fsize = previousFileSize_ = buf.st_size;
        nextFile = nextFileRaw;
        return true;
      }

      std::string BUEoLSFile = getEoLSFilePathOnBU(ls);

      if (stat(BUEoLSFile.c_str(), &buf) == 0) {
        // recheck that no raw file appeared in the meantime
        if (stat(nextFileJson.c_str(), &buf) == 0) {
          fsize = previousFileSize_ = buf.st_size;
          nextFile = nextFileJson;
          return true;
        }
        if (stat(nextFileRaw.c_str(), &buf) == 0 && rawFileHasHeader(nextFileRaw, rawHeaderSize)) {
          fsize = previousFileSize_ = buf.st_size;
          nextFile = nextFileRaw;
          return true;
        }

        int indexFilesInLS = getNFilesFromEoLS(BUEoLSFile);
        if (indexFilesInLS < 0)
          //parsing failed
          return false;
        else {
          //check index
          if ((int)index < indexFilesInLS) {
            //we have 2 files, and check for 1 failed... retry (2 will never be here)
            edm::LogError("EvFDaqDirector")
                << "Potential miss of index file in LS -: " << ls << ". Missing " << nextFile << " because "
                << indexFilesInLS - 1 << " is the highest index expected. Will not update fu.lock file";
            setExceptionState = true;
            return false;
          }
        }
        // this lumi ended, check for files
        ++ls;
        index = 0;

        //reached limit
        if (maxLS >= 0 && ls > (unsigned int)maxLS)
          return false;

        nextFileJson = getInputJsonFilePath(ls, 0);
        nextFileRaw = getRawFilePath(ls, 0);
        if (stat(nextFileJson.c_str(), &buf) == 0) {
          // a new file was found at new lumisection, index 0
          fsize = previousFileSize_ = buf.st_size;
          nextFile = nextFileJson;
          return true;
        }
        if (stat(nextFileRaw.c_str(), &buf) == 0 && rawFileHasHeader(nextFileRaw, rawHeaderSize)) {
          fsize = previousFileSize_ = buf.st_size;
          nextFile = nextFileRaw;
          return true;
        }
        return false;
      }
    }
    // no new file found
    return false;
  }

  void EvFDaqDirector::tryInitializeFuLockFile() {
    if (fu_rw_lock_stream == nullptr)
      edm::LogError("EvFDaqDirector") << "Error creating fu read/write lock stream " << strerror(errno);
    else {
      edm::LogInfo("EvFDaqDirector") << "Initializing FU LOCK FILE";
      unsigned int readLs = 1, readIndex = 0;
      fprintf(fu_rw_lock_stream, "%u %u", readLs, readIndex);
    }
  }

  void EvFDaqDirector::openFULockfileStream(bool create) {
    if (create) {
      fu_readwritelock_fd_ =
          open(fulockfile_.c_str(), O_RDWR | O_CREAT, S_IRWXU | S_IWGRP | S_IRGRP | S_IWOTH | S_IROTH);
      chmod(fulockfile_.c_str(), 0766);
    } else {
      fu_readwritelock_fd_ = open(fulockfile_.c_str(), O_RDWR, S_IRWXU);
    }
    if (fu_readwritelock_fd_ == -1)
      edm::LogError("EvFDaqDirector") << "problem with creating filedesc for fuwritelock -: " << fulockfile_
                                      << " create:" << create << " error:" << strerror(errno);
    else
      LogDebug("EvFDaqDirector") << "creating filedesc for fureadwritelock -: " << fu_readwritelock_fd_;

    fu_rw_lock_stream = fdopen(fu_readwritelock_fd_, "r+");
    if (fu_rw_lock_stream == nullptr)
      edm::LogError("EvFDaqDirector") << "problem with opening fuwritelock file stream -: " << strerror(errno);
  }

  void EvFDaqDirector::lockInitLock() { pthread_mutex_lock(&init_lock_); }

  void EvFDaqDirector::unlockInitLock() { pthread_mutex_unlock(&init_lock_); }

  void EvFDaqDirector::lockFULocal() {
    //fcntl(fulocal_rwlock_fd_, F_SETLKW, &fulocal_rw_flk);
    flock(fulocal_rwlock_fd_, LOCK_SH);
  }

  void EvFDaqDirector::unlockFULocal() {
    //fcntl(fulocal_rwlock_fd_, F_SETLKW, &fulocal_rw_fulk);
    flock(fulocal_rwlock_fd_, LOCK_UN);
  }

  void EvFDaqDirector::lockFULocal2() {
    //fcntl(fulocal_rwlock_fd2_, F_SETLKW, &fulocal_rw_flk2);
    flock(fulocal_rwlock_fd2_, LOCK_EX);
  }

  void EvFDaqDirector::unlockFULocal2() {
    //fcntl(fulocal_rwlock_fd2_, F_SETLKW, &fulocal_rw_fulk2);
    flock(fulocal_rwlock_fd2_, LOCK_UN);
  }

  void EvFDaqDirector::createBoLSFile(const uint32_t lumiSection, bool checkIfExists) const {
    //used for backpressure mechanisms and monitoring
    const std::string fuBoLS = getBoLSFilePathOnFU(lumiSection);
    struct stat buf;
    if (checkIfExists == false || stat(fuBoLS.c_str(), &buf) != 0) {
      int bol_fd = open(fuBoLS.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH);
      close(bol_fd);
    }
  }

  void EvFDaqDirector::createLumiSectionFiles(const uint32_t lumiSection,
                                              const uint32_t currentLumiSection,
                                              bool doCreateBoLS,
                                              bool doCreateEoLS) {
    if (currentLumiSection > 0) {
      const std::string fuEoLS = getEoLSFilePathOnFU(currentLumiSection);
      struct stat buf;
      bool found = (stat(fuEoLS.c_str(), &buf) == 0);
      if (!found) {
        if (doCreateEoLS) {
          int eol_fd =
              open(fuEoLS.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH);
          close(eol_fd);
        }
        if (doCreateBoLS)
          createBoLSFile(lumiSection, false);
      }
    } else if (doCreateBoLS) {
      createBoLSFile(lumiSection, true);  //needed for initial lumisection
    }
  }

  int EvFDaqDirector::parseFRDFileHeader(std::string const& rawSourcePath,
                                         int& rawFd,
                                         uint16_t& rawHeaderSize,
                                         uint16_t& rawDataType,
                                         uint32_t& lsFromHeader,
                                         int32_t& eventsFromHeader,
                                         int64_t& fileSizeFromHeader,
                                         bool requireHeader,
                                         bool retry,
                                         bool closeFile) {
    int infile;

    if ((infile = ::open(rawSourcePath.c_str(), O_RDONLY)) < 0) {
      if (retry) {
        edm::LogWarning("EvFDaqDirector")
            << "parseFRDFileHeader - failed to open input file -: " << rawSourcePath << " : " << strerror(errno);
        return parseFRDFileHeader(rawSourcePath,
                                  rawFd,
                                  rawHeaderSize,
                                  rawDataType,
                                  lsFromHeader,
                                  eventsFromHeader,
                                  fileSizeFromHeader,
                                  requireHeader,
                                  false,
                                  closeFile);
      } else {
        if ((infile = ::open(rawSourcePath.c_str(), O_RDONLY)) < 0) {
          edm::LogError("EvFDaqDirector")
              << "parseFRDFileHeader - failed to open input file -: " << rawSourcePath << " : " << strerror(errno);
          if (errno == ENOENT)
            return 1;  // error && file not found
          else
            return -1;
        }
      }
    }

    //v2 is the largest possible read
    char hdr[sizeof(FRDFileHeader_v2)];
    if (!checkFileRead(hdr, infile, sizeof(FRDFileHeaderIdentifier), rawSourcePath))
      return -1;

    FRDFileHeaderIdentifier* fileId = (FRDFileHeaderIdentifier*)hdr;
    uint16_t frd_version = getFRDFileHeaderVersion(fileId->id_, fileId->version_);

    if (frd_version == 0) {
      //no header (specific sequence not detected)
      if (requireHeader) {
        edm::LogError("EvFDaqDirector") << "no header or invalid version string found in:" << rawSourcePath;
        close(infile);
        return -1;
      } else {
        //no header, but valid file
        lseek(infile, 0, SEEK_SET);
        rawHeaderSize = 0;
        lsFromHeader = 0;
        eventsFromHeader = -1;
        fileSizeFromHeader = -1;
      }
    } else if (frd_version == 1) {
      //version 1 header
      if (!checkFileRead(hdr, infile, sizeof(FRDFileHeaderContent_v1), rawSourcePath))
        return -1;
      FRDFileHeaderContent_v1* fhContent = (FRDFileHeaderContent_v1*)hdr;
      uint32_t headerSizeRaw = fhContent->headerSize_;
      if (headerSizeRaw != sizeof(FRDFileHeader_v1)) {
        edm::LogError("EvFDaqDirector") << "inconsistent header size: " << rawSourcePath << " size: " << headerSizeRaw
                                        << " v:" << frd_version;
        close(infile);
        return -1;
      }
      //allow header size to exceed read size. Future header versions will not break this, but the size can change.
      rawDataType = 0;
      lsFromHeader = fhContent->lumiSection_;
      eventsFromHeader = (int32_t)fhContent->eventCount_;
      fileSizeFromHeader = (int64_t)fhContent->fileSize_;
      rawHeaderSize = fhContent->headerSize_;

    } else if (frd_version == 2) {
      //version 2 heade
      if (!checkFileRead(hdr, infile, sizeof(FRDFileHeaderContent_v2), rawSourcePath))
        return -1;
      FRDFileHeaderContent_v2* fhContent = (FRDFileHeaderContent_v2*)hdr;
      uint32_t headerSizeRaw = fhContent->headerSize_;
      if (headerSizeRaw != sizeof(FRDFileHeader_v2)) {
        edm::LogError("EvFDaqDirector") << "inconsistent header size: " << rawSourcePath << " size: " << headerSizeRaw
                                        << " v:" << frd_version;
        close(infile);
        return -1;
      }
      //allow header size to exceed read size. Future header versions will not break this, but the size can change.
      rawDataType = fhContent->dataType_;
      lsFromHeader = fhContent->lumiSection_;
      eventsFromHeader = (int32_t)fhContent->eventCount_;
      fileSizeFromHeader = (int64_t)fhContent->fileSize_;
      rawHeaderSize = fhContent->headerSize_;
    }

    if (closeFile) {
      close(infile);
      infile = -1;
    }

    rawFd = infile;
    return 0;  //OK
  }

  bool EvFDaqDirector::checkFileRead(char* buf, int infile, std::size_t buf_sz, std::string const& path) {
    ssize_t sz_read = ::read(infile, buf, buf_sz);
    if (sz_read < 0) {
      edm::LogError("EvFDaqDirector") << "rawFileHasHeader - unable to read " << path << " : " << strerror(errno);
      if (infile != -1)
        close(infile);
      return false;
    }
    if ((size_t)sz_read < buf_sz) {
      edm::LogError("EvFDaqDirector") << "rawFileHasHeader - file smaller than header: " << path;
      if (infile != -1)
        close(infile);
      return false;
    }
    return true;
  }

  bool EvFDaqDirector::rawFileHasHeader(std::string const& rawSourcePath, uint16_t& rawHeaderSize) {
    int infile;
    if ((infile = ::open(rawSourcePath.c_str(), O_RDONLY)) < 0) {
      edm::LogWarning("EvFDaqDirector") << "rawFileHasHeader - failed to open input file -: " << rawSourcePath << " : "
                                        << strerror(errno);
      return false;
    }
    //try to read FRD header size (v2 is the biggest, use read buffer of that size)
    char hdr[sizeof(FRDFileHeader_v2)];
    if (!checkFileRead(hdr, infile, sizeof(FRDFileHeaderIdentifier), rawSourcePath))
      return false;
    FRDFileHeaderIdentifier* fileId = (FRDFileHeaderIdentifier*)hdr;
    uint16_t frd_version = getFRDFileHeaderVersion(fileId->id_, fileId->version_);

    if (frd_version == 1) {
      if (!checkFileRead(hdr, infile, sizeof(FRDFileHeaderContent_v1), rawSourcePath))
        return false;
      FRDFileHeaderContent_v1* fhContent = (FRDFileHeaderContent_v1*)hdr;
      rawHeaderSize = fhContent->headerSize_;
      close(infile);
      return true;
    } else if (frd_version == 2) {
      if (!checkFileRead(hdr, infile, sizeof(FRDFileHeaderContent_v2), rawSourcePath))
        return false;
      FRDFileHeaderContent_v2* fhContent = (FRDFileHeaderContent_v2*)hdr;
      rawHeaderSize = fhContent->headerSize_;
      close(infile);
      return true;
    } else
      edm::LogError("EvFDaqDirector") << "rawFileHasHeader - unknown version: " << frd_version;

    close(infile);
    rawHeaderSize = 0;
    return false;
  }

  int EvFDaqDirector::grabNextJsonFromRaw(std::string const& rawSourcePath,
                                          int& rawFd,
                                          uint16_t& rawHeaderSize,
                                          int64_t& fileSizeFromHeader,
                                          bool& fileFound,
                                          uint32_t serverLS,
                                          bool closeFile,
                                          bool requireHeader) {
    fileFound = true;

    //take only first three tokens delimited by "_" in the renamed raw file name
    std::string jsonStem = std::filesystem::path(rawSourcePath).stem().string();
    size_t pos = 0, n_tokens = 0;
    while (n_tokens++ < 3 && (pos = jsonStem.find('_', pos + 1)) != std::string::npos) {
    }
    std::string reducedJsonStem = jsonStem.substr(0, pos);

    std::ostringstream fileNameWithPID;
    //should be ported to use fffnaming
    fileNameWithPID << reducedJsonStem << "_pid" << std::setfill('0') << std::setw(5) << pid_ << ".jsn";

    std::string jsonDestPath = baseRunDir() + "/" + fileNameWithPID.str();

    LogDebug("EvFDaqDirector") << "RAW parse -: " << rawSourcePath << " and JSON create " << jsonDestPath;

    //parse RAW file header if it exists
    uint32_t lsFromRaw;
    int32_t nbEventsWrittenRaw;
    int64_t fileSizeFromRaw;
    uint16_t rawDataType;
    auto ret = parseFRDFileHeader(rawSourcePath,
                                  rawFd,
                                  rawHeaderSize,
                                  rawDataType,
                                  lsFromRaw,
                                  nbEventsWrittenRaw,
                                  fileSizeFromRaw,
                                  requireHeader,
                                  true,
                                  closeFile);
    if (ret != 0) {
      if (ret == 1)
        fileFound = false;
      return -1;
    }

    int outfile;
    int oflag = O_CREAT | O_WRONLY | O_TRUNC | O_EXCL;  //file should not exist
    int omode = S_IWUSR | S_IRUSR | S_IWGRP | S_IRGRP | S_IWOTH | S_IROTH;
    if ((outfile = ::open(jsonDestPath.c_str(), oflag, omode)) < 0) {
      if (errno == EEXIST) {
        edm::LogError("EvFDaqDirector") << "grabNextJsonFromRaw - destination file already exists -: " << jsonDestPath
                                        << " : ";
        return -1;
      }
      edm::LogError("EvFDaqDirector") << "grabNextJsonFromRaw - failed to open output file -: " << jsonDestPath << " : "
                                      << strerror(errno);
      struct stat out_stat;
      if (stat(jsonDestPath.c_str(), &out_stat) == 0) {
        edm::LogWarning("EvFDaqDirector")
            << "grabNextJsonFromRaw - output file possibly got created with error, deleting and retry -: "
            << jsonDestPath;
        if (unlink(jsonDestPath.c_str()) == -1) {
          edm::LogWarning("EvFDaqDirector")
              << "grabNextJsonFromRaw - failed to remove -: " << jsonDestPath << " : " << strerror(errno);
        }
      }
      if ((outfile = ::open(jsonDestPath.c_str(), oflag, omode)) < 0) {
        edm::LogError("EvFDaqDirector") << "grabNextJsonFromRaw - failed to open output file (on retry) -: "
                                        << jsonDestPath << " : " << strerror(errno);
        return -1;
      }
    }
    //write JSON file (TODO: use jsoncpp)
    std::stringstream ss;
    ss << "{\"data\":[" << nbEventsWrittenRaw << "," << fileSizeFromRaw << ",\"" << rawSourcePath << "\"]}";
    std::string sstr = ss.str();

    if (::write(outfile, sstr.c_str(), sstr.size()) < 0) {
      edm::LogError("EvFDaqDirector") << "grabNextJsonFromRaw - failed to write to output file file -: " << jsonDestPath
                                      << " : " << strerror(errno);
      return -1;
    }
    close(outfile);
    if (serverLS && serverLS != lsFromRaw)
      edm::LogWarning("EvFDaqDirector") << "grabNextJsonFromRaw - mismatch in expected (server) LS " << serverLS
                                        << " and raw file header LS " << lsFromRaw;

    fileSizeFromHeader = fileSizeFromRaw;
    return nbEventsWrittenRaw;
  }

  int EvFDaqDirector::grabNextJsonFile(std::string const& jsonSourcePath,
                                       std::string const& rawSourcePath,
                                       int64_t& fileSizeFromJson,
                                       bool& fileFound) {
    fileFound = true;

    //should be ported to use fffnaming
    std::ostringstream fileNameWithPID;
    fileNameWithPID << std::filesystem::path(rawSourcePath).stem().string() << "_pid" << std::setfill('0')
                    << std::setw(5) << pid_ << ".jsn";

    // assemble json destination path
    std::string jsonDestPath = baseRunDir() + "/" + fileNameWithPID.str();

    LogDebug("EvFDaqDirector") << "JSON rename -: " << jsonSourcePath << " to " << jsonDestPath;

    int infile = -1, outfile = -1;

    if ((infile = ::open(jsonSourcePath.c_str(), O_RDONLY)) < 0) {
      edm::LogWarning("EvFDaqDirector") << "grabNextJsonFile - failed to open input file -: " << jsonSourcePath << " : "
                                        << strerror(errno);
      if ((infile = ::open(jsonSourcePath.c_str(), O_RDONLY)) < 0) {
        edm::LogError("EvFDaqDirector") << "grabNextJsonFile - failed to open input file (on retry) -: "
                                        << jsonSourcePath << " : " << strerror(errno);
        if (errno == ENOENT)
          fileFound = false;
        return -1;
      }
    }

    int oflag = O_CREAT | O_WRONLY | O_TRUNC | O_EXCL;  //file should not exist
    int omode = S_IWUSR | S_IRUSR | S_IWGRP | S_IRGRP | S_IWOTH | S_IROTH;
    if ((outfile = ::open(jsonDestPath.c_str(), oflag, omode)) < 0) {
      if (errno == EEXIST) {
        edm::LogError("EvFDaqDirector") << "grabNextJsonFile - destination file already exists -: " << jsonDestPath
                                        << " : ";
        ::close(infile);
        return -1;
      }
      edm::LogError("EvFDaqDirector") << "grabNextJsonFile - failed to open output file -: " << jsonDestPath << " : "
                                      << strerror(errno);
      struct stat out_stat;
      if (stat(jsonDestPath.c_str(), &out_stat) == 0) {
        edm::LogWarning("EvFDaqDirector")
            << "grabNextJsonFile - output file possibly got created with error, deleting and retry -: " << jsonDestPath;
        if (unlink(jsonDestPath.c_str()) == -1) {
          edm::LogWarning("EvFDaqDirector")
              << "grabNextJsonFile - failed to remove -: " << jsonDestPath << " : " << strerror(errno);
        }
      }
      if ((outfile = ::open(jsonDestPath.c_str(), oflag, omode)) < 0) {
        edm::LogError("EvFDaqDirector") << "grabNextJsonFile - failed to open output file (on retry) -: "
                                        << jsonDestPath << " : " << strerror(errno);
        ::close(infile);
        return -1;
      }
    }
    //copy contents
    const std::size_t buf_sz = 512;
    std::size_t tot_written = 0;
    std::unique_ptr<char[]> buf(new char[buf_sz]);

    ssize_t sz, sz_read = 1, sz_write;
    while (sz_read > 0 && (sz_read = ::read(infile, buf.get(), buf_sz)) > 0) {
      sz_write = 0;
      do {
        assert(sz_read - sz_write > 0);
        if ((sz = ::write(outfile, buf.get() + sz_write, sz_read - sz_write)) < 0) {
          sz_read = sz;  // cause read loop termination
          break;
        }
        assert(sz > 0);
        sz_write += sz;
        tot_written += sz;
      } while (sz_write < sz_read);
    }
    close(infile);
    close(outfile);

    if (tot_written > 0) {
      //leave file if it was empty for diagnosis
      if (unlink(jsonSourcePath.c_str()) == -1) {
        edm::LogError("EvFDaqDirector") << "grabNextJsonFile - failed to remove -: " << jsonSourcePath << " : "
                                        << strerror(errno);
        return -1;
      }
    } else {
      edm::LogError("EvFDaqDirector") << "grabNextJsonFile - failed to copy json file or file was empty -: "
                                      << jsonSourcePath;
      return -1;
    }

    Json::Value deserializeRoot;
    Json::Reader reader;

    std::string data;
    std::stringstream ss;
    bool result;
    try {
      if (tot_written <= buf_sz) {
        result = reader.parse(buf.get(), deserializeRoot);
      } else {
        //json will normally not be bigger than buf_sz bytes
        try {
          std::ifstream ij(jsonDestPath);
          ss << ij.rdbuf();
        } catch (std::filesystem::filesystem_error const& ex) {
          edm::LogError("EvFDaqDirector") << "grabNextJsonFile - FILESYSTEM ERROR CAUGHT -: " << ex.what();
          return -1;
        }
        result = reader.parse(ss.str(), deserializeRoot);
      }
      if (!result) {
        if (tot_written <= buf_sz)
          ss << buf.get();
        edm::LogError("EvFDaqDirector") << "Failed to deserialize JSON file -: " << jsonDestPath << "\nERROR:\n"
                                        << reader.getFormatedErrorMessages() << "CONTENT:\n"
                                        << ss.str() << ".";
        return -1;
      }

      //read BU JSON
      DataPoint dp;
      dp.deserialize(deserializeRoot);
      bool success = false;
      for (unsigned int i = 0; i < dpd_->getNames().size(); i++) {
        if (dpd_->getNames().at(i) == "NEvents")
          if (i < dp.getData().size()) {
            data = dp.getData()[i];
            success = true;
            break;
          }
      }
      if (!success) {
        if (!dp.getData().empty())
          data = dp.getData()[0];
        else {
          edm::LogError("EvFDaqDirector::grabNextJsonFile")
              << "grabNextJsonFile - "
              << " error reading number of events from BU JSON; No input value. data -: " << data;
          return -1;
        }
      }

      //try to read raw file size
      fileSizeFromJson = -1;
      for (unsigned int i = 0; i < dpd_->getNames().size(); i++) {
        if (dpd_->getNames().at(i) == "NBytes") {
          if (i < dp.getData().size()) {
            std::string dataSize = dp.getData()[i];
            try {
              fileSizeFromJson = std::stol(dataSize);
            } catch (const std::exception&) {
              //non-fatal currently, processing can continue without this value
              edm::LogWarning("EvFDaqDirector") << "grabNextJsonFile - error parsing number of Bytes from BU JSON. "
                                                << "Input value is -: " << dataSize;
            }
            break;
          }
        }
      }
      return std::stoi(data);
    } catch (const std::out_of_range& e) {
      edm::LogError("EvFDaqDirector") << "grabNextJsonFile - error parsing number of events from BU JSON. "
                                      << "Input value is -: " << data;
    } catch (const std::invalid_argument& e) {
      edm::LogError("EvFDaqDirector") << "grabNextJsonFile - argument error parsing events from BU JSON. "
                                      << "Input value is -: " << data;
    } catch (std::runtime_error const& e) {
      //Can be thrown by Json parser
      edm::LogError("EvFDaqDirector") << "grabNextJsonFile - std::runtime_error exception -: " << e.what();
    }

    catch (std::exception const& e) {
      edm::LogError("EvFDaqDirector") << "grabNextJsonFile - SOME OTHER EXCEPTION OCCURED! -: " << e.what();
    } catch (...) {
      //unknown exception
      edm::LogError("EvFDaqDirector") << "grabNextJsonFile - SOME OTHER EXCEPTION OCCURED!";
    }

    return -1;
  }

  int EvFDaqDirector::grabNextJsonFileAndUnlock(std::filesystem::path const& jsonSourcePath) {
    std::string data;
    try {
      // assemble json destination path
      std::filesystem::path jsonDestPath(baseRunDir());

      //should be ported to use fffnaming
      std::ostringstream fileNameWithPID;
      fileNameWithPID << jsonSourcePath.stem().string() << "_pid" << std::setfill('0') << std::setw(5) << getpid()
                      << ".jsn";
      jsonDestPath /= fileNameWithPID.str();

      LogDebug("EvFDaqDirector") << "JSON rename -: " << jsonSourcePath << " to " << jsonDestPath;
      try {
        std::filesystem::copy(jsonSourcePath, jsonDestPath);
      } catch (std::filesystem::filesystem_error const& ex) {
        // Input dir gone?
        edm::LogError("EvFDaqDirector") << "grabNextFile BOOST FILESYSTEM ERROR CAUGHT -: " << ex.what();
        //                                     << " Maybe the file is not yet visible by FU. Trying again in one second";
        sleep(1);
        std::filesystem::copy(jsonSourcePath, jsonDestPath);
      }
      unlockFULocal();

      try {
        //sometimes this fails but file gets deleted
        std::filesystem::remove(jsonSourcePath);
      } catch (std::filesystem::filesystem_error const& ex) {
        // Input dir gone?
        edm::LogError("EvFDaqDirector") << "grabNextFile BOOST FILESYSTEM ERROR CAUGHT -: " << ex.what();
      } catch (std::exception const& ex) {
        // Input dir gone?
        edm::LogError("EvFDaqDirector") << "grabNextFile std::exception CAUGHT -: " << ex.what();
      }

      std::ifstream ij(jsonDestPath);
      Json::Value deserializeRoot;
      Json::Reader reader;

      std::stringstream ss;
      ss << ij.rdbuf();
      if (!reader.parse(ss.str(), deserializeRoot)) {
        edm::LogError("EvFDaqDirector") << "grabNextFile Failed to deserialize JSON file -: " << jsonDestPath
                                        << "\nERROR:\n"
                                        << reader.getFormatedErrorMessages() << "CONTENT:\n"
                                        << ss.str() << ".";
        throw std::runtime_error("Cannot deserialize input JSON file");
      }

      //read BU JSON
      std::string data;
      DataPoint dp;
      dp.deserialize(deserializeRoot);
      bool success = false;
      for (unsigned int i = 0; i < dpd_->getNames().size(); i++) {
        if (dpd_->getNames().at(i) == "NEvents")
          if (i < dp.getData().size()) {
            data = dp.getData()[i];
            success = true;
          }
      }
      if (!success) {
        if (!dp.getData().empty())
          data = dp.getData()[0];
        else
          throw cms::Exception("EvFDaqDirector::grabNextJsonFileUnlock")
              << " error reading number of events from BU JSON -: No input value " << data;
      }
      return std::stoi(data);
    } catch (std::filesystem::filesystem_error const& ex) {
      // Input dir gone?
      unlockFULocal();
      edm::LogError("EvFDaqDirector") << "grabNextFile BOOST FILESYSTEM ERROR CAUGHT -: " << ex.what();
    } catch (std::runtime_error const& e) {
      // Another process grabbed the file and NFS did not register this
      unlockFULocal();
      edm::LogError("EvFDaqDirector") << "grabNextFile runtime Exception -: " << e.what();
    } catch (const std::out_of_range&) {
      edm::LogError("EvFDaqDirector") << "grabNextFile error parsing number of events from BU JSON. "
                                      << "Input value is -: " << data;
    } catch (const std::invalid_argument&) {
      edm::LogError("EvFDaqDirector") << "grabNextFile argument error parsing events from BU JSON. "
                                      << "Input value is -: " << data;
    } catch (std::exception const& e) {
      // BU run directory disappeared?
      unlockFULocal();
      edm::LogError("EvFDaqDirector") << "grabNextFile SOME OTHER EXCEPTION OCCURED!!!! -: " << e.what();
    }

    return -1;
  }

  EvFDaqDirector::FileStatus EvFDaqDirector::contactFileBroker(unsigned int& serverHttpStatus,
                                                               bool& serverError,
                                                               uint32_t& serverLS,
                                                               uint32_t& closedServerLS,
                                                               std::string& nextFileJson,
                                                               std::string& nextFileRaw,
                                                               bool& rawHeader,
                                                               int maxLS) {
    EvFDaqDirector::FileStatus fileStatus = noFile;
    serverError = false;

    boost::system::error_code ec;
    try {
      while (true) {
        //socket connect
        if (!fileBrokerKeepAlive_ || !socket_->is_open()) {
          boost::asio::connect(*socket_, *endpoint_iterator_, ec);

          if (ec) {
            edm::LogWarning("EvFDaqDirector") << "boost::asio::connect error -:" << ec;
            serverError = true;
            break;
          }
        }

        boost::asio::streambuf request;
        std::ostream request_stream(&request);
        std::string path = "/popfile?runnumber=" + run_nstring_ + "&pid=" + pid_;
        if (maxLS >= 0) {
          std::stringstream spath;
          spath << path << "&stopls=" << maxLS;
          path = spath.str();
          edm::LogWarning("EvFDaqDirector") << "Stop LS requested " << maxLS;
        }
        request_stream << "GET " << path << " HTTP/1.1\r\n";
        request_stream << "Host: " << fileBrokerHost_ << "\r\n";
        request_stream << "Accept: */*\r\n";
        request_stream << "Connection: keep-alive\r\n\r\n";

        boost::asio::write(*socket_, request, ec);
        if (ec) {
          if (fileBrokerKeepAlive_ && ec == boost::asio::error::connection_reset) {
            edm::LogInfo("EvFDaqDirector") << "reconnecting socket on received connection_reset";
            //we got disconnected, try to reconnect to the server before writing the request
            boost::asio::connect(*socket_, *endpoint_iterator_, ec);
            if (ec) {
              edm::LogWarning("EvFDaqDirector") << "boost::asio::connect error -:" << ec;
              serverError = true;
              break;
            }
            continue;
          }
          edm::LogWarning("EvFDaqDirector") << "boost::asio::write error -:" << ec;
          serverError = true;
          break;
        }

        boost::asio::streambuf response;
        boost::asio::read_until(*socket_, response, "\r\n", ec);
        if (ec) {
          edm::LogWarning("EvFDaqDirector") << "boost::asio::read_until error -:" << ec;
          serverError = true;
          break;
        }

        std::istream response_stream(&response);

        std::string http_version;
        response_stream >> http_version;

        response_stream >> serverHttpStatus;

        std::string status_message;
        std::getline(response_stream, status_message);
        if (!response_stream || http_version.substr(0, 5) != "HTTP/") {
          edm::LogWarning("EvFDaqDirector") << "Invalid server response";
          serverError = true;
          break;
        }
        if (serverHttpStatus != 200) {
          edm::LogWarning("EvFDaqDirector") << "Response returned with status code " << serverHttpStatus;
          serverError = true;
          break;
        }

        // Process the response headers.
        std::string header;
        while (std::getline(response_stream, header) && header != "\r") {
        }

        std::string fileInfo;
        std::map<std::string, std::string> serverMap;
        while (std::getline(response_stream, fileInfo) && fileInfo != "\r") {
          auto pos = fileInfo.find('=');
          if (pos == std::string::npos)
            continue;
          auto stitle = fileInfo.substr(0, pos);
          auto svalue = fileInfo.substr(pos + 1);
          serverMap[stitle] = svalue;
        }

        //check that response run number if correct
        auto server_version = serverMap.find("version");
        assert(server_version != serverMap.end());

        auto server_run = serverMap.find("runnumber");
        assert(server_run != serverMap.end());
        assert(run_nstring_ == server_run->second);

        auto server_state = serverMap.find("state");
        assert(server_state != serverMap.end());

        auto server_eols = serverMap.find("lasteols");
        assert(server_eols != serverMap.end());

        auto server_ls = serverMap.find("lumisection");

        int version_maj = 1;
        int version_min = 0;
        int version_rev = 0;
        {
          auto* s_ptr = server_version->second.c_str();
          if (!server_version->second.empty() && server_version->second[0] == '"')
            s_ptr++;
          auto res = sscanf(s_ptr, "%d.%d.%d", &version_maj, &version_min, &version_rev);
          if (res < 3) {
            res = sscanf(s_ptr, "%d.%d", &version_maj, &version_min);
            if (res < 2) {
              res = sscanf(s_ptr, "%d", &version_maj);
              if (res < 1) {
                //expecting at least 1 number (major version)
                edm::LogWarning("EvFDaqDirector") << "Can not parse server version " << server_version->second;
              }
            }
          }
        }

        closedServerLS = (uint64_t)std::max(0, atoi(server_eols->second.c_str()));
        if (server_ls != serverMap.end())
          serverLS = (uint64_t)std::max(1, atoi(server_ls->second.c_str()));
        else
          serverLS = closedServerLS + 1;

        std::string s_state = server_state->second;
        if (s_state == "STARTING")  //initial, always empty starting with LS 1
        {
          auto server_file = serverMap.find("file");
          assert(server_file == serverMap.end());  //no file with starting state
          fileStatus = noFile;
          edm::LogInfo("EvFDaqDirector") << "Got STARTING notification with last EOLS " << closedServerLS;
        } else if (s_state == "READY") {
          auto server_file = serverMap.find("file");
          if (server_file == serverMap.end()) {
            //can be returned by server if files from new LS already appeared but LS is not yet closed
            if (serverLS <= closedServerLS)
              serverLS = closedServerLS + 1;
            fileStatus = noFile;
            edm::LogInfo("EvFDaqDirector")
                << "Got READY notification with last EOLS " << closedServerLS << " and no new file";
          } else {
            std::string filestem;
            std::string fileprefix;
            auto server_fileprefix = serverMap.find("fileprefix");

            if (server_fileprefix != serverMap.end()) {
              auto pssize = server_fileprefix->second.size();
              if (pssize > 1 && server_fileprefix->second[0] == '"' && server_fileprefix->second[pssize - 1] == '"')
                fileprefix = server_fileprefix->second.substr(1, pssize - 2);
              else
                fileprefix = server_fileprefix->second;
            }

            //remove string literals
            auto ssize = server_file->second.size();
            if (ssize > 1 && server_file->second[0] == '"' && server_file->second[ssize - 1] == '"')
              filestem = server_file->second.substr(1, ssize - 2);
            else
              filestem = server_file->second;
            assert(!filestem.empty());
            if (version_maj > 1) {
              nextFileRaw = bu_run_dir_ + "/" + fileprefix + filestem + ".raw";  //filestem should be raw
              filestem = bu_run_dir_ + "/" + fileprefix + filestem;
              nextFileJson = "";
              rawHeader = true;
            } else {
              nextFileRaw = bu_run_dir_ + "/" + filestem + ".raw";  //raw files are not moved
              filestem = bu_run_dir_ + "/" + fileprefix + filestem;
              nextFileJson = filestem + ".jsn";
              rawHeader = false;
            }
            fileStatus = newFile;
            edm::LogInfo("EvFDaqDirector") << "Got READY notification with last EOLS " << closedServerLS << " new LS "
                                           << serverLS << " file:" << filestem;
          }
        } else if (s_state == "EOLS") {
          serverLS = closedServerLS + 1;
          edm::LogInfo("EvFDaqDirector") << "Got EOLS notification with last EOLS " << closedServerLS;
          fileStatus = noFile;
        } else if (s_state == "EOR") {
          //server_eor = serverMap.find("iseor");
          edm::LogInfo("EvFDaqDirector") << "Got EOR notification with last EOLS " << closedServerLS;
          fileStatus = runEnded;
        } else if (s_state == "NORUN") {
          auto err_msg = serverMap.find("errormessage");
          if (err_msg != serverMap.end())
            edm::LogWarning("EvFDaqDirector") << "Server NORUN -:" << server_state->second << " : " << err_msg->second;
          else
            edm::LogWarning("EvFDaqDirector") << "Server NORUN ";
          edm::LogWarning("EvFDaqDirector") << "executing run end";
          fileStatus = runEnded;
        } else if (s_state == "ERROR") {
          auto err_msg = serverMap.find("errormessage");
          if (err_msg != serverMap.end())
            edm::LogWarning("EvFDaqDirector") << "Server error -:" << server_state->second << " : " << err_msg->second;
          else
            edm::LogWarning("EvFDaqDirector") << "Server error -:" << server_state->second;
          fileStatus = noFile;
          serverError = true;
        } else {
          edm::LogWarning("EvFDaqDirector") << "Unknown Server state -:" << server_state->second;
          fileStatus = noFile;
          serverError = true;
        }

        // Read until EOF, writing data to output as we go.
        if (!fileBrokerKeepAlive_) {
          while (boost::asio::read(*socket_, response, boost::asio::transfer_at_least(1), ec)) {
          }
          if (ec != boost::asio::error::eof) {
            edm::LogWarning("EvFDaqDirector") << "boost::asio::read_until error -:" << ec;
            serverError = true;
          }
        }

        break;
      }

    } catch (std::exception const& e) {
      edm::LogWarning("EvFDaqDirector") << "Exception in socket handling";
      serverError = true;
    }

    if (!fileBrokerKeepAlive_ && socket_->is_open()) {
      socket_->shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
      if (ec) {
        edm::LogWarning("EvFDaqDirector") << "socket shutdown error -:" << ec;
      }
      socket_->close(ec);
      if (ec) {
        edm::LogWarning("EvFDaqDirector") << "socket close error -:" << ec;
      }
    }

    if (serverError) {
      if (socket_->is_open())
        socket_->close(ec);
      if (ec) {
        edm::LogWarning("EvFDaqDirector") << "socket close error -:" << ec;
      }
      fileStatus = noFile;
      sleep(1);  //back-off if error detected
    }

    return fileStatus;
  }

  EvFDaqDirector::FileStatus EvFDaqDirector::getNextFromFileBroker(const unsigned int currentLumiSection,
                                                                   unsigned int& ls,
                                                                   std::string& nextFileRaw,
                                                                   int& rawFd,
                                                                   uint16_t& rawHeaderSize,
                                                                   int32_t& serverEventsInNewFile,
                                                                   int64_t& fileSizeFromMetadata,
                                                                   uint64_t& thisLockWaitTimeUs,
                                                                   bool requireHeader) {
    EvFDaqDirector::FileStatus fileStatus = noFile;

    //int retval = -1;
    //int lock_attempts = 0;
    //long total_lock_attempts = 0;

    struct stat buf;
    int stopFileLS = -1;
    int stopFileCheck = stat(stopFilePath_.c_str(), &buf);
    int stopFilePidCheck = stat(stopFilePathPid_.c_str(), &buf);
    if (stopFileCheck == 0 || stopFilePidCheck == 0) {
      if (stopFileCheck == 0)
        stopFileLS = readLastLSEntry(stopFilePath_);
      else
        stopFileLS = 1;  //stop without drain if only pid is stopped
      if (!stop_ls_override_) {
        //if lumisection is higher than in stop file, should quit at next from current
        if (stopFileLS >= 0 && (int)ls >= stopFileLS)
          stopFileLS = stop_ls_override_ = ls;
      } else
        stopFileLS = stop_ls_override_;
      edm::LogWarning("EvFDaqDirector") << "Detected stop request from hltd. Ending run for this process after LS -: "
                                        << stopFileLS;
      //return runEnded;
    } else  //if file was removed before reaching stop condition, reset this
      stop_ls_override_ = 0;

    /* look for EoLS
    if (stat(getEoLSFilePathOnFU(currentLumiSection).c_str(),&buf)==0) {
      edm::LogWarning("EvFDaqDirector") << "Detected local EoLS for lumisection "<< currentLumiSection ; 
      ls++;
      return noFile;
    }
    */

    timeval ts_lockbegin;
    gettimeofday(&ts_lockbegin, nullptr);

    std::string nextFileJson;
    uint32_t serverLS, closedServerLS;
    unsigned int serverHttpStatus;
    bool serverError;

    //local lock to force index json and EoLS files to appear in order
    if (fileBrokerUseLocalLock_)
      lockFULocal();

    int maxLS = stopFileLS < 0 ? -1 : std::max(stopFileLS, (int)currentLumiSection);
    bool rawHeader = false;
    fileStatus = contactFileBroker(
        serverHttpStatus, serverError, serverLS, closedServerLS, nextFileJson, nextFileRaw, rawHeader, maxLS);

    if (serverError) {
      //do not update anything
      if (fileBrokerUseLocalLock_)
        unlockFULocal();
      return noFile;
    }

    //handle creation of BoLS files if lumisection has changed
    if (currentLumiSection == 0) {
      if (fileStatus == runEnded)
        createLumiSectionFiles(closedServerLS, 0, true, false);
      else
        createLumiSectionFiles(serverLS, 0, true, false);
    } else {
      if (closedServerLS >= currentLumiSection) {
        //only BoLS files
        for (uint32_t i = std::max(currentLumiSection, 1U); i <= closedServerLS; i++)
          createLumiSectionFiles(i + 1, i, true, false);
      }
    }

    bool fileFound = true;

    if (fileStatus == newFile) {
      if (rawHeader > 0)
        serverEventsInNewFile = grabNextJsonFromRaw(
            nextFileRaw, rawFd, rawHeaderSize, fileSizeFromMetadata, fileFound, serverLS, false, requireHeader);
      else
        serverEventsInNewFile = grabNextJsonFile(nextFileJson, nextFileRaw, fileSizeFromMetadata, fileFound);
    }
    //closing file in case of any error
    if (serverEventsInNewFile < 0 && rawFd != -1) {
      close(rawFd);
      rawFd = -1;
    }

    //can unlock because all files have been created locally
    if (fileBrokerUseLocalLock_)
      unlockFULocal();

    if (!fileFound) {
      //catch condition where directory got deleted
      fileStatus = noFile;
      struct stat buf;
      if (stat(bu_run_dir_.c_str(), &buf) != 0) {
        edm::LogWarning("EvFDaqDirector") << "BU run directory not found:" << bu_run_dir_;
        fileStatus = runEnded;
      }
    }

    //handle creation of EoLS files if lumisection has changed, this needs to be locked exclusively
    //so that EoLS files can not appear locally before index files
    if (currentLumiSection == 0) {
      lockFULocal2();
      if (fileStatus == runEnded) {
        createLumiSectionFiles(closedServerLS, 0, false, true);
        createLumiSectionFiles(serverLS, closedServerLS, false, true);  // +1
      } else {
        createLumiSectionFiles(serverLS, 0, false, true);
      }
      unlockFULocal2();
    } else {
      if (closedServerLS >= currentLumiSection) {
        //lock exclusive to create EoLS files
        lockFULocal2();
        for (uint32_t i = std::max(currentLumiSection, 1U); i <= closedServerLS; i++)
          createLumiSectionFiles(i + 1, i, false, true);
        unlockFULocal2();
      }
    }

    if (fileStatus == runEnded)
      ls = std::max(currentLumiSection, serverLS);
    else if (fileStatus == newFile) {
      assert(serverLS >= ls);
      ls = serverLS;
    } else if (fileStatus == noFile) {
      if (serverLS >= ls)
        ls = serverLS;
      else {
        edm::LogWarning("EvFDaqDirector") << "Server reported LS " << serverLS
                                          << " which is smaller than currently open LS " << ls << ". Ignoring response";
        sleep(1);
      }
    }

    return fileStatus;
  }

  void EvFDaqDirector::createRunOpendirMaybe() {
    // create open dir if not already there

    std::filesystem::path openPath = getRunOpenDirPath();
    if (!std::filesystem::is_directory(openPath)) {
      LogDebug("EvFDaqDirector") << "<open> FU dir not found. Creating... -:" << openPath.string();
      std::filesystem::create_directories(openPath);
    }
  }

  int EvFDaqDirector::readLastLSEntry(std::string const& file) {
    std::ifstream ij(file);
    Json::Value deserializeRoot;
    Json::Reader reader;

    if (!reader.parse(ij, deserializeRoot)) {
      edm::LogError("EvFDaqDirector") << "Cannot deserialize input JSON file -:" << file;
      return -1;
    }

    int ret = deserializeRoot.get("lastLS", "").asInt();
    return ret;
  }

  unsigned int EvFDaqDirector::getLumisectionToStart() const {
    std::string fileprefix = run_dir_ + "/" + run_string_ + "_ls";
    std::string fullpath;
    struct stat buf;
    unsigned int lscount = 1;
    do {
      std::stringstream ss;
      ss << fileprefix << std::setfill('0') << std::setw(4) << lscount << "_EoLS.jsn";
      fullpath = ss.str();
      lscount++;
    } while (stat(fullpath.c_str(), &buf) == 0);
    return lscount - 1;
  }

  //if transferSystem PSet is present in the menu, we require it to be complete and consistent for all specified streams
  void EvFDaqDirector::checkTransferSystemPSet(edm::ProcessContext const& pc) {
    if (transferSystemJson_)
      return;

    transferSystemJson_.reset(new Json::Value);
    edm::ParameterSet const& topPset = edm::getParameterSet(pc.parameterSetID());
    if (topPset.existsAs<edm::ParameterSet>("transferSystem", true)) {
      const edm::ParameterSet& tsPset(topPset.getParameterSet("transferSystem"));

      Json::Value destinationsVal(Json::arrayValue);
      std::vector<std::string> destinations = tsPset.getParameter<std::vector<std::string>>("destinations");
      for (auto& dest : destinations)
        destinationsVal.append(dest);
      (*transferSystemJson_)["destinations"] = destinationsVal;

      Json::Value modesVal(Json::arrayValue);
      std::vector<std::string> modes = tsPset.getParameter<std::vector<std::string>>("transferModes");
      for (auto& mode : modes)
        modesVal.append(mode);
      (*transferSystemJson_)["transferModes"] = modesVal;

      for (auto psKeyItr = tsPset.psetTable().begin(); psKeyItr != tsPset.psetTable().end(); ++psKeyItr) {
        if (psKeyItr->first != "destinations" && psKeyItr->first != "transferModes") {
          const edm::ParameterSet& streamDef = tsPset.getParameterSet(psKeyItr->first);
          Json::Value streamVal;
          for (auto& mode : modes) {
            //validation
            if (!streamDef.existsAs<std::vector<std::string>>(mode, true))
              throw cms::Exception("EvFDaqDirector")
                  << " Missing transfer system specification for -:" << psKeyItr->first << " (transferMode " << mode
                  << ")";
            std::vector<std::string> streamDestinations = streamDef.getParameter<std::vector<std::string>>(mode);

            Json::Value sDestsValue(Json::arrayValue);

            if (streamDestinations.empty())
              throw cms::Exception("EvFDaqDirector")
                  << " Missing transter system destination(s) for -: " << psKeyItr->first << ", mode:" << mode;

            for (auto& sdest : streamDestinations) {
              bool sDestValid = false;
              sDestsValue.append(sdest);
              for (auto& dest : destinations) {
                if (dest == sdest)
                  sDestValid = true;
              }
              if (!sDestValid)
                throw cms::Exception("EvFDaqDirector")
                    << " Invalid transter system destination specified for -: " << psKeyItr->first << ", mode:" << mode
                    << ", dest:" << sdest;
            }
            streamVal[mode] = sDestsValue;
          }
          (*transferSystemJson_)[psKeyItr->first] = streamVal;
        }
      }
    } else {
      if (requireTSPSet_)
        throw cms::Exception("EvFDaqDirector") << "transferSystem PSet not found";
    }
  }

  std::string EvFDaqDirector::getStreamDestinations(std::string const& stream) const {
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
    if (!requireTSPSet_ && (selectedTransferMode_.empty() || selectedTransferMode_ == "null")) {
      edm::LogWarning("EvFDaqDirector")
          << "Selected mode string is not provided as DaqDirector parameter."
          << "Switch on requireTSPSet parameter to enforce this requirement. Setting mode to empty string.";
      return std::string("Failsafe");
    }
    if (requireTSPSet_ && (selectedTransferMode_.empty() || selectedTransferMode_ == "null")) {
      throw cms::Exception("EvFDaqDirector") << "Selected mode string is not provided as DaqDirector parameter.";
    }
    //check if stream has properly listed transfer stream
    if (!transferSystemJson_->get(streamRequestName, "").isMember(selectedTransferMode_.c_str())) {
      std::stringstream msg;
      msg << "Selected transfer mode " << selectedTransferMode_ << " is not specified for stream " << streamRequestName;
      if (requireTSPSet_)
        throw cms::Exception("EvFDaqDirector") << msg.str();
      else
        edm::LogWarning("EvFDaqDirector") << msg.str() << " (permissive mode)";
      return std::string("Failsafe");
    }
    Json::Value destsVec = transferSystemJson_->get(streamRequestName, "").get(selectedTransferMode_, "");

    //flatten string json::Array into CSV std::string
    std::string ret;
    for (Json::Value::iterator it = destsVec.begin(); it != destsVec.end(); it++) {
      if (!ret.empty())
        ret += ",";
      ret += (*it).asString();
    }
    return ret;
  }

  void EvFDaqDirector::checkMergeTypePSet(edm::ProcessContext const& pc) {
    if (mergeTypePset_.empty())
      return;
    if (!mergeTypeMap_.empty())
      return;
    edm::ParameterSet const& topPset = edm::getParameterSet(pc.parameterSetID());
    if (topPset.existsAs<edm::ParameterSet>(mergeTypePset_, true)) {
      const edm::ParameterSet& tsPset(topPset.getParameterSet(mergeTypePset_));
      for (const std::string& pname : tsPset.getParameterNames()) {
        std::string streamType = tsPset.getParameter<std::string>(pname);
        tbb::concurrent_hash_map<std::string, std::string>::accessor ac;
        mergeTypeMap_.insert(ac, pname);
        ac->second = streamType;
        ac.release();
      }
    }
  }

  std::string EvFDaqDirector::getStreamMergeType(std::string const& stream, MergeType defaultType) {
    tbb::concurrent_hash_map<std::string, std::string>::const_accessor search_ac;
    if (mergeTypeMap_.find(search_ac, stream))
      return search_ac->second;

    edm::LogInfo("EvFDaqDirector") << " No merging type specified for stream " << stream << ". Using default value";
    std::string defaultName = MergeTypeNames_[defaultType];
    tbb::concurrent_hash_map<std::string, std::string>::accessor ac;
    mergeTypeMap_.insert(ac, stream);
    ac->second = defaultName;
    ac.release();
    return defaultName;
  }

  void EvFDaqDirector::createProcessingNotificationMaybe() const {
    std::string proc_flag = run_dir_ + "/processing";
    int proc_flag_fd = open(proc_flag.c_str(), O_RDWR | O_CREAT, S_IRWXU | S_IWGRP | S_IRGRP | S_IWOTH | S_IROTH);
    close(proc_flag_fd);
  }

  struct flock EvFDaqDirector::make_flock(short type, short whence, off_t start, off_t len, pid_t pid) {
#ifdef __APPLE__
    return {start, len, pid, type, whence};
#else
    return {type, whence, start, len, pid};
#endif
  }

  bool EvFDaqDirector::inputThrottled() {
    struct stat buf;
    return (stat(input_throttled_file_.c_str(), &buf) == 0);
  }

  bool EvFDaqDirector::lumisectionDiscarded(unsigned int ls) {
    struct stat buf;
    return (stat((discard_ls_filestem_ + std::to_string(ls)).c_str(), &buf) == 0);
  }

}  // namespace evf
