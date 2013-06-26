// $Id: ResourceMonitorCollection.cc,v 1.47 2011/11/16 14:32:22 mommsen Exp $
/// @file: ResourceMonitorCollection.cc

#include <stdio.h>
#include <string>
#include <sstream>
#include <iomanip>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <dirent.h>
#include <fnmatch.h>
#include <pwd.h>
#include <fstream>
#include <algorithm>

#include <boost/bind.hpp>
#include <boost/regex.hpp> 

#include "EventFilter/StorageManager/interface/AlarmHandler.h"
#include "EventFilter/StorageManager/interface/CurlInterface.h"
#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/ResourceMonitorCollection.h"
#include "EventFilter/StorageManager/interface/Utils.h"


namespace stor {
  
  ResourceMonitorCollection::ResourceMonitorCollection
  (
    const utils::Duration_t& updateInterval,
    AlarmHandlerPtr ah
  ) :
  MonitorCollection(updateInterval),
  updateInterval_(updateInterval),
  alarmHandler_(ah),
  numberOfCopyWorkers_(-1),
  numberOfInjectWorkers_(-1),
  nLogicalDisks_(0),
  latchedSataBeastStatus_(-1)
  {}
  
  
  void ResourceMonitorCollection::configureDisks(DiskWritingParams const& dwParams)
  {
    boost::mutex::scoped_lock sl(diskUsageListMutex_);
    
    dwParams_ = dwParams;
    
    nLogicalDisks_ = std::max(dwParams.nLogicalDisk_, 1);
    diskUsageList_.clear();
    diskUsageList_.reserve(nLogicalDisks_+dwParams.otherDiskPaths_.size()+1);
    
    for (unsigned int i=0; i<nLogicalDisks_; ++i) {
      
      std::ostringstream pathName;
      pathName << dwParams.filePath_;
      if( dwParams.nLogicalDisk_ > 0 ) {
        pathName << "/" << std::setfill('0') << std::setw(2) << i; 
      }
      addDisk(pathName.str());
    }
    addDisk(dwParams.dbFilePath_);
    
    if ( alarmParams_.isProductionSystem_ )
    {
      addOtherDisks();
    }
  }
  
  
  void ResourceMonitorCollection::addDisk(const std::string& pathname)
  {
    if ( pathname.empty() ) return;
    
    DiskUsagePtr diskUsage( new DiskUsage(pathname) );
    retrieveDiskSize(diskUsage);
    diskUsageList_.push_back(diskUsage);
  }
  
  
  void ResourceMonitorCollection::addOtherDisks()
  {
    for ( DiskWritingParams::OtherDiskPaths::const_iterator
            it = dwParams_.otherDiskPaths_.begin(),
            itEnd =  dwParams_.otherDiskPaths_.end();
          it != itEnd;
          ++it)
    {
      addDisk(*it);
    }
  }
  
  
  void ResourceMonitorCollection::configureResources
  (
    ResourceMonitorParams const& rmParams
  )
  {
    rmParams_ = rmParams;
  }
  
  
  void ResourceMonitorCollection::configureAlarms
  (
    AlarmParams const& alarmParams
  )
  {
    alarmParams_ = alarmParams;
  }
  
  
  void ResourceMonitorCollection::getStats(Stats& stats) const
  {
    getDiskStats(stats);
    
    stats.numberOfCopyWorkers = numberOfCopyWorkers_;
    stats.numberOfInjectWorkers = numberOfInjectWorkers_;
    
    stats.sataBeastStatus = latchedSataBeastStatus_;
  }
  
  
  void ResourceMonitorCollection::getDiskStats(Stats& stats) const
  {
    boost::mutex::scoped_lock sl(diskUsageListMutex_);
    
    stats.diskUsageStatsList.clear();
    stats.diskUsageStatsList.reserve(diskUsageList_.size());
    for ( DiskUsagePtrList::const_iterator it = diskUsageList_.begin(),
            itEnd = diskUsageList_.end();
          it != itEnd;
          ++it)
    {
      DiskUsageStatsPtr diskUsageStats(new DiskUsageStats);
      diskUsageStats->diskSize = (*it)->diskSize_;
      diskUsageStats->absDiskUsage = (*it)->absDiskUsage_;
      diskUsageStats->relDiskUsage = (*it)->relDiskUsage_;
      diskUsageStats->pathName = (*it)->pathName_;
      diskUsageStats->alarmState = (*it)->alarmState_;
      stats.diskUsageStatsList.push_back(diskUsageStats);
    }
  }
  
  
  void ResourceMonitorCollection::do_calculateStatistics()
  {
    calcDiskUsage();
    calcNumberOfCopyWorkers();
    calcNumberOfInjectWorkers();
    checkSataBeasts();
  }
  
  
  void ResourceMonitorCollection::do_reset()
  {
    numberOfCopyWorkers_ = -1;
    numberOfInjectWorkers_ = -1;
    latchedSataBeastStatus_ = -1;
    
    boost::mutex::scoped_lock sl(diskUsageListMutex_);
    for ( DiskUsagePtrList::const_iterator it = diskUsageList_.begin(),
            itEnd = diskUsageList_.end();
          it != itEnd;
          ++it)
    {
      if ( ! (*it)->retrievingDiskSize_ )
      {
        (*it)->diskSize_ = -1;
        (*it)->absDiskUsage_ = -1;
        (*it)->relDiskUsage_ = -1;
        (*it)->retVal_ = 0;
        (*it)->alarmState_ = AlarmHandler::OKAY;
      }
    }
  }
  
  
  void ResourceMonitorCollection::do_appendInfoSpaceItems(InfoSpaceItems& infoSpaceItems)
  {
    infoSpaceItems.push_back(std::make_pair("copyWorkers", &copyWorkers_));
    infoSpaceItems.push_back(std::make_pair("injectWorkers", &injectWorkers_));
    infoSpaceItems.push_back(std::make_pair("sataBeastStatus", &sataBeastStatus_));
    infoSpaceItems.push_back(std::make_pair("numberOfDisks", &numberOfDisks_));
    infoSpaceItems.push_back(std::make_pair("diskPaths", &diskPaths_));
    infoSpaceItems.push_back(std::make_pair("totalDiskSpace", &totalDiskSpace_));
    infoSpaceItems.push_back(std::make_pair("usedDiskSpace", &usedDiskSpace_));
  }
  
  
  void ResourceMonitorCollection::do_updateInfoSpaceItems()
  {
    Stats stats;
    getStats(stats);
    
    if (stats.numberOfCopyWorkers > 0)
      copyWorkers_ = static_cast<xdata::UnsignedInteger32>(stats.numberOfCopyWorkers);
    else
      copyWorkers_ = 0;

    if (stats.numberOfInjectWorkers > 0)
      injectWorkers_ = static_cast<xdata::UnsignedInteger32>(stats.numberOfInjectWorkers);
    else
      injectWorkers_ = 0;
    
    sataBeastStatus_ = stats.sataBeastStatus;
    numberOfDisks_ = nLogicalDisks_;
    
    diskPaths_.clear();
    totalDiskSpace_.clear();
    usedDiskSpace_.clear();
    
    diskPaths_.reserve(stats.diskUsageStatsList.size());
    totalDiskSpace_.reserve(stats.diskUsageStatsList.size());
    usedDiskSpace_.reserve(stats.diskUsageStatsList.size());
    
    for (DiskUsageStatsPtrList::const_iterator
           it = stats.diskUsageStatsList.begin(),
           itEnd = stats.diskUsageStatsList.end();
         it != itEnd;
         ++it)
    {
      diskPaths_.push_back(
        static_cast<xdata::String>( (*it)->pathName )
      );
      totalDiskSpace_.push_back(
        static_cast<xdata::UnsignedInteger32>(
          static_cast<unsigned int>( (*it)->diskSize * 1024 )
        )
      );
      usedDiskSpace_.push_back(
        static_cast<xdata::UnsignedInteger32>( 
          static_cast<unsigned int>( (*it)->absDiskUsage * 1024 )
        )
      );
    }
    
    calcDiskUsage();
  }
  
  
  void ResourceMonitorCollection::calcDiskUsage()
  {
    boost::mutex::scoped_lock sl(diskUsageListMutex_);
    
    for ( DiskUsagePtrList::iterator it = diskUsageList_.begin(),
            itEnd = diskUsageList_.end();
          it != itEnd;
          ++it)
    {
      retrieveDiskSize(*it);
    }
  }
  
  
  void ResourceMonitorCollection::retrieveDiskSize(DiskUsagePtr diskUsage)
  {
    if ( ! diskUsage->retrievingDiskSize_ )
      // don't start another thread if there's already one
    {
      boost::thread thread(
        boost::bind( &ResourceMonitorCollection::doStatFs, this, diskUsage)
      );
      if (
        ( ! thread.timed_join( boost::posix_time::milliseconds(500) ) )
        || (diskUsage->retVal_ != 0)
      )
      {
        emitDiskAlarm(diskUsage);
      }
      else
      {
        const unsigned int blksize = diskUsage->statfs_.f_bsize;
        diskUsage->diskSize_ =
          static_cast<double>(diskUsage->statfs_.f_blocks * blksize) / 1024 / 1024 / 1024;
        diskUsage->absDiskUsage_ =
          diskUsage->diskSize_ -
          static_cast<double>(diskUsage->statfs_.f_bavail * blksize) / 1024 / 1024 / 1024;
        diskUsage->relDiskUsage_ = (100 * (diskUsage->absDiskUsage_ / diskUsage->diskSize_));
        if ( diskUsage->relDiskUsage_ > dwParams_.highWaterMark_ )
        {
          emitDiskSpaceAlarm(diskUsage);
        }
        else if ( diskUsage->relDiskUsage_ < dwParams_.highWaterMark_*0.95 )
          // do not change alarm level if we are close to the high water mark
        {
          revokeDiskAlarm(diskUsage);
        }
      }
    }
  }
  
  
  void ResourceMonitorCollection::doStatFs(DiskUsagePtr diskUsage)
  {
    diskUsage->retrievingDiskSize_ = true;

    #if __APPLE__
    diskUsage->retVal_ = statfs(diskUsage->pathName_.c_str(), &(diskUsage->statfs_));
    #else
    diskUsage->retVal_ = statfs64(diskUsage->pathName_.c_str(), &(diskUsage->statfs_));
    #endif
    if (diskUsage->pathName_ == "/aSlowDiskForUnitTests") ::sleep(5);

    diskUsage->retrievingDiskSize_ = false;
  }
  
  
  void ResourceMonitorCollection::emitDiskAlarm(DiskUsagePtr diskUsage)
  {
    const std::string msg = "Cannot access " + diskUsage->pathName_ + ". Is it mounted?";

    diskUsage->diskSize_ = -1;
    diskUsage->absDiskUsage_ = -1;
    diskUsage->relDiskUsage_ = -1;

    if ( isImportantDisk(diskUsage->pathName_) )
    {
      diskUsage->alarmState_ = AlarmHandler::FATAL;
      XCEPT_DECLARE(stor::exception::DiskSpaceAlarm, ex, msg);
      alarmHandler_->moveToFailedState(ex);
    }
    else
    {    
      diskUsage->alarmState_ = AlarmHandler::ERROR;
      XCEPT_DECLARE(stor::exception::DiskSpaceAlarm, ex, msg);
      alarmHandler_->raiseAlarm(diskUsage->pathName_, diskUsage->alarmState_, ex);
    }
  }
  
  
  void ResourceMonitorCollection::emitDiskSpaceAlarm(DiskUsagePtr diskUsage)
  {
    if (
      isImportantDisk(diskUsage->pathName_) &&
      (diskUsage->relDiskUsage_ > dwParams_.failHighWaterMark_)
    )
    {
      diskUsage->alarmState_ = AlarmHandler::FATAL;
      XCEPT_DECLARE(stor::exception::DiskSpaceAlarm, ex, diskUsage->toString());
      alarmHandler_->moveToFailedState(ex);
    }
    else
    {    
      diskUsage->alarmState_ = AlarmHandler::WARNING;
      XCEPT_DECLARE(stor::exception::DiskSpaceAlarm, ex, diskUsage->toString());
      alarmHandler_->raiseAlarm(diskUsage->pathName_, diskUsage->alarmState_, ex);
    }
  }
  
  
  bool ResourceMonitorCollection::isImportantDisk(const std::string& pathName)
  {
    DiskWritingParams::OtherDiskPaths::const_iterator begin =
      dwParams_.otherDiskPaths_.begin();
    DiskWritingParams::OtherDiskPaths::const_iterator end =
      dwParams_.otherDiskPaths_.end();
    return ( std::find(begin, end, pathName) == end );
  }
  

  void ResourceMonitorCollection::revokeDiskAlarm(DiskUsagePtr diskUsage)
  {
    diskUsage->alarmState_ = AlarmHandler::OKAY;
    
    alarmHandler_->revokeAlarm(diskUsage->pathName_);
  }
  
  
  void ResourceMonitorCollection::calcNumberOfCopyWorkers()
  {
    struct passwd* passwd = getpwnam(rmParams_.copyWorkers_.user_.c_str());
    if (passwd)
    {
      numberOfCopyWorkers_ =
        getProcessCount(rmParams_.copyWorkers_.command_, passwd->pw_uid);
    }
    else
    {
      numberOfCopyWorkers_ = 0;
    }
    
    if ( alarmParams_.isProductionSystem_ && rmParams_.copyWorkers_.expectedCount_ >= 0 )
    {
      checkNumberOfCopyWorkers();
    }
  }
  
  
  void ResourceMonitorCollection::checkNumberOfCopyWorkers()
  {
    const std::string alarmName = "CopyWorkers";
    
    if ( numberOfCopyWorkers_ < rmParams_.copyWorkers_.expectedCount_ )
    {
      std::ostringstream msg;
      msg << "Expected " << rmParams_.copyWorkers_.expectedCount_ <<
        " running CopyWorkers, but found " <<
        numberOfCopyWorkers_ << ".";
      XCEPT_DECLARE(stor::exception::CopyWorkers, ex, msg.str());
      alarmHandler_->raiseAlarm(alarmName, AlarmHandler::WARNING, ex);
    }
    else
    {
      alarmHandler_->revokeAlarm(alarmName);
    }
  }
  
  
  void ResourceMonitorCollection::calcNumberOfInjectWorkers()
  {
    struct passwd* passwd = getpwnam(rmParams_.injectWorkers_.user_.c_str());
    if (passwd)
    {
      numberOfInjectWorkers_ = getProcessCount(rmParams_.injectWorkers_.command_, passwd->pw_uid);
    }
    else
    {
      numberOfInjectWorkers_ = 0;
    }
    
    if (
      alarmParams_.isProductionSystem_ &&
      rmParams_.injectWorkers_.expectedCount_ >= 0
    )
    {
      checkNumberOfInjectWorkers();
    }
  }
  
  
  void ResourceMonitorCollection::checkNumberOfInjectWorkers()
  {
    const std::string alarmName = "InjectWorkers";
    
    if ( numberOfInjectWorkers_ != rmParams_.injectWorkers_.expectedCount_ )
    {
      std::ostringstream msg;
      msg << "Expected " << rmParams_.injectWorkers_.expectedCount_ <<
        " running InjectWorkers, but found " <<
        numberOfInjectWorkers_ << ".";
      XCEPT_DECLARE(stor::exception::InjectWorkers, ex, msg.str());
      alarmHandler_->raiseAlarm(alarmName, AlarmHandler::WARNING, ex);
    }
    else
    {
      alarmHandler_->revokeAlarm(alarmName);
    }
  }
  
  
  void ResourceMonitorCollection::checkSataBeasts()
  {
    SATABeasts sataBeasts;
    if ( getSataBeasts(sataBeasts) )
    {
      for (
        SATABeasts::const_iterator it = sataBeasts.begin(),
          itEnd= sataBeasts.end();
        it != itEnd;
        ++it
      )
      {
        checkSataBeast(*it);
      }
    }
    else
    {
      latchedSataBeastStatus_ = -1;
    }
  }
  
  
  bool ResourceMonitorCollection::getSataBeasts(SATABeasts& sataBeasts)
  {
    if (! alarmParams_.isProductionSystem_) return false;
    
    std::ifstream in;
    in.open( "/proc/mounts" );
    
    if ( ! in.is_open() ) return false;
    
    std::string line;
    while( getline(in,line) )
    {
      size_t pos = line.find("sata");
      if ( pos != std::string::npos )
      {
        std::ostringstream host;
        host << "satab-c2c"
          << std::setw(2) << std::setfill('0')
          << line.substr(pos+4,1)
          << "-"
          << std::setw(2) << std::setfill('0')
          << line.substr(pos+5,1);
        sataBeasts.insert(host.str());
      }
    }
    return !sataBeasts.empty();
  }
  
  
  void ResourceMonitorCollection::checkSataBeast(const std::string& sataBeast)
  {
    if ( ! (checkSataDisks(sataBeast,"-00.cms") || checkSataDisks(sataBeast,"-10.cms")) )
    {
      XCEPT_DECLARE(stor::exception::SataBeast, ex, 
        "Failed to connect to SATA beast " + sataBeast);
      alarmHandler_->raiseAlarm(sataBeast, AlarmHandler::ERROR, ex);
      
      latchedSataBeastStatus_ = 99999;
    }
  }
  
  
  bool ResourceMonitorCollection::checkSataDisks
  (
    const std::string& sataBeast,
    const std::string& hostSuffix
  )
  {
    CurlInterfacePtr curlInterface = CurlInterface::getInterface();
    CurlInterface::Content content;
    
    // Do not try to connect if we have no user name
    if ( rmParams_.sataUser_.empty() ) return true;
    
    const CURLcode returnCode =
      curlInterface->getContent(
        "http://" + sataBeast + hostSuffix + "/status.asp",rmParams_.sataUser_,
        content
      );
    
    if (returnCode == CURLE_OK)
    {
      updateSataBeastStatus(sataBeast, std::string(&content[0]));
      return true;
    }
    else
    {
      std::ostringstream msg;
      msg << "Failed to connect to SATA controller "
        << sataBeast << hostSuffix 
        << ": " << std::string(&content[0]);
      XCEPT_DECLARE(stor::exception::SataBeast, ex, msg.str());
      alarmHandler_->notifySentinel(AlarmHandler::WARNING, ex);
      
      return false;
    }
  }
  
  
  void ResourceMonitorCollection::updateSataBeastStatus
  (
    const std::string& sataBeast,
    const std::string& content
  )
  {
    boost::regex failedEntry(">([^<]* has failed[^<]*)");
    boost::regex failedDisk("Hard disk([[:digit:]]+)");
    boost::regex failedController("RAID controller ([[:digit:]]+)");
    boost::match_results<std::string::const_iterator> matchedEntry, matchedCause;
    boost::match_flag_type flags = boost::match_default;
    
    std::string::const_iterator start = content.begin();
    std::string::const_iterator end = content.end();
    
    unsigned int newSataBeastStatus = 0;
    
    while( regex_search(start, end, matchedEntry, failedEntry, flags) )
    {
      std::string errorMsg = matchedEntry[1];
      XCEPT_DECLARE(stor::exception::SataBeast, ex, sataBeast+": "+errorMsg);
      alarmHandler_->raiseAlarm(sataBeast, AlarmHandler::ERROR, ex);
      
      // find what failed
      if ( regex_search(errorMsg, matchedCause, failedDisk) )
      {
        // Update the number of failed disks
        ++newSataBeastStatus;
      }
      else if ( regex_search(errorMsg, matchedCause, failedController) )
      {
        // Update the number of failed controllers
        newSataBeastStatus += 100;
      }
      else
      {
        // Unknown failure
        newSataBeastStatus += 1000;
      }
      
      // update search position:
      start = matchedEntry[0].second;
      // update flags:
      flags |= boost::match_prev_avail;
      flags |= boost::match_not_bob;
    }
    
    latchedSataBeastStatus_ = newSataBeastStatus;
    
    if (latchedSataBeastStatus_ == 0) // no more problems
      alarmHandler_->revokeAlarm(sataBeast);
    
  }
  
  
  namespace {
    int filter(const struct dirent *dir)
    {
      return !fnmatch("[1-9]*", dir->d_name, 0);
    }
    
    bool matchUid(const std::string& filename, const uid_t& uid)
    {
      struct stat filestat;
      int result = stat(filename.c_str(), &filestat);
      return (result == 0 && filestat.st_uid == uid);
    }
    
    bool isMaster(const char* pid)
    {
      // Adapted from procps::minimal::stat2proc
      char buf[800]; // about 40 fields, 64-bit decimal is about 20 chars
      int fd;
      int ppid = 0;
      std::ostringstream statfile;
      statfile << "/proc/" << pid << "/stat";
      snprintf(buf, 32, statfile.str().c_str(), pid);
      if ( (fd = open(buf, O_RDONLY, 0) ) == -1 ) return false;
      int num = read(fd, buf, sizeof buf - 1);
      if(num<80) return false;
      buf[num] = '\0';
      char* tmp = strrchr(buf, ')');  // split into "PID (cmd" and "<rest>"
      num = sscanf(tmp + 4,           // skip ') %c '
        "%d", &ppid);
      close(fd);
      return ( num == 1 && ppid == 1 ); // scan succeeded and parent pid is 1
    }
    
    bool grep(const std::string& cmdline, const std::string& name)
    {
      
      std::ifstream in;
      in.open( cmdline.c_str() );
      
      std::string line;
      if ( in.is_open() )
      {
        std::string tmp;
        while( getline(in,tmp,'\0') )
        {
          line.append(tmp);
          line.append(" ");
        }
        in.close();
      }

      return ( line.find(name) != std::string::npos );
    }
  }
  
  
  int ResourceMonitorCollection::getProcessCount
  (
    const std::string& processName,
    const int& uid
  )
  {
    int count(0);
    struct dirent **namelist;
    int n;
    
    #if __APPLE__
    return -1;
    #else 
    n = scandir("/proc", &namelist, filter, 0);
    #endif
    if (n < 0) return -1;
    
    while(n--)
    {
      std::ostringstream cmdline;
      cmdline << "/proc/" << namelist[n]->d_name << "/cmdline";
      
      if ( grep(cmdline.str(), processName) &&
        (uid < 0 || matchUid(cmdline.str(), uid)) &&
        isMaster(namelist[n]->d_name) )
      {
        ++count;
      }
      free(namelist[n]);
    }
    free(namelist);
    
    return count;
  }
  
  
  ResourceMonitorCollection::DiskUsage::DiskUsage(const std::string& path)
  : pathName_(path), absDiskUsage_(-1), relDiskUsage_(-1), diskSize_(-1),
    retrievingDiskSize_(false), alarmState_(AlarmHandler::OKAY), retVal_(0)
  {}
  
  
  std::string ResourceMonitorCollection::DiskUsage::toString()
  {
    std::ostringstream msg;
    msg << std::fixed << std::setprecision(1) <<
      "Disk space usage for " << pathName_ <<
      " is " << relDiskUsage_ << "% (" <<
      absDiskUsage_ << "GB of " <<
      diskSize_ << "GB).";
    return msg.str();
  }
  
} // namespace stor

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
