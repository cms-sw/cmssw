// $Id: ResourceMonitorCollection.cc,v 1.38 2010/10/19 06:15:35 mommsen Exp $
/// @file: ResourceMonitorCollection.cc

#include <stdio.h>
#include <string>
#include <sstream>
#include <iomanip>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <fcntl.h>
#include <dirent.h>
#include <fnmatch.h>
#include <pwd.h>
#include <fstream>
#include <algorithm>

#include <boost/bind.hpp>
#include <boost/regex.hpp> 

#include "EventFilter/StorageManager/interface/CurlInterface.h"
#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/ResourceMonitorCollection.h"
#include "EventFilter/StorageManager/interface/Utils.h"

using namespace stor;

ResourceMonitorCollection::ResourceMonitorCollection
(
  const utils::duration_t& updateInterval,
  boost::shared_ptr<AlarmHandler> ah
) :
MonitorCollection(updateInterval),
_updateInterval(updateInterval),
_alarmHandler(ah),
_numberOfCopyWorkers(-1),
_numberOfInjectWorkers(-1),
_nLogicalDisks(0),
_latchedSataBeastStatus(-1)
{}


void ResourceMonitorCollection::configureDisks(DiskWritingParams const& dwParams)
{
  boost::mutex::scoped_lock sl(_diskUsageListMutex);
  
  _dwParams = dwParams;

  _nLogicalDisks = std::max(dwParams._nLogicalDisk, 1);
  _diskUsageList.clear();
  _diskUsageList.reserve(_nLogicalDisks+dwParams._otherDiskPaths.size()+1);

  for (unsigned int i=0; i<_nLogicalDisks; ++i) {

    std::ostringstream pathName;
    pathName << dwParams._filePath;
    if( dwParams._nLogicalDisk > 0 ) {
      pathName << "/" << std::setfill('0') << std::setw(2) << i; 
    }
    addDisk(pathName.str());
  }
  addDisk(dwParams._dbFilePath);

  if ( _alarmParams._isProductionSystem )
  {
    addOtherDisks();
  }
}


void ResourceMonitorCollection::addDisk(const std::string& pathname)
{
  if ( pathname.empty() ) return;

  DiskUsagePtr diskUsage( new DiskUsage() );
  diskUsage->pathName = pathname;
  retrieveDiskSize(diskUsage);
  _diskUsageList.push_back(diskUsage);
}


void ResourceMonitorCollection::addOtherDisks()
{
  for ( DiskWritingParams::OtherDiskPaths::const_iterator
          it = _dwParams._otherDiskPaths.begin(),
          itEnd =  _dwParams._otherDiskPaths.end();
        it != itEnd;
        ++it)
  {
    addDisk(*it);
  }
}


void ResourceMonitorCollection::configureResources(ResourceMonitorParams const& rmParams)
{
  _rmParams = rmParams;
}


void ResourceMonitorCollection::configureAlarms(AlarmParams const& alarmParams)
{
  _alarmParams = alarmParams;
}


void ResourceMonitorCollection::getStats(Stats& stats) const
{
  getDiskStats(stats);

  stats.numberOfCopyWorkers = _numberOfCopyWorkers;
  stats.numberOfInjectWorkers = _numberOfInjectWorkers;

  stats.sataBeastStatus = _latchedSataBeastStatus;
}


void ResourceMonitorCollection::getDiskStats(Stats& stats) const
{
  boost::mutex::scoped_lock sl(_diskUsageListMutex);

  stats.diskUsageStatsList.clear();
  stats.diskUsageStatsList.reserve(_diskUsageList.size());
  for ( DiskUsagePtrList::const_iterator it = _diskUsageList.begin(),
          itEnd = _diskUsageList.end();
        it != itEnd;
        ++it)
  {
    DiskUsageStatsPtr diskUsageStats(new DiskUsageStats);
    diskUsageStats->diskSize = (*it)->diskSize;
    diskUsageStats->absDiskUsage = (*it)->absDiskUsage;
    diskUsageStats->relDiskUsage = (*it)->relDiskUsage;
    diskUsageStats->pathName = (*it)->pathName;
    diskUsageStats->alarmState = (*it)->alarmState;
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
  _numberOfCopyWorkers = -1;
  _numberOfInjectWorkers = -1;
  _latchedSataBeastStatus = -1;

  boost::mutex::scoped_lock sl(_diskUsageListMutex);
  for ( DiskUsagePtrList::const_iterator it = _diskUsageList.begin(),
          itEnd = _diskUsageList.end();
        it != itEnd;
        ++it)
  {
    (*it)->absDiskUsage = -1;
    (*it)->relDiskUsage = -1;
    (*it)->alarmState = AlarmHandler::OKAY;
  }
}


void ResourceMonitorCollection::do_appendInfoSpaceItems(InfoSpaceItems& infoSpaceItems)
{
  infoSpaceItems.push_back(std::make_pair("copyWorkers", &_copyWorkers));
  infoSpaceItems.push_back(std::make_pair("injectWorkers", &_injectWorkers));
  infoSpaceItems.push_back(std::make_pair("sataBeastStatus", &_sataBeastStatus));
  infoSpaceItems.push_back(std::make_pair("numberOfDisks", &_numberOfDisks));
  infoSpaceItems.push_back(std::make_pair("diskPaths", &_diskPaths));
  infoSpaceItems.push_back(std::make_pair("totalDiskSpace", &_totalDiskSpace));
  infoSpaceItems.push_back(std::make_pair("usedDiskSpace", &_usedDiskSpace));
}


void ResourceMonitorCollection::do_updateInfoSpaceItems()
{
  Stats stats;
  getStats(stats);

  _copyWorkers = static_cast<xdata::UnsignedInteger32>(stats.numberOfCopyWorkers);
  _injectWorkers = static_cast<xdata::UnsignedInteger32>(stats.numberOfInjectWorkers);

  _sataBeastStatus = stats.sataBeastStatus;
  _numberOfDisks = _nLogicalDisks;

  _diskPaths.clear();
  _totalDiskSpace.clear();
  _usedDiskSpace.clear();

  _diskPaths.reserve(stats.diskUsageStatsList.size());
  _totalDiskSpace.reserve(stats.diskUsageStatsList.size());
  _usedDiskSpace.reserve(stats.diskUsageStatsList.size());

  for (DiskUsageStatsPtrList::const_iterator
         it = stats.diskUsageStatsList.begin(),
         itEnd = stats.diskUsageStatsList.end();
       it != itEnd;
       ++it)
  {
    _diskPaths.push_back(
      static_cast<xdata::String>( (*it)->pathName )
    );
    _totalDiskSpace.push_back(
      static_cast<xdata::UnsignedInteger32>(
        static_cast<unsigned int>( (*it)->diskSize * 1024 )
      )
    );
    _usedDiskSpace.push_back(
      static_cast<xdata::UnsignedInteger32>( 
        static_cast<unsigned int>( (*it)->absDiskUsage * 1024 )
      )
    );
  }

  calcDiskUsage();
}


void ResourceMonitorCollection::calcDiskUsage()
{
  boost::mutex::scoped_lock sl(_diskUsageListMutex);

  for ( DiskUsagePtrList::iterator it = _diskUsageList.begin(),
          itEnd = _diskUsageList.end();
        it != itEnd;
        ++it)
  {
    retrieveDiskSize(*it);
  }
}

void ResourceMonitorCollection::retrieveDiskSize(DiskUsagePtr diskUsage)
{
  struct statfs64 buf;
  int retVal = statfs64(diskUsage->pathName.c_str(), &buf);
  if(retVal==0) {
    unsigned int blksize = buf.f_bsize;
    diskUsage->diskSize = static_cast<double>(buf.f_blocks * blksize) / 1024 / 1024 / 1024;
    diskUsage->absDiskUsage =
      diskUsage->diskSize -
      static_cast<double>(buf.f_bavail * blksize) / 1024 / 1024 / 1024;
    diskUsage->relDiskUsage = (100 * (diskUsage->absDiskUsage / diskUsage->diskSize));
    if ( diskUsage->relDiskUsage > _dwParams._highWaterMark )
    {
      emitDiskSpaceAlarm(diskUsage);
    }
    else if ( diskUsage->relDiskUsage < _dwParams._highWaterMark*0.95 )
      // do not change alarm level if we are close to the high water mark
    {
      revokeDiskAlarm(diskUsage);
    }
  }
  else
  {
    emitDiskAlarm(diskUsage, errno);
    diskUsage->diskSize = -1;
    diskUsage->absDiskUsage = -1;
    diskUsage->relDiskUsage = -1;
  }
}


void ResourceMonitorCollection::emitDiskAlarm(DiskUsagePtr diskUsage, error_t e)
// do NOT use errno here
{
  std::string msg;

  switch(e)
  {
    case ENOENT :
      diskUsage->alarmState = AlarmHandler::ERROR;
      msg = "Cannot access " + diskUsage->pathName + ". Is it mounted?";
      break;

    default :
      diskUsage->alarmState = AlarmHandler::WARNING;
      msg = "Failed to retrieve disk space information for " + diskUsage->pathName + ":"
        + strerror(e);
  }
  
  XCEPT_DECLARE(stor::exception::DiskSpaceAlarm, ex, msg);
  _alarmHandler->notifySentinel(diskUsage->alarmState, ex);
}


void ResourceMonitorCollection::emitDiskSpaceAlarm(DiskUsagePtr diskUsage)
{
  if ( diskUsage->relDiskUsage > _dwParams._failHighWaterMark )
  {
    failIfImportantDisk(diskUsage);
  }

  diskUsage->alarmState = AlarmHandler::WARNING;

  XCEPT_DECLARE(stor::exception::DiskSpaceAlarm, ex, diskUsage->toString());
  _alarmHandler->raiseAlarm(diskUsage->pathName, diskUsage->alarmState, ex);
}


void ResourceMonitorCollection::failIfImportantDisk(DiskUsagePtr diskUsage)
{
  // do not fail if the disk is one of the other disks
  DiskWritingParams::OtherDiskPaths::const_iterator begin = _dwParams._otherDiskPaths.begin();
  DiskWritingParams::OtherDiskPaths::const_iterator end = _dwParams._otherDiskPaths.end();
  if ( std::find(begin, end, diskUsage->pathName) != end ) return;
  
  diskUsage->alarmState = AlarmHandler::FATAL;
  XCEPT_RAISE(stor::exception::DiskSpaceAlarm, diskUsage->toString());
}


void ResourceMonitorCollection::revokeDiskAlarm(DiskUsagePtr diskUsage)
{
  diskUsage->alarmState = AlarmHandler::OKAY;

  _alarmHandler->revokeAlarm(diskUsage->pathName);
}


void ResourceMonitorCollection::calcNumberOfCopyWorkers()
{
  struct passwd* passwd = getpwnam(_rmParams._copyWorkers._user.c_str());
  if (passwd)
  {
    _numberOfCopyWorkers = getProcessCount(_rmParams._copyWorkers._command, passwd->pw_uid);
  }
  else
  {
    _numberOfCopyWorkers = 0;
  }

  if ( _alarmParams._isProductionSystem && _rmParams._copyWorkers._expectedCount >= 0 )
  {
    checkNumberOfCopyWorkers();
  }
}


void ResourceMonitorCollection::checkNumberOfCopyWorkers()
{
  const std::string alarmName = "CopyWorkers";

  if ( _numberOfCopyWorkers < _rmParams._copyWorkers._expectedCount )
  {
    std::ostringstream msg;
    msg << "Expected " << _rmParams._copyWorkers._expectedCount <<
      " running CopyWorkers, but found " <<
      _numberOfCopyWorkers << ".";
    XCEPT_DECLARE(stor::exception::CopyWorkers, ex, msg.str());
    _alarmHandler->raiseAlarm(alarmName, AlarmHandler::WARNING, ex);
  }
  else
  {
    _alarmHandler->revokeAlarm(alarmName);
  }
}


void ResourceMonitorCollection::calcNumberOfInjectWorkers()
{
  struct passwd* passwd = getpwnam(_rmParams._injectWorkers._user.c_str());
  if (passwd)
  {
    _numberOfInjectWorkers = getProcessCount(_rmParams._injectWorkers._command, passwd->pw_uid);
  }
  else
  {
    _numberOfInjectWorkers = 0;
  }

  if ( _alarmParams._isProductionSystem && _rmParams._injectWorkers._expectedCount >= 0 )
  {
    checkNumberOfInjectWorkers();
  }
}


void ResourceMonitorCollection::checkNumberOfInjectWorkers()
{
  const std::string alarmName = "InjectWorkers";

  if ( _numberOfInjectWorkers != _rmParams._injectWorkers._expectedCount )
  {
    std::ostringstream msg;
    msg << "Expected " << _rmParams._injectWorkers._expectedCount <<
      " running InjectWorkers, but found " <<
      _numberOfInjectWorkers << ".";
    XCEPT_DECLARE(stor::exception::InjectWorkers, ex, msg.str());
    _alarmHandler->raiseAlarm(alarmName, AlarmHandler::WARNING, ex);
  }
  else
  {
    _alarmHandler->revokeAlarm(alarmName);
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
    _latchedSataBeastStatus = -1;
  }
}


bool ResourceMonitorCollection::getSataBeasts(SATABeasts& sataBeasts)
{
  if (! _alarmParams._isProductionSystem) return false;

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
    _alarmHandler->raiseAlarm(sataBeast, AlarmHandler::ERROR, ex);

    _latchedSataBeastStatus = 99999;
  }
}


bool ResourceMonitorCollection::checkSataDisks
(
  const std::string& sataBeast,
  const std::string& hostSuffix
)
{
  stor::CurlInterface curlInterface;
  std::string content;

  // Do not try to connect if we have no user name
  if ( _rmParams._sataUser.empty() ) return true;
  
  const CURLcode returnCode =
    curlInterface.getContent(
      "http://" + sataBeast + hostSuffix + "/status.asp",_rmParams._sataUser, content
    );
  
  if (returnCode == CURLE_OK)
  {
    updateSataBeastStatus(sataBeast, content);
    return true;
  }
  else
  {
    std::ostringstream msg;
    msg << "Failed to connect to SATA controller "
      << sataBeast << hostSuffix 
      << " with user name '" << _rmParams._sataUser
      << "': " << content;
    XCEPT_DECLARE(stor::exception::SataBeast, ex, msg.str());
    _alarmHandler->notifySentinel(AlarmHandler::WARNING, ex);

    return false;
  }
}

void ResourceMonitorCollection::updateSataBeastStatus(
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
    _alarmHandler->raiseAlarm(sataBeast, AlarmHandler::ERROR, ex);

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

  _latchedSataBeastStatus = newSataBeastStatus;

  if (_latchedSataBeastStatus == 0) // no more problems
    _alarmHandler->revokeAlarm(sataBeast);

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


int ResourceMonitorCollection::getProcessCount(const std::string& processName, const int& uid)
{
  
  int count(0);
  struct dirent **namelist;
  int n;
  
  n = scandir("/proc", &namelist, filter, 0);
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


std::string ResourceMonitorCollection::DiskUsage::toString()
{
  std::ostringstream msg;
  msg << std::fixed << std::setprecision(1) <<
    "Disk space usage for " << pathName <<
    " is " << relDiskUsage << "% (" <<
    absDiskUsage << "GB of " <<
    diskSize << "GB).";
  return msg.str();
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
