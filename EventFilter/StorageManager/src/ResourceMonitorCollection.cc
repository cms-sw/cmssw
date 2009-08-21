// $Id: ResourceMonitorCollection.cc,v 1.8 2009/08/20 13:46:20 mommsen Exp $
/// @file: ResourceMonitorCollection.cc

#include <string>
#include <sstream>
#include <iomanip>
#include <sys/statfs.h>
#include <dirent.h>
#include <fnmatch.h>
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
_poolUsage(updateInterval, 10),
_numberOfCopyWorkers(updateInterval, 10),
_numberOfInjectWorkers(updateInterval, 10),
_alarmHandler(ah),
_pool(0),
_progressMarker( "unused" )
{}


void ResourceMonitorCollection::configureDisks(DiskWritingParams const& dwParams)
{
  boost::mutex::scoped_lock sl(_diskUsageListMutex);

  _highWaterMark = dwParams._highWaterMark;
  _sataUser = dwParams._sataUser;

  int nLogicalDisk = dwParams._nLogicalDisk;
  unsigned int nD = nLogicalDisk ? nLogicalDisk : 1;
  _diskUsageList.clear();
  _diskUsageList.reserve(nD);

  for (unsigned int i=0; i<nD; ++i) {

    DiskUsagePtr diskUsage( new DiskUsage(_updateInterval) );
    diskUsage->pathName = dwParams._filePath;
    if(nLogicalDisk>0) {
      std::ostringstream oss;
      oss << "/" << std::setfill('0') << std::setw(2) << i; 
      diskUsage->pathName += oss.str();
    }
    diskUsage->alarmName = "stor-diskspace-" + diskUsage->pathName;

    diskUsage->diskSize = 0;
    struct statfs64 buf;
    int retVal = statfs64(diskUsage->pathName.c_str(), &buf);
    if(retVal==0) {
      unsigned int blksize = buf.f_bsize;
      diskUsage->diskSize = buf.f_blocks * blksize / 1024 / 1024 /1024;
    }
    _diskUsageList.push_back(diskUsage);
  }
}

void ResourceMonitorCollection::setMemoryPoolPointer(toolbox::mem::Pool* pool)
{
  if ( ! _pool)
    _pool = pool;
}


void ResourceMonitorCollection::getStats(Stats& stats) const
{
  getDiskStats(stats);

  _poolUsage.getStats(stats.poolUsageStats);
  _numberOfCopyWorkers.getStats(stats.numberOfCopyWorkersStats);
  _numberOfInjectWorkers.getStats(stats.numberOfInjectWorkersStats);

  stats.sataBeastStatus = _sataBeastStatus;
}


void  ResourceMonitorCollection::getDiskStats(Stats& stats) const
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
    (*it)->absDiskUsage.getStats(diskUsageStats->absDiskUsageStats);
    (*it)->relDiskUsage.getStats(diskUsageStats->relDiskUsageStats);
    diskUsageStats->diskSize = (*it)->diskSize;
    diskUsageStats->pathName = (*it)->pathName;
    diskUsageStats->alarmState = (*it)->alarmState;
    stats.diskUsageStatsList.push_back(diskUsageStats);
  }
}


void ResourceMonitorCollection::do_calculateStatistics()
{
  calcPoolUsage();
  calcDiskUsage();
  calcNumberOfWorkers();
  checkSataBeasts();
}


void ResourceMonitorCollection::do_reset()
{
  _poolUsage.reset();
  _numberOfCopyWorkers.reset();
  _numberOfInjectWorkers.reset();
  _sataBeastStatus = 0;

  boost::mutex::scoped_lock sl(_diskUsageListMutex);
  for ( DiskUsagePtrList::const_iterator it = _diskUsageList.begin(),
          itEnd = _diskUsageList.end();
        it != itEnd;
        ++it)
  {
    (*it)->absDiskUsage.reset();
    (*it)->relDiskUsage.reset();
    (*it)->alarmState = AlarmHandler::OKAY;
  }
}


void ResourceMonitorCollection::do_appendInfoSpaceItems(InfoSpaceItems& infoSpaceItems)
{
  infoSpaceItems.push_back(std::make_pair("progressMarker", &_progressMarker));
}


void ResourceMonitorCollection::do_updateInfoSpaceItems()
{
  //nothing to do: the progressMarker does not change its value
}


void ResourceMonitorCollection::calcPoolUsage()
{
  if (_pool)
  {
    try {
      _pool->lock();
      _poolUsage.addSample( _pool->getMemoryUsage().getUsed() );
      _pool->unlock();
    }
    catch (...)
    {
      _pool->unlock();
    }
  }
  _poolUsage.calculateStatistics();
}


void ResourceMonitorCollection::calcDiskUsage()
{
  boost::mutex::scoped_lock sl(_diskUsageListMutex);

  for ( DiskUsagePtrList::iterator it = _diskUsageList.begin(),
          itEnd = _diskUsageList.end();
        it != itEnd;
        ++it)
  {
    struct statfs64 buf;
    int retVal = statfs64((*it)->pathName.c_str(), &buf);
    if(retVal==0) {
      unsigned int blksize = buf.f_bsize;
      double absused = 
        (*it)->diskSize -
        buf.f_bavail  * blksize / 1024 / 1024 /1024;
      double relused = (100 * (absused / (*it)->diskSize)); 
      (*it)->absDiskUsage.addSample(absused);
      (*it)->absDiskUsage.calculateStatistics();
      (*it)->relDiskUsage.addSample(relused);
      (*it)->relDiskUsage.calculateStatistics();
      if (relused > _highWaterMark*100)
      {
        emitDiskUsageAlarm(*it);
      }
      else if (relused < _highWaterMark*95)
        // do not change alarm level if we are close to the high water mark
      {
        revokeDiskUsageAlarm(*it);
      }
    }
  }
}


void ResourceMonitorCollection::emitDiskUsageAlarm(DiskUsagePtr diskUsage)
{
  diskUsage->alarmState = AlarmHandler::WARNING;

  MonitoredQuantity::Stats relUsageStats, absUsageStats;
  diskUsage->relDiskUsage.getStats(relUsageStats);
  diskUsage->absDiskUsage.getStats(absUsageStats);

  std::ostringstream msg;
  msg << std::fixed << std::setprecision(1) <<
    "Disk space usage for " << diskUsage->pathName <<
    " is " << relUsageStats.getLastSampleValue() << "% (" <<
    absUsageStats.getLastSampleValue() << "GB of " <<
    diskUsage->diskSize << "GB).";

  XCEPT_DECLARE(stor::exception::DiskSpaceAlarm, ex, msg.str());
  _alarmHandler->raiseAlarm(diskUsage->alarmName, diskUsage->alarmState, ex);

}


void ResourceMonitorCollection::revokeDiskUsageAlarm(DiskUsagePtr diskUsage)
{
  diskUsage->alarmState = AlarmHandler::OKAY;

  _alarmHandler->revokeAlarm(diskUsage->alarmName);
}


void ResourceMonitorCollection::calcNumberOfWorkers()
{
  _numberOfCopyWorkers.addSample( getProcessCount("CopyWorker.pl") );
  _numberOfInjectWorkers.addSample( getProcessCount("InjectWorker.pl") );
  
  _numberOfCopyWorkers.calculateStatistics();
  _numberOfInjectWorkers.calculateStatistics();
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
}


bool ResourceMonitorCollection::getSataBeasts(SATABeasts& sataBeasts)
{
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

    _sataBeastStatus = 99999;
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
  
  const CURLcode returnCode =
    curlInterface.getContent(
      "http://" + sataBeast + hostSuffix + "/status.asp",_sataUser, content
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
      << sataBeast << hostSuffix << ": " << content;
    XCEPT_DECLARE(stor::exception::SataBeast, ex, msg.str());
    _alarmHandler->raiseAlarm(sataBeast, AlarmHandler::WARNING, ex);

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

  _sataBeastStatus = newSataBeastStatus;

  if (_sataBeastStatus == 0) // no more problems
    _alarmHandler->revokeAlarm(sataBeast);

}


namespace {
  int filter(const struct dirent *dir)
  {
    return !fnmatch("[1-9]*", dir->d_name, 0);
  }
  
  bool grep(const struct dirent *dir, const std::string name)
  {
    bool match = false;
    
    std::ostringstream cmdline;
    cmdline << "/proc/" << dir->d_name << "/cmdline";
    
    std::ifstream in;
    in.open( cmdline.str().c_str() );
    
    if ( in.is_open() )
    {
      std::string line;
      while( getline(in,line) )
      {
        if ( line.find(name) != std::string::npos )
        {
          match = true;
          break;
        }
      }
      in.close();
    }
    
    return match;
  }
}


int ResourceMonitorCollection::getProcessCount(const std::string processName)
{
  
  int count(0);
  struct dirent **namelist;
  int n;
  
  n = scandir("/proc", &namelist, filter, 0);
  
  if (n < 0) return -1;
  
  while(n--)
  {
    if ( grep(namelist[n], processName) )
    {
      ++count;
    }
    free(namelist[n]);
  }
  free(namelist);
  
  return count;
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
