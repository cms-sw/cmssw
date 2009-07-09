// $Id: ResourceMonitorCollection.cc,v 1.3 2009/06/15 12:32:25 mommsen Exp $

#include <string>
#include <sstream>
#include <iomanip>
#include <sys/statfs.h>
#include <dirent.h>
#include <fnmatch.h>

#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/ResourceMonitorCollection.h"
#include "EventFilter/StorageManager/interface/Utils.h"

using namespace stor;

ResourceMonitorCollection::ResourceMonitorCollection(xdaq::Application *app) :
MonitorCollection(app),
_app(app),
_pool(0),
_progressMarker( "unused" )
{
  _infoSpaceItems.push_back(std::make_pair("progressMarker", &_progressMarker));

  putItemsIntoInfoSpace();
}


void ResourceMonitorCollection::configureDisks(DiskWritingParams const& dwParams)
{
  boost::mutex::scoped_lock sl(_diskUsageListMutex);

  _highWaterMark = dwParams._highWaterMark;

  int nLogicalDisk = dwParams._nLogicalDisk;
  unsigned int nD = nLogicalDisk ? nLogicalDisk : 1;
  _diskUsageList.clear();
  _diskUsageList.reserve(nD);

  for (unsigned int i=0; i<nD; ++i) {

    DiskUsagePtr diskUsage(new DiskUsage);
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
    diskUsageStats->warningColor = (*it)->warningColor;
    stats.diskUsageStatsList.push_back(diskUsageStats);
  }
}


void ResourceMonitorCollection::do_calculateStatistics()
{
  calcPoolUsage();
  calcDiskUsage();
  calcNumberOfWorkers();
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
  diskUsage->warningColor = "#EF5A10";

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
  utils::raiseAlarm(diskUsage->alarmName, "warning", ex, _app);
}


void ResourceMonitorCollection::revokeDiskUsageAlarm(DiskUsagePtr diskUsage)
{
  diskUsage->warningColor = "#FFFFFF";

  utils::revokeAlarm(diskUsage->alarmName, _app);
}


void ResourceMonitorCollection::calcNumberOfWorkers()
{
  _numberOfCopyWorkers.addSample( getProcessCount("CopyWorker.pl") );
  _numberOfInjectWorkers.addSample( getProcessCount("InjectWorker.pl") );
  
  _numberOfCopyWorkers.calculateStatistics();
  _numberOfInjectWorkers.calculateStatistics();
}


void ResourceMonitorCollection::do_updateInfoSpace()
{
  //nothing to do: the progressMarker does not change its value
}


void ResourceMonitorCollection::do_reset()
{
  _poolUsage.reset();
  _numberOfCopyWorkers.reset();
  _numberOfInjectWorkers.reset();

  boost::mutex::scoped_lock sl(_diskUsageListMutex);
  for ( DiskUsagePtrList::const_iterator it = _diskUsageList.begin(),
          itEnd = _diskUsageList.end();
        it != itEnd;
        ++it)
  {
    (*it)->absDiskUsage.reset();
    (*it)->relDiskUsage.reset();
    (*it)->warningColor = "#FFFFFF";
  }
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
