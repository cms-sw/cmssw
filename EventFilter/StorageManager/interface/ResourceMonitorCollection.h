// $Id: ResourceMonitorCollection.h,v 1.2 2009/06/10 08:15:23 dshpakov Exp $

#ifndef StorageManager_ResourceMonitorCollection_h
#define StorageManager_ResourceMonitorCollection_h

#include <vector>
#include <string>

#include <boost/thread/mutex.hpp>
#include <boost/shared_ptr.hpp>

#include "toolbox/mem/Pool.h"
#include "xdata/String.h"

#include "EventFilter/StorageManager/interface/Configuration.h"
#include "EventFilter/StorageManager/interface/MonitorCollection.h"


namespace stor {

  /**
   * A collection of MonitoredQuantities related to resource usages
   *
   * $Author: dshpakov $
   * $Revision: 1.2 $
   * $Date: 2009/06/10 08:15:23 $
   */
  
  class ResourceMonitorCollection : public MonitorCollection
  {
  public:

    struct DiskUsageStats
    {
      MonitoredQuantity::Stats absDiskUsageStats;  // absolute disk usage in GB
      MonitoredQuantity::Stats relDiskUsageStats;  // percentage of disk space occupied
      size_t diskSize;                             // absolute size of disk in GB
      std::string pathName;                        // path of the disk
      std::string warningColor;                    // HTML color code for disk usage warning
    };
    typedef boost::shared_ptr<DiskUsageStats> DiskUsageStatsPtr;
    typedef std::vector<DiskUsageStatsPtr> DiskUsageStatsPtrList;

    struct Stats
    {
      DiskUsageStatsPtrList diskUsageStatsList;

      MonitoredQuantity::Stats poolUsageStats; // I2O message pool usage in bytes
      MonitoredQuantity::Stats numberOfCopyWorkersStats;
      MonitoredQuantity::Stats numberOfInjectWorkersStats;
    };


    explicit ResourceMonitorCollection(xdaq::Application*);

    /**
     * Stores the given memory pool pointer if not yet set.
     * If it is already set, the argument is ignored.
     */
    void setMemoryPoolPointer(toolbox::mem::Pool*);

    /**
     * Configures the disks used to write events
     */
    void configureDisks(DiskWritingParams const&);

    /**
     * Write all our collected statistics into the given Stats struct.
     */
    void getStats(Stats&) const;


  private:

    struct DiskUsage
    {
      MonitoredQuantity absDiskUsage;
      MonitoredQuantity relDiskUsage;
      size_t diskSize;
      std::string pathName;
      std::string warningColor;
      std::string alarmName;
    };
    typedef boost::shared_ptr<DiskUsage> DiskUsagePtr;
    typedef std::vector<DiskUsagePtr> DiskUsagePtrList;
    DiskUsagePtrList _diskUsageList;
    mutable boost::mutex _diskUsageListMutex;

    MonitoredQuantity _poolUsage;
    MonitoredQuantity _numberOfCopyWorkers;
    MonitoredQuantity _numberOfInjectWorkers;


    //Prevent copying of the ResourceMonitorCollection
    ResourceMonitorCollection(ResourceMonitorCollection const&);
    ResourceMonitorCollection& operator=(ResourceMonitorCollection const&);

    virtual void do_calculateStatistics();
    virtual void do_reset();
    virtual void do_appendInfoSpaceItems(InfoSpaceItems&);
    virtual void do_updateInfoSpaceItems();

    void emitDiskUsageAlarm(DiskUsagePtr);
    void revokeDiskUsageAlarm(DiskUsagePtr);

    void getDiskStats(Stats&) const;
    void calcPoolUsage();
    void calcDiskUsage();
    void calcNumberOfWorkers();
    int getProcessCount(const std::string processName);

    xdaq::Application* _app;
    toolbox::mem::Pool* _pool;
    double _highWaterMark;     //percentage of disk full when issuing an alarm

    // Unused status string from old SM
    xdata::String _progressMarker;
  };
  
} // namespace stor

#endif // StorageManager_ResourceMonitorCollection_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
