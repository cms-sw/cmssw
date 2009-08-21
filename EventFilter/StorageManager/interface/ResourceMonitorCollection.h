// $Id: ResourceMonitorCollection.h,v 1.9 2009/08/21 13:47:59 mommsen Exp $
/// @file: ResourceMonitorCollection.h 

#ifndef StorageManager_ResourceMonitorCollection_h
#define StorageManager_ResourceMonitorCollection_h

#include <set>
#include <vector>
#include <string>
#include <errno.h>

#include <boost/thread/mutex.hpp>
#include <boost/shared_ptr.hpp>

#include "xdata/String.h"

#include "EventFilter/StorageManager/interface/AlarmHandler.h"
#include "EventFilter/StorageManager/interface/Configuration.h"
#include "EventFilter/StorageManager/interface/MonitorCollection.h"


namespace stor {

  /**
   * A collection of MonitoredQuantities related to resource usages
   *
   * $Author: mommsen $
   * $Revision: 1.9 $
   * $Date: 2009/08/21 13:47:59 $
   */
  
  class ResourceMonitorCollection : public MonitorCollection
  {
  public:

    // Allow unit test to access the private methods
    friend class testResourceMonitorCollection;

    struct DiskUsageStats
    {
      MonitoredQuantity::Stats absDiskUsageStats;  // absolute disk usage in GB
      MonitoredQuantity::Stats relDiskUsageStats;  // percentage of disk space occupied
      size_t diskSize;                             // absolute size of disk in GB
      std::string pathName;                        // path of the disk
      AlarmHandler::ALARM_LEVEL alarmState;        // alarm level of the disk usage
    };
    typedef boost::shared_ptr<DiskUsageStats> DiskUsageStatsPtr;
    typedef std::vector<DiskUsageStatsPtr> DiskUsageStatsPtrList;

    struct Stats
    {
      DiskUsageStatsPtrList diskUsageStatsList;

      MonitoredQuantity::Stats numberOfCopyWorkersStats;
      MonitoredQuantity::Stats numberOfInjectWorkersStats;
      unsigned int             sataBeastStatus; // status code of SATA beast
    };


    explicit ResourceMonitorCollection
    (
      const utils::duration_t& updateInterval,
      boost::shared_ptr<AlarmHandler>
    );

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
      AlarmHandler::ALARM_LEVEL alarmState;

      DiskUsage(const utils::duration_t& updateInterval) :
        absDiskUsage(updateInterval,10),
        relDiskUsage(updateInterval,10) {}

      bool retrieveDiskSize();
    };
    typedef boost::shared_ptr<DiskUsage> DiskUsagePtr;
    typedef std::vector<DiskUsagePtr> DiskUsagePtrList;
    DiskUsagePtrList _diskUsageList;
    mutable boost::mutex _diskUsageListMutex;

    const utils::duration_t _updateInterval;

    MonitoredQuantity _numberOfCopyWorkers;
    MonitoredQuantity _numberOfInjectWorkers;
    unsigned int      _sataBeastStatus;


    //Prevent copying of the ResourceMonitorCollection
    ResourceMonitorCollection(ResourceMonitorCollection const&);
    ResourceMonitorCollection& operator=(ResourceMonitorCollection const&);

    virtual void do_calculateStatistics();
    virtual void do_reset();
    virtual void do_appendInfoSpaceItems(InfoSpaceItems&);
    virtual void do_updateInfoSpaceItems();

    void emitDiskAlarm(DiskUsagePtr, error_t);
    void emitDiskSpaceAlarm(DiskUsagePtr);
    void revokeDiskAlarm(DiskUsagePtr);

    void getDiskStats(Stats&) const;
    void calcDiskUsage();
    void calcNumberOfWorkers();
    int getProcessCount(const std::string processName);

    typedef std::set<std::string> SATABeasts;
    void checkSataBeasts();
    bool getSataBeasts(SATABeasts& sataBeasts);
    void checkSataBeast(const std::string& sataBeast);
    bool checkSataDisks(const std::string& sataBeast, const std::string& hostSuffix);
    void updateSataBeastStatus(const std::string& sataBeast, const std::string& content);


    boost::shared_ptr<AlarmHandler> _alarmHandler;
    double _highWaterMark;     // percentage of disk full when issuing an alarm
    std::string _sataUser;     // user name to log into SATA controller

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
