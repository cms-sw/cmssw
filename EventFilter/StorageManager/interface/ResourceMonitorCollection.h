// $Id: ResourceMonitorCollection.h,v 1.26 2010/04/12 15:25:26 mommsen Exp $
/// @file: ResourceMonitorCollection.h 

#ifndef StorageManager_ResourceMonitorCollection_h
#define StorageManager_ResourceMonitorCollection_h

#include <set>
#include <vector>
#include <string>
#include <errno.h>
#ifdef __APPLE__
typedef int error_t;
#endif

#include <boost/thread/mutex.hpp>
#include <boost/shared_ptr.hpp>

#include "xdata/Integer32.h"
#include "xdata/String.h"
#include "xdata/UnsignedInteger32.h"
#include "xdata/Vector.h"

#include "EventFilter/StorageManager/interface/AlarmHandler.h"
#include "EventFilter/StorageManager/interface/Configuration.h"
#include "EventFilter/StorageManager/interface/MonitorCollection.h"


namespace stor {

  class testResourceMonitorCollection;

  /**
   * A collection of MonitoredQuantities related to resource usages
   *
   * $Author: mommsen $
   * $Revision: 1.26 $
   * $Date: 2010/04/12 15:25:26 $
   */
  
  class ResourceMonitorCollection : public MonitorCollection
  {
  public:

    // Allow unit test to access the private methods
    friend class testResourceMonitorCollection;

    struct DiskUsageStats
    {
      double absDiskUsage;                         // absolute disk usage in GB
      double relDiskUsage;                         // percentage of disk space occupied
      double diskSize;                             // absolute size of disk in GB
      std::string pathName;                        // path of the disk
      AlarmHandler::ALARM_LEVEL alarmState;        // alarm level of the disk usage
    };
    typedef boost::shared_ptr<DiskUsageStats> DiskUsageStatsPtr;
    typedef std::vector<DiskUsageStatsPtr> DiskUsageStatsPtrList;

    struct Stats
    {
      DiskUsageStatsPtrList diskUsageStatsList;

      int numberOfCopyWorkers;
      int numberOfInjectWorkers;
      int sataBeastStatus;       // status code of SATA beast
    };


    /**
     * Constructor.
     */
    ResourceMonitorCollection
    (
      const utils::duration_t& updateInterval,
      boost::shared_ptr<AlarmHandler>
    );

    /**
     * Configures the disks used to write events
     */
    void configureDisks(DiskWritingParams const&);

    /**
     * Configures the resources to be monitored
     */
    void configureResources(ResourceMonitorParams const&);

    /**
     * Configures the alarms
     */
    void configureAlarms(AlarmParams const&);

    /**
     * Write all our collected statistics into the given Stats struct.
     */
    void getStats(Stats&) const;


  private:

    struct DiskUsage
    {
      double absDiskUsage;
      double relDiskUsage;
      double diskSize;
      std::string pathName;
      AlarmHandler::ALARM_LEVEL alarmState;
      std::string toString();
    };
    typedef boost::shared_ptr<DiskUsage> DiskUsagePtr;
    typedef std::vector<DiskUsagePtr> DiskUsagePtrList;
    DiskUsagePtrList _diskUsageList;
    mutable boost::mutex _diskUsageListMutex;

    const utils::duration_t _updateInterval;
    boost::shared_ptr<AlarmHandler> _alarmHandler;

    //Prevent copying of the ResourceMonitorCollection
    ResourceMonitorCollection(ResourceMonitorCollection const&);
    ResourceMonitorCollection& operator=(ResourceMonitorCollection const&);

    virtual void do_calculateStatistics();
    virtual void do_reset();
    virtual void do_appendInfoSpaceItems(InfoSpaceItems&);
    virtual void do_updateInfoSpaceItems();

    void addDisk(const std::string&);
    void addOtherDisks();
    void failIfImportantDisk(DiskUsagePtr);
    void emitDiskAlarm(DiskUsagePtr, error_t);
    void emitDiskSpaceAlarm(DiskUsagePtr);
    void revokeDiskAlarm(DiskUsagePtr);

    void getDiskStats(Stats&) const;
    void calcDiskUsage();
    void retrieveDiskSize(DiskUsagePtr);

    void calcNumberOfCopyWorkers();
    void calcNumberOfInjectWorkers();
    void checkNumberOfCopyWorkers();
    void checkNumberOfInjectWorkers();
    int getProcessCount(const std::string& processName, const int& uid=-1);

    typedef std::set<std::string> SATABeasts;
    void checkSataBeasts();
    bool getSataBeasts(SATABeasts& sataBeasts);
    void checkSataBeast(const std::string& sataBeast);
    bool checkSataDisks(const std::string& sataBeast, const std::string& hostSuffix);
    void updateSataBeastStatus(const std::string& sataBeast, const std::string& content);

    DiskWritingParams _dwParams;
    ResourceMonitorParams _rmParams;
    AlarmParams _alarmParams;

    int _numberOfCopyWorkers;
    int _numberOfInjectWorkers;
    unsigned int _nLogicalDisks;
    int _latchedSataBeastStatus;
    
    xdata::UnsignedInteger32 _copyWorkers;     // number of running copyWorkers
    xdata::UnsignedInteger32 _injectWorkers;   // number of running injectWorkers
    xdata::Integer32 _sataBeastStatus;         // status code of SATA beast
    xdata::UnsignedInteger32 _numberOfDisks;   // number of disks used for writing
    xdata::Vector<xdata::String> _diskPaths;   // list of disk paths
    xdata::Vector<xdata::UnsignedInteger32> _totalDiskSpace; // total disk space
    xdata::Vector<xdata::UnsignedInteger32> _usedDiskSpace;  // used disk space

  };
  
} // namespace stor

#endif // StorageManager_ResourceMonitorCollection_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
