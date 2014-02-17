// $Id: DQMEventMonitorCollection.h,v 1.9 2011/04/04 12:03:30 mommsen Exp $
/// @file: DQMEventMonitorCollection.h 

#ifndef EventFilter_StorageManager_DQMEventMonitorCollection_h
#define EventFilter_StorageManager_DQMEventMonitorCollection_h

#include "xdata/Double.h"
#include "xdata/UnsignedInteger32.h"

#include "EventFilter/StorageManager/interface/MonitorCollection.h"


namespace stor {

  /**
   * A collection of MonitoredQuantities related to fragments
   *
   * $Author: mommsen $
   * $Revision: 1.9 $
   * $Date: 2011/04/04 12:03:30 $
   */
  
  class DQMEventMonitorCollection : public MonitorCollection
  {
  private:

    MonitoredQuantity droppedDQMEventCounts_;

    MonitoredQuantity dqmEventSizes_;
    MonitoredQuantity servedDQMEventSizes_;
    MonitoredQuantity writtenDQMEventSizes_;

    MonitoredQuantity dqmEventBandwidth_;
    MonitoredQuantity servedDQMEventBandwidth_;
    MonitoredQuantity writtenDQMEventBandwidth_;

    MonitoredQuantity numberOfTopLevelFolders_;
    MonitoredQuantity numberOfUpdates_;
    MonitoredQuantity numberOfWrittenTopLevelFolders_;

    MonitoredQuantity numberOfCompleteUpdates_;

  public:

    struct DQMEventStats
    {
      MonitoredQuantity::Stats droppedDQMEventCountsStats;  //number of events
      
      MonitoredQuantity::Stats dqmEventSizeStats;             //MB
      MonitoredQuantity::Stats servedDQMEventSizeStats;       //MB
      MonitoredQuantity::Stats writtenDQMEventSizeStats;      //MB
      
      MonitoredQuantity::Stats dqmEventBandwidthStats;        //MB/s
      MonitoredQuantity::Stats servedDQMEventBandwidthStats;  //MB/s
      MonitoredQuantity::Stats writtenDQMEventBandwidthStats; //MB/s

      MonitoredQuantity::Stats numberOfTopLevelFoldersStats;  //number of top level folders
      MonitoredQuantity::Stats numberOfUpdatesStats;          //number of received updates per DQMKey
      MonitoredQuantity::Stats numberOfWrittenTopLevelFoldersStats; //number of top level folders written to disk

      MonitoredQuantity::Stats numberOfCompleteUpdatesStats;  //number of complete updates
    };

    explicit DQMEventMonitorCollection(const utils::Duration_t& updateInterval);

    const MonitoredQuantity& getDroppedDQMEventCountsMQ() const {
      return droppedDQMEventCounts_;
    }
    MonitoredQuantity& getDroppedDQMEventCountsMQ() {
      return droppedDQMEventCounts_;
    }

    const MonitoredQuantity& getDQMEventSizeMQ() const {
      return dqmEventSizes_;
    }
    MonitoredQuantity& getDQMEventSizeMQ() {
      return dqmEventSizes_;
    }

    const MonitoredQuantity& getServedDQMEventSizeMQ() const {
      return servedDQMEventSizes_;
    }
    MonitoredQuantity& getServedDQMEventSizeMQ() {
      return servedDQMEventSizes_;
    }

    const MonitoredQuantity& getWrittenDQMEventSizeMQ() const {
      return writtenDQMEventSizes_;
    }
    MonitoredQuantity& getWrittenDQMEventSizeMQ() {
      return writtenDQMEventSizes_;
    }

    const MonitoredQuantity& getDQMEventBandwidthMQ() const {
      return dqmEventBandwidth_;
    }
    MonitoredQuantity& getDQMEventBandwidthMQ() {
      return dqmEventBandwidth_;
    }

    const MonitoredQuantity& getServedDQMEventBandwidthMQ() const {
      return servedDQMEventBandwidth_;
    }
    MonitoredQuantity& getServedDQMEventBandwidthMQ() {
      return servedDQMEventBandwidth_;
    }

    const MonitoredQuantity& getWrittenDQMEventBandwidthMQ() const {
      return writtenDQMEventBandwidth_;
    }
    MonitoredQuantity& getWrittenDQMEventBandwidthMQ() {
      return writtenDQMEventBandwidth_;
    }

    const MonitoredQuantity& getNumberOfTopLevelFoldersMQ() const {
      return numberOfTopLevelFolders_;
    }
    MonitoredQuantity& getNumberOfTopLevelFoldersMQ() {
      return numberOfTopLevelFolders_;
    }

    const MonitoredQuantity& getNumberOfUpdatesMQ() const {
      return numberOfUpdates_;
    }
    MonitoredQuantity& getNumberOfUpdatesMQ() {
      return numberOfUpdates_;
    }

    const MonitoredQuantity& getNumberOfCompleteUpdatesMQ() const {
      return numberOfCompleteUpdates_;
    }
    MonitoredQuantity& getNumberOfCompleteUpdatesMQ() {
      return numberOfCompleteUpdates_;
    }

    const MonitoredQuantity& getNumberOfWrittenTopLevelFoldersMQ() const {
      return numberOfWrittenTopLevelFolders_;
    }
    MonitoredQuantity& getNumberOfWrittenTopLevelFoldersMQ() {
      return numberOfWrittenTopLevelFolders_;
    }

   /**
    * Write all our collected statistics into the given Stats struct.
    */
    void getStats(DQMEventStats& stats) const;


  private:

    //Prevent copying of the DQMEventMonitorCollection
    DQMEventMonitorCollection(DQMEventMonitorCollection const&);
    DQMEventMonitorCollection& operator=(DQMEventMonitorCollection const&);

    virtual void do_calculateStatistics();
    virtual void do_reset();
    virtual void do_appendInfoSpaceItems(InfoSpaceItems&);
    virtual void do_updateInfoSpaceItems();

    xdata::Double dqmFoldersPerEP_;
    xdata::UnsignedInteger32 processedDQMEvents_;
    xdata::UnsignedInteger32 droppedDQMEvents_;
    xdata::Double completeDQMUpdates_;
  };
  
} // namespace stor

#endif // EventFilter_StorageManager_DQMEventMonitorCollection_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
