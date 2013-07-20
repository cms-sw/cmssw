// $Id: FragmentMonitorCollection.h,v 1.8 2011/03/07 15:31:32 mommsen Exp $
/// @file: FragmentMonitorCollection.h 

#ifndef EventFilter_StorageManager_FragmentMonitorCollection_h
#define EventFilter_StorageManager_FragmentMonitorCollection_h

#include "xdata/Double.h"
#include "xdata/UnsignedInteger32.h"

#include "EventFilter/StorageManager/interface/MonitorCollection.h"


namespace stor {

  /**
   * A collection of MonitoredQuantities related to fragments
   *
   * $Author: mommsen $
   * $Revision: 1.8 $
   * $Date: 2011/03/07 15:31:32 $
   */
  
  class FragmentMonitorCollection : public MonitorCollection
  {
  private:

    MonitoredQuantity allFragmentSizes_;
    MonitoredQuantity allFragmentBandwidth_;

    MonitoredQuantity eventFragmentSizes_;
    MonitoredQuantity eventFragmentBandwidth_;

    MonitoredQuantity dqmEventFragmentSizes_;
    MonitoredQuantity dqmEventFragmentBandwidth_;


  public:

    struct FragmentStats
    {
      MonitoredQuantity::Stats allFragmentSizeStats;
      MonitoredQuantity::Stats allFragmentBandwidthStats;

      MonitoredQuantity::Stats eventFragmentSizeStats;
      MonitoredQuantity::Stats eventFragmentBandwidthStats;

      MonitoredQuantity::Stats dqmEventFragmentSizeStats;
      MonitoredQuantity::Stats dqmEventFragmentBandwidthStats;
    };

    explicit FragmentMonitorCollection(const utils::Duration_t& updateInterval);

    /**
      Add a generic fragment size of bytes
    */
    void addFragmentSample(const double bytecount);

    /**
      Add an event fragment size of bytes
    */
    void addEventFragmentSample(const double bytecount);

    /**
      Add a DQM event fragment size of bytes
    */
     void addDQMEventFragmentSample(const double bytecount);

    const MonitoredQuantity& getAllFragmentSizeMQ() const {
      return allFragmentSizes_;
    }
    MonitoredQuantity& getAllFragmentSizeMQ() {
      return allFragmentSizes_;
    }

    const MonitoredQuantity& getEventFragmentSizeMQ() const {
      return eventFragmentSizes_;
    }
    MonitoredQuantity& getEventFragmentSizeMQ() {
      return eventFragmentSizes_;
    }

    const MonitoredQuantity& getDQMEventFragmentSizeMQ() const {
      return dqmEventFragmentSizes_;
    }
    MonitoredQuantity& getDQMEventFragmentSizeMQ() {
      return dqmEventFragmentSizes_;
    }

    const MonitoredQuantity& getAllFragmentBandwidthMQ() const {
      return allFragmentBandwidth_;
    }
    MonitoredQuantity& getAllFragmentBandwidthMQ() {
      return allFragmentBandwidth_;
    }

    const MonitoredQuantity& getEventFragmentBandwidthMQ() const {
      return eventFragmentBandwidth_;
    }
    MonitoredQuantity& getEventFragmentBandwidthMQ() {
      return eventFragmentBandwidth_;
    }

    const MonitoredQuantity& getDQMEventFragmentBandwidthMQ() const {
      return dqmEventFragmentBandwidth_;
    }
    MonitoredQuantity& getDQMEventFragmentBandwidthMQ() {
      return dqmEventFragmentBandwidth_;
    }

   /**
    * Write all our collected statistics into the given Stats struct.
    */
    void getStats(FragmentStats& stats) const;


  private:

    //Prevent copying of the FragmentMonitorCollection
    FragmentMonitorCollection(FragmentMonitorCollection const&);
    FragmentMonitorCollection& operator=(FragmentMonitorCollection const&);

    virtual void do_calculateStatistics();
    virtual void do_reset();
    virtual void do_appendInfoSpaceItems(InfoSpaceItems&);
    virtual void do_updateInfoSpaceItems();

    xdata::UnsignedInteger32 receivedFrames_; // Total I2O frames received
    xdata::Double instantBandwidth_;          // Recent bandwidth in MB/s
    xdata::Double instantRate_;               // Recent number of frames/s

  };
  
} // namespace stor

#endif // EventFilter_StorageManager_FragmentMonitorCollection_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
