// $Id: FragmentMonitorCollection.h,v 1.6 2009/08/24 14:31:11 mommsen Exp $
/// @file: FragmentMonitorCollection.h 

#ifndef StorageManager_FragmentMonitorCollection_h
#define StorageManager_FragmentMonitorCollection_h

#include "xdata/Double.h"
#include "xdata/UnsignedInteger32.h"

#include "EventFilter/StorageManager/interface/MonitorCollection.h"


namespace stor {

  /**
   * A collection of MonitoredQuantities related to fragments
   *
   * $Author: mommsen $
   * $Revision: 1.6 $
   * $Date: 2009/08/24 14:31:11 $
   */
  
  class FragmentMonitorCollection : public MonitorCollection
  {
  private:

    MonitoredQuantity _allFragmentSizes;
    MonitoredQuantity _allFragmentBandwidth;

    MonitoredQuantity _eventFragmentSizes;
    MonitoredQuantity _eventFragmentBandwidth;

    MonitoredQuantity _dqmEventFragmentSizes;
    MonitoredQuantity _dqmEventFragmentBandwidth;


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

    explicit FragmentMonitorCollection(const utils::duration_t& updateInterval);

    void addEventFragmentSample(const double bytecount);

    void addDQMEventFragmentSample(const double bytecount);

    const MonitoredQuantity& getAllFragmentSizeMQ() const {
      return _allFragmentSizes;
    }
    MonitoredQuantity& getAllFragmentSizeMQ() {
      return _allFragmentSizes;
    }

    const MonitoredQuantity& getEventFragmentSizeMQ() const {
      return _eventFragmentSizes;
    }
    MonitoredQuantity& getEventFragmentSizeMQ() {
      return _eventFragmentSizes;
    }

    const MonitoredQuantity& getDQMEventFragmentSizeMQ() const {
      return _dqmEventFragmentSizes;
    }
    MonitoredQuantity& getDQMEventFragmentSizeMQ() {
      return _dqmEventFragmentSizes;
    }

    const MonitoredQuantity& getAllFragmentBandwidthMQ() const {
      return _allFragmentBandwidth;
    }
    MonitoredQuantity& getAllFragmentBandwidthMQ() {
      return _allFragmentBandwidth;
    }

    const MonitoredQuantity& getEventFragmentBandwidthMQ() const {
      return _eventFragmentBandwidth;
    }
    MonitoredQuantity& getEventFragmentBandwidthMQ() {
      return _eventFragmentBandwidth;
    }

    const MonitoredQuantity& getDQMEventFragmentBandwidthMQ() const {
      return _dqmEventFragmentBandwidth;
    }
    MonitoredQuantity& getDQMEventFragmentBandwidthMQ() {
      return _dqmEventFragmentBandwidth;
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

    xdata::UnsignedInteger32 _receivedFrames; // Total I2O frames received
    xdata::Double _instantBandwidth;          // Recent bandwidth in MB/s
    xdata::Double _instantRate;               // Recent number of frames/s

  };
  
} // namespace stor

#endif // StorageManager_FragmentMonitorCollection_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
