// $Id: FragmentMonitorCollection.h,v 1.2 2009/06/10 08:15:22 dshpakov Exp $

#ifndef StorageManager_FragmentMonitorCollection_h
#define StorageManager_FragmentMonitorCollection_h

#include "xdata/Double.h"
#include "xdata/UnsignedInteger32.h"

#include "EventFilter/StorageManager/interface/MonitorCollection.h"


namespace stor {

  /**
   * A collection of MonitoredQuantities related to fragments
   *
   * $Author: dshpakov $
   * $Revision: 1.2 $
   * $Date: 2009/06/10 08:15:22 $
   */
  
  class FragmentMonitorCollection : public MonitorCollection
  {
  private:

    MonitoredQuantity _allFragmentSizes;
    MonitoredQuantity _eventFragmentSizes;
    MonitoredQuantity _dqmEventFragmentSizes;

    MonitoredQuantity _allFragmentBandwidth;
    MonitoredQuantity _eventFragmentBandwidth;
    MonitoredQuantity _dqmEventFragmentBandwidth;


  public:

    struct FragmentStats
    {
      MonitoredQuantity::Stats allFragmentSizeStats;
      MonitoredQuantity::Stats eventFragmentSizeStats;
      MonitoredQuantity::Stats dqmEventFragmentSizeStats;
      
      MonitoredQuantity::Stats allFragmentBandwidthStats;
      MonitoredQuantity::Stats eventFragmentBandwidthStats;
      MonitoredQuantity::Stats dqmEventFragmentBandwidthStats;
    };

    FragmentMonitorCollection();

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

    // InfoSpace items which were defined in the old SM
    // xdata::Double _duration;                  // Duration of run in seconds
    // xdata::UnsignedInteger32 _totalSamples;   // Total number of samples used for measurement
                                              // (same as receivedFrames)
    // xdata::UnsignedInteger32 _dqmRecords;     // Total number of DQM records (frames) received

    // xdata::Double _meanBandwidth;    // Total average bandwidth in MB/s
    // xdata::Double _meanRate;         // Total avarage number of frames/s
    // xdata::Double _meanLatency;      // Total average latency in micro-seconds/frame
    // xdata::Double _receivedVolume;   // Total received data in MB

    // xdata::UnsignedInteger32 _receivedPeriod4Stats;  // Time period per recent measurements
    // xdata::UnsignedInteger32 _receivedSamples4Stats; // Number of recent samples used for measurement
    // xdata::Double _instantLatency;   // Recent latency in micro-seconds/frame
    // xdata::Double _maxBandwidth;     // Recent maximum bandwidth in MB/s
    // xdata::Double _minBandwidth;     // Recent minimum bandwidth in MB/s

    // Why are these put into infospace if none of the DQM related measurements are?
    // xdata::UnsignedInteger32 _receivedDQMPeriod4Stats;  // Number of recent samples used for DQM measurement
    // xdata::UnsignedInteger32 _receivedDQMSamples4Stats; // Time period per recent DQMmeasurements

  };
  
} // namespace stor

#endif // StorageManager_FragmentMonitorCollection_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
