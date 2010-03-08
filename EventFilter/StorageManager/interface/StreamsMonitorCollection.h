// $Id: StreamsMonitorCollection.h,v 1.7 2009/12/16 14:44:43 mommsen Exp $
/// @file: StreamsMonitorCollection.h 

#ifndef StorageManager_StreamsMonitorCollection_h
#define StorageManager_StreamsMonitorCollection_h

#include <sstream>
#include <iomanip>
#include <vector>

#include <boost/thread/mutex.hpp>
#include <boost/shared_ptr.hpp>

#include "xdata/Double.h"
#include "xdata/String.h"
#include "xdata/UnsignedInteger32.h"
#include "xdata/Vector.h"

#include "EventFilter/StorageManager/interface/MonitorCollection.h"
#include "EventFilter/StorageManager/interface/Utils.h"


namespace stor {

  /**
   * A collection of MonitoredQuantities of output streams
   *
   * $Author: mommsen $
   * $Revision: 1.7 $
   * $Date: 2009/12/16 14:44:43 $
   */
  
  class StreamsMonitorCollection : public MonitorCollection
  {
  public:

    struct StreamRecord
    {
      std::string streamName;       // name of the stream
      double fractionToDisk;        // fraction of events written to disk
      MonitoredQuantity fileCount;  // number of files written for this stream
      MonitoredQuantity volume;     // data in MBytes stored in this stream
      MonitoredQuantity bandwidth;  // bandwidth in MBytes for this stream
      void incrementFileCount();
      void addSizeInBytes(double);
      StreamsMonitorCollection* parentCollection;

      StreamRecord
      (
        StreamsMonitorCollection* coll,
        const utils::duration_t& updateInterval,
        const utils::duration_t& timeWindowForRecentResults
      ) :
        fileCount(updateInterval,timeWindowForRecentResults),
        volume(updateInterval,timeWindowForRecentResults),
        bandwidth(updateInterval,timeWindowForRecentResults),
        parentCollection(coll) {}
    };

    // We do not know how many streams there will be.
    // Thus, we need a vector of them.
    typedef boost::shared_ptr<StreamRecord> StreamRecordPtr;
    typedef std::vector<StreamRecordPtr> StreamRecordList;


    explicit StreamsMonitorCollection(const utils::duration_t& updateInterval);

    const StreamRecordPtr getNewStreamRecord();

    const StreamRecordList& getStreamRecordsMQ() const {
      return _streamRecords;
    }

    const MonitoredQuantity& getAllStreamsFileCountMQ() const {
      return _allStreamsFileCount;
    }
    MonitoredQuantity& getAllStreamsFileCountMQ() {
      return _allStreamsFileCount;
    }

    const MonitoredQuantity& getAllStreamsVolumeMQ() const {
      return _allStreamsVolume;
    }
    MonitoredQuantity& getAllStreamsVolumeMQ() {
      return _allStreamsVolume;
    }

    const MonitoredQuantity& getAllStreamsBandwidthMQ() const {
      return _allStreamsBandwidth;
    }
    MonitoredQuantity& getAllStreamsBandwidthMQ() {
      return _allStreamsBandwidth;
    }



  private:

    //Prevent copying of the StreamsMonitorCollection
    StreamsMonitorCollection(StreamsMonitorCollection const&);
    StreamsMonitorCollection& operator=(StreamsMonitorCollection const&);

    virtual void do_calculateStatistics();
    virtual void do_reset();
    virtual void do_appendInfoSpaceItems(InfoSpaceItems&);
    virtual void do_updateInfoSpaceItems();

    StreamRecordList _streamRecords;
    mutable boost::mutex _streamRecordsMutex;

    const utils::duration_t _updateInterval;
    const utils::duration_t _timeWindowForRecentResults;

    MonitoredQuantity _allStreamsFileCount;
    MonitoredQuantity _allStreamsVolume;
    MonitoredQuantity _allStreamsBandwidth;

    xdata::UnsignedInteger32 _storedEvents;   // number of events stored in all streams
    xdata::Double _storedVolume;              // total volume in MB stored on disk
    xdata::Double _bandwidthToDisk;           // recent bandwidth in MB/s written to disk
    xdata::Vector<xdata::String> _streamNames; // names of all streams written
    xdata::Vector<xdata::UnsignedInteger32> _eventsPerStream; // total number of events stored per stream
    xdata::Vector<xdata::Double> _ratePerStream; // recent event rate (Hz) per stream
    xdata::Vector<xdata::Double> _bandwidthPerStream; // recent bandwidth (MB/s) per stream
  };
  
} // namespace stor

#endif // StorageManager_StreamsMonitorCollection_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
