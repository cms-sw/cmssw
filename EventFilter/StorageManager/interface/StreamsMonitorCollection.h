// $Id: StreamsMonitorCollection.h,v 1.3 2009/07/09 15:34:44 mommsen Exp $
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
   * $Revision: 1.3 $
   * $Date: 2009/07/09 15:34:44 $
   */
  
  class StreamsMonitorCollection : public MonitorCollection
  {
  private:

    MonitoredQuantity _allStreamsFileCount;
    MonitoredQuantity _allStreamsVolume;
    MonitoredQuantity _allStreamsBandwidth;

  public:

    struct StreamRecord
    {
      std::string streamName;       // name of the stream
      MonitoredQuantity fileCount;  // number of files written for this stream
      MonitoredQuantity volume;     // data in MBytes stored in this stream
      MonitoredQuantity bandwidth;  // bandwidth in MBytes for this stream
      void incrementFileCount();
      void addSizeInBytes(double);
      StreamsMonitorCollection* parentCollection;

      StreamRecord(StreamsMonitorCollection* coll):
        parentCollection(coll) {}
    };

    // We do not know how many streams there will be.
    // Thus, we need a vector of them.
    typedef boost::shared_ptr<StreamRecord> StreamRecordPtr;
    typedef std::vector<StreamRecordPtr> StreamRecordList;


    StreamsMonitorCollection();

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

    utils::duration_t _timeWindowForRecentResults;

    xdata::UnsignedInteger32 _storedEvents;   // number of events stored in all streams
    xdata::Double _storedVolume;              // total volume in MB stored on disk

    // InfoSpace items which were defined in the old SM
    // xdata::Vector<xdata::String> _namesOfStream;                   // vector of stream names
    // xdata::Vector<xdata::UnsignedInteger32> _storedEventsInStream; // vector of events stored in stream N

  };
  
} // namespace stor

#endif // StorageManager_StreamsMonitorCollection_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
