// $Id: StreamsMonitorCollection.h,v 1.17 2011/11/17 17:35:41 mommsen Exp $
/// @file: StreamsMonitorCollection.h 

#ifndef EventFilter_StorageManager_StreamsMonitorCollection_h
#define EventFilter_StorageManager_StreamsMonitorCollection_h

#include <sstream>
#include <iomanip>
#include <vector>
#include <set>

#include <boost/thread/mutex.hpp>
#include <boost/shared_ptr.hpp>

#include "xdata/Double.h"
#include "xdata/String.h"
#include "xdata/UnsignedInteger32.h"
#include "xdata/Vector.h"

#include "EventFilter/StorageManager/interface/DbFileHandler.h"
#include "EventFilter/StorageManager/interface/MonitorCollection.h"
#include "EventFilter/StorageManager/interface/Utils.h"


namespace stor {

  /**
   * A collection of MonitoredQuantities of output streams
   *
   * $Author: mommsen $
   * $Revision: 1.17 $
   * $Date: 2011/11/17 17:35:41 $
   */
  
  class StreamsMonitorCollection : public MonitorCollection
  {
  public:

    struct StreamRecord
    {
      StreamRecord
      (
        StreamsMonitorCollection* coll,
        const utils::Duration_t& updateInterval,
        const utils::Duration_t& timeWindowForRecentResults
      ) :
      streamName(""),
      outputModuleLabel(""),
      fractionToDisk(1),
      fileCount(updateInterval,timeWindowForRecentResults),
      volume(updateInterval,timeWindowForRecentResults),
      bandwidth(updateInterval,timeWindowForRecentResults),
      parentCollection(coll) {}

      ~StreamRecord()
      { fileCountPerLS.clear(); }

      void incrementFileCount(const uint32_t lumiSection);
      void addSizeInBytes(double);
      bool reportLumiSectionInfo
      (
        const uint32_t& lumiSection,
        std::string& str
      );
      
      std::string streamName;       // name of the stream
      std::string outputModuleLabel;// label of the associated output module
      double fractionToDisk;        // fraction of events written to disk
      MonitoredQuantity fileCount;  // number of files written for this stream
      MonitoredQuantity volume;     // data in MBytes stored in this stream
      MonitoredQuantity bandwidth;  // bandwidth in MBytes for this stream

      StreamsMonitorCollection* parentCollection;

      typedef std::map<uint32_t, unsigned int> FileCountPerLumiSectionMap;
      FileCountPerLumiSectionMap fileCountPerLS;
    };

    // We do not know how many streams there will be.
    // Thus, we need a vector of them.
    typedef boost::shared_ptr<StreamRecord> StreamRecordPtr;
    typedef std::vector<StreamRecordPtr> StreamRecordList;


    struct EndOfRunReport
    {
      EndOfRunReport() { reset(); }

      void reset()
      { latestLumiSectionWritten = eolsCount = lsCountWithFiles = 0; }

      void updateLatestWrittenLumiSection(uint32_t ls)
      {
        if (ls > latestLumiSectionWritten) latestLumiSectionWritten = ls;
      }

      uint32_t latestLumiSectionWritten;
      unsigned int eolsCount;
      unsigned int lsCountWithFiles;
    };
    typedef boost::shared_ptr<EndOfRunReport> EndOfRunReportPtr;


    explicit StreamsMonitorCollection(const utils::Duration_t& updateInterval);

    StreamRecordPtr getNewStreamRecord();

    void getStreamRecords(StreamRecordList&) const;

    bool getStreamRecordsForOutputModuleLabel(const std::string&, StreamRecordList&) const;

    bool streamRecordsExist() const;

    const MonitoredQuantity& getAllStreamsFileCountMQ() const {
      return allStreamsFileCount_;
    }
    MonitoredQuantity& getAllStreamsFileCountMQ() {
      return allStreamsFileCount_;
    }

    const MonitoredQuantity& getAllStreamsVolumeMQ() const {
      return allStreamsVolume_;
    }
    MonitoredQuantity& getAllStreamsVolumeMQ() {
      return allStreamsVolume_;
    }

    const MonitoredQuantity& getAllStreamsBandwidthMQ() const {
      return allStreamsBandwidth_;
    }
    MonitoredQuantity& getAllStreamsBandwidthMQ() {
      return allStreamsBandwidth_;
    }

    void reportAllLumiSectionInfos(DbFileHandlerPtr, EndOfRunReportPtr);


  private:

    //Prevent copying of the StreamsMonitorCollection
    StreamsMonitorCollection(StreamsMonitorCollection const&);
    StreamsMonitorCollection& operator=(StreamsMonitorCollection const&);

    typedef std::set<uint32_t> UnreportedLS;
    void getListOfAllUnreportedLS(UnreportedLS&);

    virtual void do_calculateStatistics();
    virtual void do_reset();
    virtual void do_appendInfoSpaceItems(InfoSpaceItems&);
    virtual void do_updateInfoSpaceItems();

    StreamRecordList streamRecords_;
    mutable boost::mutex streamRecordsMutex_;

    const utils::Duration_t updateInterval_;
    const utils::Duration_t timeWindowForRecentResults_;

    MonitoredQuantity allStreamsFileCount_;
    MonitoredQuantity allStreamsVolume_;
    MonitoredQuantity allStreamsBandwidth_;

    xdata::UnsignedInteger32 storedEvents_;   // number of events stored in all streams
    xdata::Double storedVolume_;              // total volume in MB stored on disk
    xdata::Double bandwidthToDisk_;           // recent bandwidth in MB/s written to disk
    xdata::Vector<xdata::String> streamNames_; // names of all streams written
    xdata::Vector<xdata::UnsignedInteger32> eventsPerStream_; // total number of events stored per stream
    xdata::Vector<xdata::Double> ratePerStream_; // recent event rate (Hz) per stream
    xdata::Vector<xdata::Double> bandwidthPerStream_; // recent bandwidth (MB/s) per stream
  };
  
} // namespace stor

#endif // EventFilter_StorageManager_StreamsMonitorCollection_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
