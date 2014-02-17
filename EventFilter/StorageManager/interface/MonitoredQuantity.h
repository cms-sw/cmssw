// $Id: MonitoredQuantity.h,v 1.10 2011/03/07 15:31:32 mommsen Exp $
/// @file: MonitoredQuantity.h 

#ifndef EventFilter_StorageManager_MonitoredQuantity_h
#define EventFilter_StorageManager_MonitoredQuantity_h

#include "boost/shared_ptr.hpp"
#include "boost/thread/mutex.hpp"

#include <math.h>
#include <stdint.h>
#include <vector>

#include "EventFilter/StorageManager/interface/Utils.h"


namespace stor
{

  /**
   * This class keeps track of statistics for a set of sample values 
   * and provides timing information on the samples.
   *
   * $Author: mommsen $
   * $Revision: 1.10 $
   * $Date: 2011/03/07 15:31:32 $
   */

  class MonitoredQuantity
  {
    
  public:
    class Stats;

    enum DataSetType { FULL = 0,      // the full data set (all samples)
                       RECENT = 1 };  // recent data only

    explicit MonitoredQuantity
    (
      utils::Duration_t expectedCalculationInterval,
      utils::Duration_t timeWindowForRecentResults
    );

    /**
     * Adds the specified doubled valued sample value to the monitor instance.
     */
    void addSample(const double& value = 1);

    /**
     * Adds the specified integer valued sample value to the monitor instance.
     */
    void addSample(const int& value = 1);

    /**
     * Adds the specified unsigned integer valued sample value to the monitor instance.
     */
    void addSample(const unsigned int& value = 1);

    /**
     * Adds the specified long valued sample value to the monitor instance.
     */
    void addSample(const long& value = 1);

    /**
     * Adds the specified unsigned long valued sample value to the monitor instance.
     */
    void addSample(const unsigned long& value = 1);

    /**
     * Adds the specified long long valued sample value to the monitor instance.
     */
    void addSample(const long long& value = 1);

    /**
     * Adds the specified unsigned long long valued sample value to the monitor instance.
     */
    void addSample(const unsigned long long& value = 1);

    /**
     * Adds the specified double valued sample value to the monitor instance
     * if it is larger than the previously added value.
     */
    void addSampleIfLarger(const double& value);

    /**
     * Forces a calculation of the statistics for the monitored quantity.
     * The frequency of the updates to the statistics is driven by how
     * often this method is called.  It is expected that this method
     * will be called once per interval specified by
     * expectedCalculationInterval
     */
    void calculateStatistics(const utils::TimePoint_t& currentTime = 
                             utils::getCurrentTime());

    /**
     * Resets the monitor (zeroes out all counters and restarts the
     * time interval).
     */
    void reset();

    /**
     * Enables the monitor (and resets the statistics to provide a
     * fresh start).
     */
    void enable();

    /**
     * Disables the monitor.
     */
    void disable();

    /**
     * Tests whether the monitor is currently enabled.
     */
    bool isEnabled() const {return enabled_;}

    /**
     * Specifies a new time interval to be used when calculating
     * "recent" statistics.
     */
    void setNewTimeWindowForRecentResults(const utils::Duration_t& interval);

    /**
     * Returns the length of the time window that has been specified
     * for recent results.  (This may be different than the actual
     * length of the recent time window which is affected by the
     * interval of calls to the calculateStatistics() method.  Use
     * a getDuration(RECENT) call to determine the actual recent
     * time window.)
     */
    utils::Duration_t getTimeWindowForRecentResults() const
    {
      return intervalForRecentStats_;
    }

    utils::Duration_t ExpectedCalculationInterval() const
    {
      return expectedCalculationInterval_;
    }

    /**
       Write all our collected statistics into the given Stats struct.
     */
    void getStats(Stats& stats) const;

  private:

    // Prevent copying of the MonitoredQuantity
    MonitoredQuantity(MonitoredQuantity const&);
    MonitoredQuantity& operator=(MonitoredQuantity const&);

    // Helper functions.
    void resetAccumulators();
    void resetResults();

    utils::TimePoint_t lastCalculationTime_;
    uint64_t workingSampleCount_;
    double workingValueSum_;
    double workingValueSumOfSquares_;
    double workingValueMin_;
    double workingValueMax_;
    double workingLastSampleValue_;

    mutable boost::mutex accumulationMutex_;

    uint32_t binCount_;
    uint32_t workingBinId_;
    std::vector<uint64_t> binSampleCount_;
    std::vector<double> binValueSum_;
    std::vector<double> binValueSumOfSquares_;
    std::vector<double> binValueMin_;
    std::vector<double> binValueMax_;
    std::vector<utils::Duration_t> binDuration_;
    std::vector<utils::TimePoint_t> binSnapshotTime_;

    uint64_t fullSampleCount_;
    double fullSampleRate_;
    double fullValueSum_;
    double fullValueSumOfSquares_;
    double fullValueAverage_;
    double fullValueRMS_;
    double fullValueMin_;
    double fullValueMax_;
    double fullValueRate_;
    utils::Duration_t fullDuration_;

    uint64_t recentSampleCount_;
    double recentSampleRate_;
    double recentValueSum_;
    double recentValueSumOfSquares_;
    double recentValueAverage_;
    double recentValueRMS_;
    double recentValueMin_;
    double recentValueMax_;
    double recentValueRate_;
    utils::Duration_t recentDuration_;
    double lastLatchedSampleValue_;
    double lastLatchedValueRate_;

    mutable boost::mutex resultsMutex_;

    bool enabled_;
    utils::Duration_t intervalForRecentStats_;  // seconds
    const utils::Duration_t expectedCalculationInterval_;  // seconds
  };

  struct MonitoredQuantity::Stats
  {
    uint64_t fullSampleCount;
    double fullSampleRate;
    double fullValueSum;
    double fullValueSumOfSquares;
    double fullValueAverage;
    double fullValueRMS;
    double fullValueMin;
    double fullValueMax;
    double fullValueRate;
    double fullSampleLatency;
    utils::Duration_t fullDuration;

    uint64_t recentSampleCount;
    double recentSampleRate;
    double recentValueSum;
    double recentValueSumOfSquares;
    double recentValueAverage;
    double recentValueRMS;
    double recentValueMin;
    double recentValueMax;
    double recentValueRate;
    double recentSampleLatency;
    utils::Duration_t recentDuration;
    std::vector<uint64_t> recentBinnedSampleCounts;
    std::vector<double> recentBinnedValueSums;
    std::vector<utils::Duration_t> recentBinnedDurations;
    std::vector<utils::TimePoint_t> recentBinnedSnapshotTimes;

    double lastSampleValue;
    double lastValueRate;
    bool   enabled;

    uint64_t getSampleCount(DataSetType t = FULL) const { return t == RECENT ? recentSampleCount : fullSampleCount; }
    double getValueSum(DataSetType t = FULL) const { return t == RECENT ? recentValueSum : fullValueSum; }
    double getValueAverage(DataSetType t = FULL) const { return t == RECENT ? recentValueAverage : fullValueAverage; }
    double getValueRate(DataSetType t = FULL) const { return t== RECENT ? recentValueRate : fullValueRate; }
    double getValueRMS(DataSetType t = FULL) const { return t == RECENT ? recentValueRMS : fullValueRMS; }
    double getValueMin(DataSetType t = FULL) const { return t == RECENT ? recentValueMin : fullValueMin; }
    double getValueMax(DataSetType t = FULL) const { return t == RECENT ? recentValueMax : fullValueMax; }
    utils::Duration_t getDuration(DataSetType t = FULL) const { return t == RECENT ? recentDuration : fullDuration; }
    double getSampleRate(DataSetType t = FULL) const { return t == RECENT ? recentSampleRate : fullSampleRate; }
    double getSampleLatency(DataSetType t = FULL) const { double v=getSampleRate(t); return v  ? 1e6/v : INFINITY;}
    double getLastSampleValue() const { return lastSampleValue; }
    double getLastValueRate() const { return lastValueRate; }
    bool   isEnabled() const { return enabled; }
  };

  typedef boost::shared_ptr<MonitoredQuantity> MonitoredQuantityPtr;

} // namespace stor

#endif // EventFilter_StorageManager_MonitoredQuantity_h



/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
