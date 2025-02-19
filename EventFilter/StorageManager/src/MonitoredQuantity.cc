// $Id: MonitoredQuantity.cc,v 1.13 2011/04/07 08:01:40 mommsen Exp $
/// @file: MonitoredQuantity.cc

#include "EventFilter/StorageManager/interface/MonitoredQuantity.h"

#include <algorithm>
#include <math.h>


namespace stor {
  
  MonitoredQuantity::MonitoredQuantity
  (
    utils::Duration_t expectedCalculationInterval,
    utils::Duration_t timeWindowForRecentResults
  ):
  enabled_(true),
  expectedCalculationInterval_(expectedCalculationInterval)
  {
    setNewTimeWindowForRecentResults(timeWindowForRecentResults);
  }
  
  void MonitoredQuantity::addSample(const double& value)
  {
    if (! enabled_) {return;}
    
    boost::mutex::scoped_lock sl(accumulationMutex_);
    
    if ( lastCalculationTime_.is_not_a_date_time() )
    {
      lastCalculationTime_ = utils::getCurrentTime();
    }
    
    ++workingSampleCount_;
    workingValueSum_ += value;
    workingValueSumOfSquares_ += (value * value);
    
    if (value < workingValueMin_) workingValueMin_ = value;
    if (value > workingValueMax_) workingValueMax_ = value;
    
    workingLastSampleValue_ = value;
  }
  
  void MonitoredQuantity::addSample(const int& value)
  {
    addSample(static_cast<double>(value));
  }
  
  void MonitoredQuantity::addSample(const unsigned int& value)
  {
    addSample(static_cast<double>(value));
  }
  
  void MonitoredQuantity::addSample(const long& value)
  {
    addSample(static_cast<double>(value));
  }
  
  void MonitoredQuantity::addSample(const unsigned long& value)
  {
    addSample(static_cast<double>(value));
  }
  
  void MonitoredQuantity::addSample(const long long& value)
  {
    addSample(static_cast<double>(value));
  }
  
  void MonitoredQuantity::addSample(const unsigned long long& value)
  {
    addSample(static_cast<double>(value));
  }
  
  void MonitoredQuantity::addSampleIfLarger(const double& value)
  {
    if (value > workingLastSampleValue_)
      addSample(value);
  }
  
  void MonitoredQuantity::calculateStatistics(const utils::TimePoint_t& currentTime)
  {
    if (! enabled_) {return;}
    
    // create local copies of the working values to minimize the
    // time that we could block a thread trying to add a sample.
    // Also, reset the working values.
    long long latestSampleCount;
    double latestValueSum;
    double latestValueSumOfSquares;
    double latestValueMin;
    double latestValueMax;
    utils::Duration_t latestDuration;
    utils::TimePoint_t latestSnapshotTime;
    double latestLastLatchedSampleValue;
    {
      boost::mutex::scoped_lock sl(accumulationMutex_);

      if (lastCalculationTime_.is_not_a_date_time()) {return;}
      if (currentTime - lastCalculationTime_ < expectedCalculationInterval_) {return;}
      
      latestSampleCount = workingSampleCount_;
      latestValueSum = workingValueSum_;
      latestValueSumOfSquares = workingValueSumOfSquares_;
      latestValueMin = workingValueMin_;
      latestValueMax = workingValueMax_;
      latestDuration = currentTime - lastCalculationTime_;
      latestSnapshotTime = currentTime;
      latestLastLatchedSampleValue = workingLastSampleValue_;
      
      lastCalculationTime_ = currentTime;
      workingSampleCount_ = 0;
      workingValueSum_ = 0.0;
      workingValueSumOfSquares_ = 0.0;
      workingValueMin_ =  INFINITY;
      workingValueMax_ = -INFINITY;
    }
    
    // lock out any interaction with the results while we update them
    {
      boost::mutex::scoped_lock sl(resultsMutex_);
      lastLatchedSampleValue_ = latestLastLatchedSampleValue;
      
      // we simply add the latest results to the full set
      fullSampleCount_ += latestSampleCount;
      fullValueSum_ += latestValueSum;
      fullValueSumOfSquares_ += latestValueSumOfSquares;
      if (latestValueMin < fullValueMin_) {fullValueMin_ = latestValueMin;}
      if (latestValueMax > fullValueMax_) {fullValueMax_ = latestValueMax;}
      fullDuration_ += latestDuration;
      
      // for the recent results, we need to replace the contents of
      // the working bin and re-calculate the recent values
      binSampleCount_[workingBinId_] = latestSampleCount;
      binValueSum_[workingBinId_] = latestValueSum;
      binValueSumOfSquares_[workingBinId_] = latestValueSumOfSquares;
      binValueMin_[workingBinId_] = latestValueMin;
      binValueMax_[workingBinId_] = latestValueMax;
      binDuration_[workingBinId_] = latestDuration;
      binSnapshotTime_[workingBinId_] = latestSnapshotTime;
      
      lastLatchedValueRate_ = latestValueSum / utils::durationToSeconds(latestDuration);
      
      recentSampleCount_ = 0;
      recentValueSum_ = 0.0;
      recentValueSumOfSquares_ = 0.0;
      recentValueMin_ =  INFINITY;
      recentValueMax_ = -INFINITY;
      recentDuration_ = boost::posix_time::seconds(0);
      
      for (unsigned int idx = 0; idx < binCount_; ++idx) {
        recentSampleCount_ += binSampleCount_[idx];
        recentValueSum_ += binValueSum_[idx];
        recentValueSumOfSquares_ += binValueSumOfSquares_[idx];
        if (binValueMin_[idx] < recentValueMin_) {
          recentValueMin_ = binValueMin_[idx];
        }
        if (binValueMax_[idx] > recentValueMax_) {
          recentValueMax_ = binValueMax_[idx];
        }
        recentDuration_ += binDuration_[idx];
      }
      
      // update the working bin ID here so that we are ready for
      // the next calculation request
      ++workingBinId_;
      if (workingBinId_ >= binCount_) {workingBinId_ = 0;}
      
      // calculate the derived full values
      const double fullDuration = utils::durationToSeconds(fullDuration_);
      fullSampleRate_ = fullSampleCount_ / fullDuration;
      fullValueRate_ = fullValueSum_ / fullDuration;
      
      if (fullSampleCount_ > 0) {
        fullValueAverage_ = fullValueSum_ / static_cast<double>(fullSampleCount_);
        
        double squareAvg = fullValueSumOfSquares_ / static_cast<double>(fullSampleCount_);
        double avg = fullValueSum_ / static_cast<double>(fullSampleCount_);
        double sigSquared = squareAvg - avg*avg;
        if(sigSquared > 0.0) {
          fullValueRMS_ = sqrt(sigSquared);
        }
        else {
          fullValueRMS_ = 0.0;
        }
      }
      else {
        fullValueAverage_ = 0.0;
        fullValueRMS_ = 0.0;
      }
      
      // calculate the derived recent values
      const double recentDuration = utils::durationToSeconds(recentDuration_);
      if (recentDuration > 0) {
        recentSampleRate_ = recentSampleCount_ / recentDuration;
        recentValueRate_ = recentValueSum_ / recentDuration;
      }
      else {
        recentSampleRate_ = 0.0;
        recentValueRate_ = 0.0;
      }
      
      if (recentSampleCount_ > 0) {
        recentValueAverage_ = recentValueSum_ / static_cast<double>(recentSampleCount_);
        
        double squareAvg = recentValueSumOfSquares_ /
          static_cast<double>(recentSampleCount_);
        double avg = recentValueSum_ / static_cast<double>(recentSampleCount_);
        double sigSquared = squareAvg - avg*avg;
        if(sigSquared > 0.0) {
          recentValueRMS_ = sqrt(sigSquared);
        }
        else {
          recentValueRMS_ = 0.0;
        }
      }
      else {
        recentValueAverage_ = 0.0;
        recentValueRMS_ = 0.0;
      }
    }
  }
  
  void MonitoredQuantity::resetAccumulators()
  {
    lastCalculationTime_ = boost::posix_time::not_a_date_time;
    workingSampleCount_ = 0;
    workingValueSum_ = 0.0;
    workingValueSumOfSquares_ = 0.0;
    workingValueMin_ =  INFINITY;
    workingValueMax_ = -INFINITY;
    workingLastSampleValue_ = 0;
  }
  
  void MonitoredQuantity::resetResults()
  {
    workingBinId_ = 0;
    for (unsigned int idx = 0; idx < binCount_; ++idx) {
      binSampleCount_[idx] = 0;
      binValueSum_[idx] = 0.0;
      binValueSumOfSquares_[idx] = 0.0;
      binValueMin_[idx] =  INFINITY;
      binValueMax_[idx] = -INFINITY;
      binDuration_[idx] = boost::posix_time::seconds(0);
      binSnapshotTime_[idx] = boost::posix_time::not_a_date_time;
    }
    
    fullSampleCount_ = 0;
    fullSampleRate_ = 0.0;
    fullValueSum_ = 0.0;
    fullValueSumOfSquares_ = 0.0;
    fullValueAverage_ = 0.0;
    fullValueRMS_ = 0.0;
    fullValueMin_ =  INFINITY;
    fullValueMax_ = -INFINITY;
    fullValueRate_ = 0.0;
    fullDuration_ = boost::posix_time::seconds(0);
    
    recentSampleCount_ = 0;
    recentSampleRate_ = 0.0;
    recentValueSum_ = 0.0;
    recentValueSumOfSquares_ = 0.0;
    recentValueAverage_ = 0.0;
    recentValueRMS_ = 0.0;
    recentValueMin_ =  INFINITY;
    recentValueMax_ = -INFINITY;
    recentValueRate_ = 0.0;
    recentDuration_ = boost::posix_time::seconds(0);
    lastLatchedSampleValue_ = 0.0;
    lastLatchedValueRate_ = 0.0;
  }
  
  void MonitoredQuantity::reset()
  {
    {
      boost::mutex::scoped_lock sl(accumulationMutex_);
      resetAccumulators();
    }
    
    {
      boost::mutex::scoped_lock sl(resultsMutex_);
      resetResults();
    }
  }
  
  void MonitoredQuantity::enable()
  {
    if (! enabled_) {
      reset();
      enabled_ = true;
    }
  }
  
  void MonitoredQuantity::disable()
  {
    // It is faster to just set enabled_ to false than to test and set
    // it conditionally.
    enabled_ = false;
  }
  
  void MonitoredQuantity::setNewTimeWindowForRecentResults(const utils::Duration_t& interval)
  {
    // lock the results objects since we're dramatically changing the
    // bins used for the recent results
    {
      boost::mutex::scoped_lock sl(resultsMutex_);
      
      intervalForRecentStats_ = interval;
      
      // determine how many bins we should use in our sliding window
      // by dividing the input time window by the expected calculation
      // interval and rounding to the nearest integer.
      // In case that the calculation interval is larger then the 
      // interval for recent stats, keep the last one.
      binCount_ = std::max(1U,
        static_cast<unsigned int>(
          (intervalForRecentStats_.total_nanoseconds() / expectedCalculationInterval_.total_nanoseconds()) + 0.5
        )      
      );
      
      // create the vectors for the binned quantities
      binSampleCount_.reserve(binCount_);
      binValueSum_.reserve(binCount_);
      binValueSumOfSquares_.reserve(binCount_);
      binValueMin_.reserve(binCount_);
      binValueMax_.reserve(binCount_);
      binDuration_.reserve(binCount_);
      binSnapshotTime_.reserve(binCount_);
      
      resetResults();
    }
    
    {
      boost::mutex::scoped_lock sl(accumulationMutex_);
      resetAccumulators();
    }
    
    // call the reset method to populate the correct initial values
    // for the internal sample data
    //reset();
  }
  
  void
  MonitoredQuantity::getStats(Stats& s) const
  {
    boost::mutex::scoped_lock results(resultsMutex_);
    
    s.fullSampleCount = fullSampleCount_;
    s.fullSampleRate = fullSampleRate_;
    s.fullValueSum = fullValueSum_;
    s.fullValueSumOfSquares = fullValueSumOfSquares_;
    s.fullValueAverage = fullValueAverage_;
    s.fullValueRMS = fullValueRMS_;
    s.fullValueMin = fullValueMin_;
    s.fullValueMax = fullValueMax_;
    s.fullValueRate = fullValueRate_;
    s.fullDuration = fullDuration_;
    
    s.recentSampleCount = recentSampleCount_;
    s.recentSampleRate = recentSampleRate_;
    s.recentValueSum = recentValueSum_;
    s.recentValueSumOfSquares = recentValueSumOfSquares_;
    s.recentValueAverage = recentValueAverage_;
    s.recentValueRMS = recentValueRMS_;
    s.recentValueMin = recentValueMin_;
    s.recentValueMax = recentValueMax_;
    s.recentValueRate = recentValueRate_;
    s.recentDuration = recentDuration_;
    
    s.recentBinnedSampleCounts.resize(binCount_);
    s.recentBinnedValueSums.resize(binCount_);
    s.recentBinnedDurations.resize(binCount_);
    s.recentBinnedSnapshotTimes.resize(binCount_);
    uint32_t sourceBinId = workingBinId_;
    for (uint32_t idx = 0; idx < binCount_; ++idx) {
      if (sourceBinId >= binCount_) {sourceBinId = 0;}
      s.recentBinnedSampleCounts[idx] = binSampleCount_[sourceBinId];
      s.recentBinnedValueSums[idx] = binValueSum_[sourceBinId];
      s.recentBinnedDurations[idx] = binDuration_[sourceBinId];
      s.recentBinnedSnapshotTimes[idx] = binSnapshotTime_[sourceBinId];
      ++sourceBinId;
    }
    
    s.lastSampleValue = lastLatchedSampleValue_;
    s.lastValueRate = lastLatchedValueRate_;
    s.enabled = enabled_;
  }
  
} // namespace stor


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
