#ifndef STOR_ROLLING_INTERVAL_COUNTER_H
#define STOR_ROLLING_INTERVAL_COUNTER_H

/**
 * This class keeps track of a sum of values over a fixed period of time.
 * As time passes, values that are no longer included in the
 * time window are dropped from the counter.
 *
 * $Id: RollingIntervalCounter.h,v 1.12 2008/03/03 20:09:36 biery Exp $
 */

#include "EventFilter/StorageManager/interface/BaseCounter.h"
#include "boost/shared_ptr.hpp"
#include "boost/thread/recursive_mutex.hpp"
#include <vector>

namespace stor
{
  class RollingIntervalCounter : public BaseCounter
  {

   public:

    enum AccumulationStyle { INCLUDE_SAMPLES_AFTER_BINNING = 0,
                             INCLUDE_SAMPLES_IMMEDIATELY = 1 };

    RollingIntervalCounter(double timeWindowSize = -1.0,
                           double timeBinSize = -1.0,
                           double validSubWindowSize = -1.0,
                   AccumulationStyle style = INCLUDE_SAMPLES_AFTER_BINNING);

    void addSample(double value = 1.0, double currentTime = getCurrentTime());
    bool hasValidResult(double currentTime = getCurrentTime());
    unsigned int getSampleCount(double currentTime = getCurrentTime());
    double getSampleRate(double currentTime = getCurrentTime());
    double getValueSum(double currentTime = getCurrentTime());
    double getValueAverage(double currentTime = getCurrentTime());
    double getValueRate(double currentTime = getCurrentTime());
    double getDuration(double currentTime = getCurrentTime());

    void dumpData(std::ostream& outStream);

   private:

    void shuffleBins(double currentTime);
    long long getBinId(double currentTime);

    AccumulationStyle accumStyle_;

    double windowSize_;
    double binSize_;
    int binCount_;
    int validBinCount_;

    double startTime_;
    long long processedBinCount_;

    double workingBinSum_;
    unsigned int workingBinSampleCount_;
    long long workingBinId_;

    double currentTotal_;
    unsigned int currentSampleCount_;

    boost::shared_ptr< std::vector<double> > binContents_;
    boost::shared_ptr< std::vector<unsigned int> > binSamples_;

    boost::recursive_mutex dataMutex_;

  };
}

#endif
