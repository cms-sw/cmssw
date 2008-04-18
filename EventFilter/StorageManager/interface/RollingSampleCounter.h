#ifndef STOR_ROLLING_SAMPLE_COUNTER_H
#define STOR_ROLLING_SAMPLE_COUNTER_H

/**
 * This class keeps track of a sum of values over a fixed number of samples.
 * After a specified number of samples are added to the counter, old values
 * are removed from the counter to make room for the new ones.
 *
 * $Id: RollingSampleCounter.h,v 1.1 2008/04/14 15:42:28 biery Exp $
 */

#include "EventFilter/StorageManager/interface/BaseCounter.h"
#include "boost/shared_ptr.hpp"
#include "boost/thread/recursive_mutex.hpp"
#include <vector>

namespace stor
{
  class RollingSampleCounter : public BaseCounter
  {

   public:

    enum AccumulationStyle { INCLUDE_SAMPLES_AFTER_BINNING = 0,
                             INCLUDE_SAMPLES_IMMEDIATELY = 1 };

    RollingSampleCounter(int windowSize = -1, int binSize = -1,
                         int validSubWindowSize = -1,
                 AccumulationStyle style = INCLUDE_SAMPLES_AFTER_BINNING);

    void addSample(double value = 1.0, double currentTime = getCurrentTime());
    bool hasValidResult();
    int getSampleCount();
    double getSampleRate(double currentTime = 0.0);
    double getValueSum();
    double getValueAverage();
    double getValueRate();
    double getDuration(double currentTime = 0.0);

    void dumpData(std::ostream& outStream);

   private:

    void shuffleBins(long long sampleCount);
    long long getBinId(long long sampleCount);

    AccumulationStyle accumStyle_;

    int windowSize_;
    int binSize_;
    int binCount_;
    int validBinCount_;

    long long sampleCount_;
    long long processedBinCount_;

    double workingBinSum_;
    double workingBinStartTime_;
    long long workingBinId_;

    double currentTotal_;

    boost::shared_ptr< std::vector<double> > binStartTimes_;
    boost::shared_ptr< std::vector<double> > binStopTimes_;
    boost::shared_ptr< std::vector<double> > binContents_;

    boost::recursive_mutex dataMutex_;

  };
}

#endif
