#ifndef STOR_FOREVER_AVERAGE_COUNTER_H
#define STOR_FOREVER_AVERAGE_COUNTER_H

/**
 * This class keeps track of a the average of a value and also the
 * the minimum and maximum values, and the rms
 *
 * $Id$
 */

#include "boost/thread/recursive_mutex.hpp"

namespace stor
{
  class ForeverAverageCounter
  {

   public:

    ForeverAverageCounter();

    void addSample(double value);
    void reset();
    long long getSampleCount();
    double getValueSum();
    double getValueSumSquares();
    double getValueAverage();
    double getValueRMS();
    double getValueMin();
    double getValueMax();

   private:

    long long sampleCount_;
    double currentTotal_;
    double currentTotalSquares_;
    double currentMin_;
    double currentMax_;

    boost::recursive_mutex dataMutex_;

  };
}

#endif
