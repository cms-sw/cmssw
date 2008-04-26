#ifndef STOR_FOREVER_COUNTER_H
#define STOR_FOREVER_COUNTER_H

/**
 * This class keeps track of a sum of values over an infinite interval
 * (forever).  It provides functionality to track and report rates and
 * averages which is the additional value over a simple integer or double.
 *
 * $Id: ForeverCounter.h,v 1.12 2008/03/03 20:09:36 biery Exp $
 */

#include "EventFilter/StorageManager/interface/BaseCounter.h"
#include "boost/thread/recursive_mutex.hpp"

namespace stor
{
  class ForeverCounter : public BaseCounter
  {

   public:

    ForeverCounter();

    void addSample(double value = 1.0);
    long long getSampleCount();
    double getSampleRate(double currentTime = getCurrentTime());
    double getValueSum();
    double getValueAverage();
    double getValueRate(double currentTime = getCurrentTime());
    double getDuration(double currentTime = getCurrentTime());

   private:

    long long sampleCount_;
    double startTime_;
    double currentTotal_;

    boost::recursive_mutex dataMutex_;

  };
}

#endif
