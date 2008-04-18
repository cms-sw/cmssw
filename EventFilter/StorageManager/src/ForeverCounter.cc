/**
 * $Id: ForeverCounter.cc,v 1.19 2008/03/03 20:09:37 biery Exp $
 */

#include "EventFilter/StorageManager/interface/ForeverCounter.h"

using namespace stor;

/**
 * Constructor.
 */
ForeverCounter::ForeverCounter():
  sampleCount_(0),
  startTime_(0.0),
  currentTotal_(0.0)
{
}

/**
 * Adds the specified sample value to the counter instance.
 */
void ForeverCounter::addSample(double value)
{
  boost::recursive_mutex::scoped_lock sl(dataMutex_);

  if (sampleCount_ == 0) {
    startTime_ = getCurrentTime();
  }

  currentTotal_ += value;
  ++sampleCount_;
}

/**
 * Returns the number of samples stored in the counter.
 */
long long ForeverCounter::getSampleCount()
{
  boost::recursive_mutex::scoped_lock sl(dataMutex_);

  return sampleCount_;
}

/**
 * Returns the rate of samples stored in the counter
 * (number of samples divided by duration).  The units of the
 * return value are samples per second.
 */
double ForeverCounter::getSampleRate(double currentTime)
{
  boost::recursive_mutex::scoped_lock sl(dataMutex_);

  double duration = getDuration(currentTime);
  if (duration > 0.0) {
    return ((double) sampleCount_) / duration;
  }
  else {
    return 0.0;
  }
}

/**
 * Returns the sum of all sample values stored in the counter.
 */
double ForeverCounter::getValueSum() {
  boost::recursive_mutex::scoped_lock sl(dataMutex_);

  return currentTotal_;
}

/**
 * Returns the average value of the samples that have been stored in
 * the counter or 0.0 if no samples have been added.
 */
double ForeverCounter::getValueAverage()
{
  boost::recursive_mutex::scoped_lock sl(dataMutex_);

  if (sampleCount_ > 0) {
    return ((double) currentTotal_) / ((double) sampleCount_);
  }
  else {
    return 0.0;
  }
}

/**
 * Returns the sample value rate (the sum of all sample values stored
 * in the counter divided by the duration) or 0.0 if no samples have been
 * stored.  The units of the return
 * value are [the units of the sample values] per second (e.g. MByte/sec).
 */
double ForeverCounter::getValueRate(double currentTime)
{
  boost::recursive_mutex::scoped_lock sl(dataMutex_);

  double duration = getDuration(currentTime);
  if (duration > 0.0) {
    return ((double) currentTotal_) / duration;
  }
  else {
    return 0.0;
  }
}

/**
 * Returns the amount of time (seconds) that this counter instance has
 * been processing values starting with the time of the first sample
 * or 0.0 if no samples have been stored in the counter.
 */
double ForeverCounter::getDuration(double currentTime)
{
  boost::recursive_mutex::scoped_lock sl(dataMutex_);

  if (sampleCount_ == 0) {
    return 0.0;
  }
  else {
    return (currentTime - startTime_);
  }
}
