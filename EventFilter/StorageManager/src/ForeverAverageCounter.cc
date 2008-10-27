/**
 * $Id$
 */

#include "EventFilter/StorageManager/interface/ForeverAverageCounter.h"
#include <math.h>

using namespace stor;

/**
 * Constructor.
 */
ForeverAverageCounter::ForeverAverageCounter():
  sampleCount_(0),
  currentTotal_(0.0),
  currentTotalSquares_(0.0),
  currentMin_(9999999999.0),
  currentMax_(-1.0)
{
}

/**
 * Adds the specified sample value to the counter instance.
 */
void ForeverAverageCounter::addSample(double value)
{
  boost::recursive_mutex::scoped_lock sl(dataMutex_);

  currentTotal_ += value;
  currentTotalSquares_ += value*value;
  ++sampleCount_;
  if(value < currentMin_) currentMin_ = value;
  if(value > currentMax_) currentMax_ = value;
}

/**
 * Change back to beginning state
 */
void ForeverAverageCounter::reset()
{
  boost::recursive_mutex::scoped_lock sl(dataMutex_);

  sampleCount_ = 0;
  currentTotal_ = 0.0;
  currentTotalSquares_ = 0.0;
  currentMin_ = 9999999999.0;
  currentMax_ = -1.0;
}

/**
 * Returns the number of samples stored in the counter.
 */
long long ForeverAverageCounter::getSampleCount()
{
  boost::recursive_mutex::scoped_lock sl(dataMutex_);

  return sampleCount_;
}

/**
 * Returns the sum of all sample values stored in the counter.
 */
double ForeverAverageCounter::getValueSum() {
  boost::recursive_mutex::scoped_lock sl(dataMutex_);

  return currentTotal_;
}

/**
 * Returns the sum of squares of all sample values stored in the counter.
 */
double ForeverAverageCounter::getValueSumSquares() {
  boost::recursive_mutex::scoped_lock sl(dataMutex_);

  return currentTotalSquares_;
}

/**
 * Returns the average value of the samples that have been stored in
 * the counter or 0.0 if no samples have been added.
 */
double ForeverAverageCounter::getValueAverage()
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
 * Returns the RMS of the sample values that have been stored in
 * the counter or 0.0 if no samples have been added.
 */
double ForeverAverageCounter::getValueRMS()
{
  boost::recursive_mutex::scoped_lock sl(dataMutex_);

  if (sampleCount_ > 0) {
    double ave_squares = (currentTotalSquares_ / (double) sampleCount_);
    double ave = (currentTotal_ / (double) sampleCount_);
    double sig_squared = ave_squares - ave*ave;
    if(sig_squared > 0.0) 
       return sqrt(sig_squared);
    else
       return 0.0;
  }
  else {
    return 0.0;
  }
}

/**
 * Returns the minimum value of all sample values stored in the counter.
 */
double ForeverAverageCounter::getValueMin() {
  boost::recursive_mutex::scoped_lock sl(dataMutex_);

  return currentMin_;
}

/**
 * Returns the maximum value of all sample values stored in the counter.
 */
double ForeverAverageCounter::getValueMax() {
  boost::recursive_mutex::scoped_lock sl(dataMutex_);

  return currentMax_;
}

