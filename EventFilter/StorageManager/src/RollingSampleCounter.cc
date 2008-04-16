/**
 * $Id: RollingSampleCounter.cc,v 1.1 2008/04/14 15:42:28 biery Exp $
 */

#include "EventFilter/StorageManager/interface/RollingSampleCounter.h"

using namespace stor;

/**
 * Constructor.
 */
RollingSampleCounter::
RollingSampleCounter(int windowSize, int binSize, int validSubWindowSize,
                     AccumulationStyle style):
  accumStyle_(style),
  sampleCount_(0),
  processedBinCount_(0),
  workingBinSum_(0.0),
  workingBinStartTime_(0.0),
  currentTotal_(0.0)
{
  // provide a reasonable default for the window size, if needed
  if (windowSize <= 0) {
    this->windowSize_ = 100;  // 100 samples
  }
  else {
    this->windowSize_ = windowSize;
  }

  // provide a reasonable default for the bin size, if needed
  if (binSize <= 0) {
    this->binSize_ = this->windowSize_;
  }
  else {
    this->binSize_ = binSize;
  }

  // determine the bin count from the window and bin sizes
  if (this->windowSize_ > 0 && this->binSize_ > 0) {
    this->binCount_ =
      (int) (0.5 + ((double) this->windowSize_ / (double) this->binSize_));
    if (this->binCount_ < 1) {
      this->binCount_ = 1;
    }

    // recalculate the window size to handle rounding
    this->windowSize_ = this->binCount_ * this->binSize_;
  }
  else {
    this->binCount_ = 1;
  }

  // determine the number of bins needed for a valid result
  this->validBinCount_ =
    (int) (0.5 + ((double) validSubWindowSize / (double) this->binSize_));
  if (this->validBinCount_ <= 0) {
    this->validBinCount_ = 1;
  }

  // if working with data immediately, assume the first bin is good
  if (this->accumStyle_ == INCLUDE_SAMPLES_IMMEDIATELY) {
    processedBinCount_ = 1;
  }

  // initialize times and bins
  this->workingBinId_ = getBinId(this->sampleCount_);
  this->binStartTimes_.reset(new std::vector<double>);
  this->binStopTimes_.reset(new std::vector<double>);
  this->binContents_.reset(new std::vector<double>);
  for (int idx = 0; idx < this->binCount_; ++idx) {
    this->binStartTimes_->push_back(0.0);
    this->binStopTimes_->push_back(0.0);
    this->binContents_->push_back(0.0);
  }
}

/**
 * Adds the specified sample value to the counter instance.
 */
void RollingSampleCounter::addSample(double value, double currentTime)
{
  boost::recursive_mutex::scoped_lock sl(dataMutex_);

  // shuffle the bins so that they correctly reflect the sample window
  shuffleBins(sampleCount_);
  ++sampleCount_;

  // add the new sample to either the working sum or the current
  // sum depending on the accumulation style
  if (accumStyle_ == INCLUDE_SAMPLES_AFTER_BINNING) {
    workingBinSum_ += value;
    if (sampleCount_ == 1) {
      workingBinStartTime_ = currentTime;
    }
  }
  else {
    int binIndex = (int) (workingBinId_ % binCount_);
    (*binContents_)[binIndex] += value;
    currentTotal_ += value;
    if (sampleCount_ == 1) {
      (*binStartTimes_)[binIndex] = currentTime;
    }
    (*binStopTimes_)[binIndex] = currentTime;
  }
}

/**
 * Tests if the counter has a valid result (enough samples have
 * been stored).
 */
bool RollingSampleCounter::hasValidResult()
{
  boost::recursive_mutex::scoped_lock sl(dataMutex_);

  return (processedBinCount_ >= validBinCount_);
}

/**
 * Returns the number of samples stored in the counter.  This
 * value will stop increasing when the full window size is reached.
 */
int RollingSampleCounter::getSampleCount()
{
  boost::recursive_mutex::scoped_lock sl(dataMutex_);

  int result = windowSize_;
  if (sampleCount_ < windowSize_) {
    result = (int) sampleCount_;
  }
  else if (accumStyle_ == INCLUDE_SAMPLES_IMMEDIATELY && binSize_ > 1) {
    // NEED TO FIX!!!
    result = 1 + (int) ((sampleCount_ - 1) % windowSize_);
  }

  return result;
}

/**
 * Returns the rate of samples stored in the counter
 * (number of samples divided by duration) or 0.0 if no samples have
 * been stored.  The units of the return value are samples per second.
 */
double RollingSampleCounter::getSampleRate(double currentTime)
{
  boost::recursive_mutex::scoped_lock sl(dataMutex_);

  double duration = getDuration(currentTime);
  if (duration > 0.0) {
    return ((double) getSampleCount()) / duration;
  }
  else {
    return 0.0;
  }
}

/**
 * Returns the sum of all sample values stored in the counter.
 */
double RollingSampleCounter::getValueSum() {
  boost::recursive_mutex::scoped_lock sl(dataMutex_);

  return currentTotal_;
}

/**
 * Returns the average value of the samples that have been stored in
 * the counter or 0.0 if no samples have been added.
 */
double RollingSampleCounter::getValueAverage()
{
  boost::recursive_mutex::scoped_lock sl(dataMutex_);

  if (getSampleCount() > 0) {
    return ((double) currentTotal_) / ((double) getSampleCount());
  }
  else {
    return 0.0;
  }
}

/**
 * Returns the sample value rate (the sum of all sample values stored
 * in the counter divided by the duration) or 0.0 if no samples have
 * been stored.  The units of the return
 * value are [the units of the sample values] per second (e.g. MByte/sec).
 */
double RollingSampleCounter::getValueRate()
{
  boost::recursive_mutex::scoped_lock sl(dataMutex_);

  double duration = getDuration();
  if (duration > 0.0) {
    return ((double) currentTotal_) / duration;
  }
  else {
    return 0.0;
  }
}

/**
 * Returns the amount of time (in seconds) that has elapsed between the
 * time that the earliest sample in the counter was stored and the
 * time that the latest was stored.  Since old samples are discarded
 * to make room for new ones, this does *not* correspond to the time
 * since the first sample was added.
 * <br/>
 * 15-Apr-2008, KAB: added time argument.  If it is non-zero, then the
 * result of this method is the time between the earliest sample and
 * the specified time.
 */
double RollingSampleCounter::getDuration(double currentTime)
{
  boost::recursive_mutex::scoped_lock sl(dataMutex_);

  //if (processedBinCount_ < 1 || ! hasValidResult()) {
  if (! hasValidResult()) {
    return 0.0;
  }
  else {
    int actualBinCount = binCount_;
    if (processedBinCount_ < binCount_) {
      actualBinCount = (int) processedBinCount_;
    }
    long long startBinId = workingBinId_ - actualBinCount;
    long long stopBinId = workingBinId_ - 1;
    if (accumStyle_ == RollingSampleCounter::INCLUDE_SAMPLES_IMMEDIATELY) {
      ++startBinId;
      ++stopBinId;
    }

    int startBinIndex = (int) (startBinId % binCount_);
    int stopBinIndex = (int) (stopBinId % binCount_);

    if ((*binStopTimes_)[stopBinIndex] > 0.0 &&
        (*binStartTimes_)[startBinIndex] > 0.0) {
      if (currentTime <= 0.0) {
        return ((*binStopTimes_)[stopBinIndex] -
                (*binStartTimes_)[startBinIndex]);
      }
      else {
        return (currentTime - (*binStartTimes_)[startBinIndex]);
      }
    }
    else {
      return 0.0;
    }
  }
}

/**
 * Dumps the contents of the counter to the specified output stream.
 */
void RollingSampleCounter::dumpData(std::ostream& outStream)
{
  outStream << "RollingSampleCounter 0x" << std::hex
            << ((int) this) << std::dec << std::endl;
  char nowString[32];
  sprintf(nowString, "%16.4f", getCurrentTime());
  outStream << "  Now = " << nowString << std::endl;
  outStream << "  Window size = " << windowSize_ << std::endl;
  outStream << "  Bin size = " << binSize_ << std::endl;
  outStream << "  Sample count = " << sampleCount_ << std::endl;
  outStream << "  Processed bin count = " << processedBinCount_ << std::endl;
  outStream << "  Working index = "
            << ((int) (workingBinId_ % binCount_)) << std::endl;
  outStream << "  Working value = " << workingBinSum_ << std::endl;
  outStream << "  Current total = " << currentTotal_ << std::endl;

  char binString[200];
  for (int idx = 0; idx < binCount_; idx++) {
    sprintf(binString,
            "    bin %2d, value %10.2f, startTime %13.2f, stopTime %13.2f",
            idx, (*binContents_)[idx], (*binStartTimes_)[idx],
            (*binStopTimes_)[idx]);
    outStream << binString << std::endl;
  }
}

/**
 * Modifies the internal list of bins so that it correctly reflects
 * the current window.
 */
void RollingSampleCounter::shuffleBins(long long sampleCount)
{
  // determine the current bin
  long long currentBinId = getBinId(sampleCount);
  //std::cout << "Shuffle " << workingBinId_ << " "
  //          << currentBinId << std::endl;

  // if we're still working on the current bin, no shuffling is needed
  if (currentBinId == workingBinId_) {
    return;
  }
  ++processedBinCount_;
  double now = getCurrentTime();

  // handle the different accumulation styles
  if (accumStyle_ == INCLUDE_SAMPLES_AFTER_BINNING) {

    // move the working bin value into the list
    int binIndex = (int) (workingBinId_ % binCount_);
    currentTotal_ -= (*binContents_)[binIndex];
    (*binContents_)[binIndex] = workingBinSum_;
    (*binStartTimes_)[binIndex] = workingBinStartTime_;
    (*binStopTimes_)[binIndex] = now;
    currentTotal_ += workingBinSum_;
    workingBinSum_ = 0.0;
    workingBinStartTime_ = now;
  }

  else {

    // clear out the current bin value so it's ready for new data
    int binIndex = (int) (currentBinId % binCount_);
    currentTotal_ -= (*binContents_)[binIndex];
    (*binContents_)[binIndex] = 0.0;

    int binIndex2 = (int) (workingBinId_ % binCount_);
    (*binStartTimes_)[binIndex] = (*binStopTimes_)[binIndex2];
  }

  // update the working bin ID to the current bin
  workingBinId_ = currentBinId;
}

/**
 * Calculates the bin ID for the given sample count.
 */
long long RollingSampleCounter::getBinId(long long sampleCount)
{
  return (long long) (sampleCount / (long long) binSize_);
}
