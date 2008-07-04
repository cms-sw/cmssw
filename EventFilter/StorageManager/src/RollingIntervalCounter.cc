/**
 * $Id: RollingIntervalCounter.cc,v 1.19 2008/03/03 20:09:37 biery Exp $
 */

#include "EventFilter/StorageManager/interface/RollingIntervalCounter.h"

using namespace stor;

/**
 * Constructor.  All sizes should be specified in units of seconds.
 */
RollingIntervalCounter::
RollingIntervalCounter(double timeWindowSize, double timeBinSize,
                       double validSubWindowSize, AccumulationStyle style):
  accumStyle_(style),
  startTime_(0.0),
  processedBinCount_(0),
  workingBinSum_(0.0),
  workingBinSampleCount_(0),
  workingBinId_(-1),
  currentTotal_(0.0),
  currentSampleCount_(0)
{
  // provide a reasonable default for the window size, if needed
  if (timeWindowSize <= 0.0) {
    this->windowSize_ = 180.0;  // 3 minutes
  }
  else {
    this->windowSize_ = timeWindowSize;
  }

  // provide a reasonable default for the bin size, if needed
  if (timeBinSize <= 0.0) {
    this->binSize_ = this->windowSize_;
  }
  else {
    this->binSize_ = timeBinSize;
  }

  // determine the bin count from the window and bin sizes
  if (this->windowSize_ > 0.0 && this->binSize_ > 0.0) {
    this->binCount_ = (int) (0.5 + (this->windowSize_ / this->binSize_));
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
  this->validBinCount_ = (int) (0.5 + (validSubWindowSize / this->binSize_));
  if (this->validBinCount_ <= 0) {
    this->validBinCount_ = 1;
  }

  // if working with data immediately, assume the first bin is good
  if (this->accumStyle_ == INCLUDE_SAMPLES_IMMEDIATELY) {
    processedBinCount_ = 1;
  }

  // initialize times and bins
  this->binContents_.reset(new std::vector<double>);
  this->binSamples_.reset(new std::vector<unsigned int>);
  for (int idx = 0; idx < this->binCount_; ++idx) {
    this->binContents_->push_back(0.0);
    this->binSamples_->push_back(0);
  }
}

/**
 * Adds the specified sample value to the counter instance.
 */
void RollingIntervalCounter::addSample(double value, double currentTime)
{
  boost::recursive_mutex::scoped_lock sl(dataMutex_);

  // initialize the working bin, if needed
  if (workingBinId_ < 0) {
    workingBinId_ = getBinId(currentTime);
  }

  // initialize the start time, if needed
  if (startTime_ <= 0.0) {
    startTime_ = currentTime;
  }

  // shuffle the bins so that they correctly reflect the sample window
  shuffleBins(currentTime);

  // add the new sample to either the working sum or the current
  // sum depending on the accumulation style
  if (accumStyle_ == INCLUDE_SAMPLES_AFTER_BINNING) {
    workingBinSum_ += value;
    ++workingBinSampleCount_;
  }
  else {
    int binIndex = (int) (workingBinId_ % binCount_);
    (*binContents_)[binIndex] += value;
    currentTotal_ += value;
    (*binSamples_)[binIndex] += 1;
    ++currentSampleCount_;
  }
}

/**
 * Tests if the counter has a valid result (enough time has passed).
 */
bool RollingIntervalCounter::hasValidResult(double currentTime)
{
  boost::recursive_mutex::scoped_lock sl(dataMutex_);
  shuffleBins(currentTime);

  return (processedBinCount_ >= validBinCount_);
}

/**
 * Returns the number of samples stored in the counter.  This
 * value corresponds to only those samples that are current part of the
 * time window for the counter.
 */
unsigned int RollingIntervalCounter::getSampleCount(double currentTime)
{
  boost::recursive_mutex::scoped_lock sl(dataMutex_);
  shuffleBins(currentTime);

  return currentSampleCount_;
}

/**
 * Returns the rate of samples stored in the counter
 * (number of samples divided by duration) or 0.0 if no samples have
 * been stored.  The units of the return value are samples per second.
 */
double RollingIntervalCounter::getSampleRate(double currentTime)
{
  boost::recursive_mutex::scoped_lock sl(dataMutex_);
  shuffleBins(currentTime);

  double duration = getDuration();
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
double RollingIntervalCounter::getValueSum(double currentTime)
{
  boost::recursive_mutex::scoped_lock sl(dataMutex_);
  shuffleBins(currentTime);

  return currentTotal_;
}

/**
 * Returns the average value of the samples that have been stored in
 * the counter or 0.0 if no samples have been added.
 */
double RollingIntervalCounter::getValueAverage(double currentTime)
{
  boost::recursive_mutex::scoped_lock sl(dataMutex_);
  shuffleBins(currentTime);

  if (getSampleCount() > 0) {
    return (currentTotal_ / ((double) getSampleCount()));
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
double RollingIntervalCounter::getValueRate(double currentTime)
{
  boost::recursive_mutex::scoped_lock sl(dataMutex_);
  shuffleBins(currentTime);

  double duration = getDuration();
  if (duration > 0.0) {
    return (currentTotal_ / duration);
  }
  else {
    return 0.0;
  }
}

/**
 * Returns the amount of time (in seconds) that this counter has
 * data for.  This time should max out at the timeWindowSize specified
 * in the constructor.
 */
double RollingIntervalCounter::getDuration(double currentTime)
{
  boost::recursive_mutex::scoped_lock sl(dataMutex_);
  shuffleBins(currentTime);

  // include a lastTime_ for immediate mode??

  if (! hasValidResult()) {
    return 0.0;
  }
  else if (processedBinCount_ > binCount_) {
    return (((double) binCount_) * binSize_);
  }
  else {
    double val1 = ((double) processedBinCount_) * binSize_;
    double val2 = (workingBinId_ * binSize_) - startTime_;
    if (accumStyle_ == INCLUDE_SAMPLES_IMMEDIATELY) {
      val2 = currentTime - startTime_;
    }
    //std::cout << "Val1 = " << val1 << ", val2 = " << val2 << std::endl;
    if (val1 < val2) {return val1;}
    else {return val2;}
  }
}

/**
 * Dumps the contents of the counter to the specified output stream.
 */
void RollingIntervalCounter::dumpData(std::ostream& outStream)
{
  outStream << "RollingIntervalCounter 0x" << std::hex
            << ((int) this) << std::dec << std::endl;
  char nowString[32];
  sprintf(nowString, "%16.4f", getCurrentTime());
  outStream << "  Now = " << nowString << std::endl;
  outStream << "  Window size = " << windowSize_ << std::endl;
  outStream << "  Bin size = " << binSize_ << std::endl;
  outStream << "  Processed bin count = " << processedBinCount_ << std::endl;
  outStream << "  Working index = "
            << ((int) (workingBinId_ % binCount_)) << std::endl;
  outStream << "  Working value = " << workingBinSum_ << std::endl;
  outStream << "  Current total = " << currentTotal_ << std::endl;
  outStream << "  Working sample count = "
            << workingBinSampleCount_ << std::endl;
  outStream << "  Current sample count = " << currentSampleCount_ << std::endl;

  char binString[200];
  for (int idx = 0; idx < binCount_; idx++) {
    sprintf(binString,
            "    bin %2d, value %10.2f, sampleCount %10d",
            idx, (*binContents_)[idx], (*binSamples_)[idx]);
    outStream << binString << std::endl;
  }
}

/**
 * Modifies the internal list of bins so that it correctly reflects
 * the current window.
 */
void RollingIntervalCounter::shuffleBins(double currentTime)
{
  // don't start shuffling until the first sample has been added
  if (workingBinId_ < 0) {
    return;
  }

  // determine the current time bin
  long long currentTimeId = getBinId(currentTime);
  //std::cout << "RollingIntervalCounter 0x" << std::hex
  //          << ((int) this) << std::dec << std::endl;
  //std::cout << "Shuffle: " << currentTimeId
  //          << ", " << workingBinId_ << std::endl;

  // if we're still working on the current time bin, no shuffling is needed
  if (currentTimeId == workingBinId_) {
    return;
  }

  // clear out entries in the list whose time has passed
  long long firstIdToClear = workingBinId_;
  long long lastIdToClear = currentTimeId - 1;
  if (accumStyle_ == INCLUDE_SAMPLES_IMMEDIATELY) {
    ++firstIdToClear;
    ++lastIdToClear;
  }
  for (long long idx = firstIdToClear; idx <= lastIdToClear; ++idx) {
    int binIndex = (int) (idx % binCount_);
    currentTotal_ -= (*binContents_)[binIndex];
    (*binContents_)[binIndex] = 0.0;
    currentSampleCount_ -= (*binSamples_)[binIndex];
    (*binSamples_)[binIndex] = 0;
    ++processedBinCount_;
  }

  // move the working bin value into the list, if needed
  if (accumStyle_ == INCLUDE_SAMPLES_AFTER_BINNING) {
    int binIndex = (int) (workingBinId_ % binCount_);
    (*binContents_)[binIndex] = workingBinSum_;
    currentTotal_ += workingBinSum_;
    workingBinSum_ = 0.0;
    (*binSamples_)[binIndex] = workingBinSampleCount_;
    currentSampleCount_ += workingBinSampleCount_;
    workingBinSampleCount_ = 0;
  }

  // update the working bin ID to the current time bin
  workingBinId_ = currentTimeId;
}

/**
 * Calculates the bin ID for the given timestamp.
 */
long long RollingIntervalCounter::getBinId(double currentTime)
{
  return (long long) (currentTime / binSize_);
}
