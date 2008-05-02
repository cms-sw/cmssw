/**
 * $Id: RateLimiter.cc,v 1.19 2008/03/03 20:09:37 biery Exp $
 */

#include "EventFilter/StorageManager/interface/RateLimiter.h"
#include "EventFilter/StorageManager/interface/BaseCounter.h"

using namespace stor;

/**
 * Constructor.  The event rate should be specified in units of
 * events per second.  The data rate should be specified in units
 * of MByte per second.
 */
RateLimiter::RateLimiter(double maxEventRate, double maxDataRate)
{
  this->maxEventRate_ = maxEventRate;
  this->maxDataRate_ = maxDataRate;
}

/**
 * Adds the specified consumer to the rate limiter instance.
 */
void RateLimiter::addConsumer(uint32 consumerId)
{
  boost::mutex::scoped_lock sl(dataMutex_);

  // add the consumer to our internal list
  consumerList_.push_back(consumerId);

  // create a rate analyzer for this consumer
  boost::shared_ptr<RollingIntervalCounter>
    analyzer(new RollingIntervalCounter(180.0, 5.0, 10.0));
  dataRateTable_[consumerId] = analyzer;

  generator_.reset(new boost::uniform_01<boost::mt19937>(baseGenerator_));
}

/**
 * Removes the specified consumer from the rate limiter instance.
 */
void RateLimiter::removeConsumer(uint32 consumerId)
{
  boost::mutex::scoped_lock sl(dataMutex_);

  // discard the rate analyzer for the consumer
  dataRateTable_.erase(consumerId);

  // remove the consumer from our internal list
  std::vector<uint32>::iterator vecIter =
    std::find(consumerList_.begin(), consumerList_.end(), consumerId);
  consumerList_.erase(vecIter);
}

/**
 * Fetches a list of consumers which are allowed to receive an
 * event with the specified data size based on the input list of
 * candidate consumers.
 *
 * Note that invalid consumer IDs in the candidate list are silently ignored.
 */
//uint32 debugCounter = 0;
std::vector<uint32> RateLimiter::
getAllowedConsumersFromList(double dataSize,
                            const std::vector<uint32>& candidateList)
{
  boost::mutex::scoped_lock sl(dataMutex_);
  //++debugCounter;

  // initialization
  boost::shared_ptr<RollingIntervalCounter> rateAnalyzer;
  std::vector<uint32> allowedList;
  double now = BaseCounter::getCurrentTime();

  // nothing to do if there are no candidate consumers
  if (candidateList.size() == 0) {
    return allowedList;
  }

  // update the rate calculations for each of the candidate consumers
  for (uint32 idx = 0; idx < candidateList.size(); ++idx) {
    uint32 consumerId = candidateList[idx];
    rateAnalyzer = dataRateTable_[consumerId];
    if (rateAnalyzer.get() != 0) {
      rateAnalyzer->addSample(dataSize, now);
    }
  }

  // delare the lists of raw rates and target prescales that
  // will be used while running the fair-share algorithm
  uint32 consumerCount = consumerList_.size();
  std::vector<double> rawEventRates;
  std::vector<double> rawDataRates;
  std::vector<double> eventPrescales;
  std::vector<double> dataPrescales;

  // fetch the current event and data rates for all consumers
  for (uint32 idx = 0; idx < consumerCount; ++idx) {
    uint32 consumerId = consumerList_[idx];
    rateAnalyzer = dataRateTable_[consumerId];

    double eventRate = rateAnalyzer->getSampleRate(now);
    //std::cout << "Event rate = " << eventRate << std::endl;
    if (eventRate > 0.0) {
      // event rate is reasonable, so use it
      rawEventRates.push_back(eventRate);
    }
    //else if (eventRate < 0.0) {
    else if (! rateAnalyzer->hasValidResult()) {
      // event rate has not yet been determined, use a suitable large value
      // as a conservative guess
      rawEventRates.push_back(10.0 * maxEventRate_ / consumerCount);
    }
    else {
      // the event rate is so low that it appears to have fallen to zero
      // (we use a suitable small value here to avoid having zero in the
      // rawEventRates list)
      rawEventRates.push_back(0.1 * maxEventRate_ / consumerCount);
    }

    double dataRate = rateAnalyzer->getValueRate(now);
    if (dataRate > 0.0) {
      // data rate is reasonable, so use it
      rawDataRates.push_back(dataRate);
    }
    //else if (dataRate < 0.0) {
    else if (! rateAnalyzer->hasValidResult()) {
      // data rate has not yet been determined, use a suitable large value
      // as a conservative guess
      rawDataRates.push_back(10.0 * maxDataRate_ / consumerCount);
    }
    else {
      // the data rate is so low that it appears to have fallen to zero
      // (we use a suitable small value here to avoid having zero in the
      // rawDataRates list)
      rawDataRates.push_back(0.1 * maxDataRate_ / consumerCount);
    }

    //if ((debugCounter % 500) == 0) {
    //  std::cout << "Consumer ID " << consumerId
    //            << ", event rate = " << rawEventRates.back()
    //            << " (" << eventRate << ")"
    //            << ", data rate = " << rawDataRates.back()
    //            << " (" << dataRate << ")" << std::endl;
    //}
  }

  // run the fair-share algorithm from both event and data points of view
  determineTargetPrescales(maxEventRate_, rawEventRates, eventPrescales);
  determineTargetPrescales(maxDataRate_, rawDataRates, dataPrescales);

  // pick the appropriate prescale for each consumer
  std::vector<double> overallPrescales;
  std::vector<double> minimumPrescales;
  int eventLimitCount = 0;
  int dataLimitCount = 0;
  for (uint32 idx = 0; idx < consumerCount; ++idx) {
    if (eventPrescales[idx] > dataPrescales[idx]) {
      overallPrescales.push_back(eventPrescales[idx]);
      minimumPrescales.push_back(dataPrescales[idx]);
      ++eventLimitCount;
    }
    else if (dataPrescales[idx] > eventPrescales[idx]) {
      overallPrescales.push_back(dataPrescales[idx]);
      minimumPrescales.push_back(eventPrescales[idx]);
      ++dataLimitCount;
    }
    else {
      overallPrescales.push_back(eventPrescales[idx]);
      minimumPrescales.push_back(dataPrescales[idx]);
    }
  }

  //if ((debugCounter % 500) == 0) {
  //  for (uint32 idx = 0; idx < consumerCount; ++idx) {
  //    uint32 consumerId = consumerList_[idx];
  //    double psValue = overallPrescales[idx];
  //    std::cout << "Consumer ID = " << consumerId
  //              << ", prescale = " << psValue
  //              << ", (min = " << minimumPrescales[idx] << ")" << std::endl;
  //  }
  //}

  // if prescales have been imposed based on both event rate and data rates,
  // then it is likely that we can loosen the prescales a bit
  if (eventLimitCount > 0 && dataLimitCount > 0) {
    loosenPrescales(rawEventRates, maxEventRate_,
                    rawDataRates, maxDataRate_,
                    overallPrescales, minimumPrescales);

    //if ((debugCounter % 500) == 0) {
    //  for (uint32 idx = 0; idx < consumerCount; ++idx) {
    //    uint32 consumerId = consumerList_[idx];
    //    double psValue = overallPrescales[idx];
    //    std::cout << "Consumer ID = " << consumerId
    //              << ", loosened prescale = " << psValue << std::endl;
    //  }
    //}
  }

  // loop over the candidate consumers and test if we're willing to let
  // them have this event
  for (uint32 idx = 0; idx < candidateList.size(); ++idx) {
    uint32 consumerId = candidateList[idx];
    for (uint32 jdx = 0; jdx < consumerCount; ++jdx) {
      if (consumerList_[jdx] == consumerId) {
        double psValue = overallPrescales[jdx];
        if (psValue <= 1.0) {
          allowedList.push_back(consumerId);
        }
        else {
          double instantRatio = 1.0 / psValue;
          double randValue = (*generator_)();
          if (randValue < instantRatio) {
            allowedList.push_back(consumerId);
          }
        }
        break;
      }
    }
  }

  // return the list of allowed consumers
  return allowedList;
}

/**
 * Prints information about the current set of consumers
 * to the specified output stream.
 */
void RateLimiter::dumpData(std::ostream& outStream)
{
  boost::shared_ptr<RollingIntervalCounter> rateAnalyzer;
  char nowString[32];

  outStream << "RateLimiter 0x" << std::hex
            << ((int) this) << std::dec << std::endl;
  sprintf(nowString, "%16.4f", BaseCounter::getCurrentTime());
  outStream << "  Now = " << nowString << std::endl;
  outStream << "  Consumers:" << std::endl;
  for (uint32 idx = 0; idx < consumerList_.size(); ++idx) {
    uint32 consumerId = consumerList_[idx];
    rateAnalyzer = dataRateTable_[consumerId];
    outStream << "    ID = " << consumerId
              << ", event rate = " << rateAnalyzer->getSampleRate()
              << ", data rate = " << rateAnalyzer->getValueRate()
              << std::endl;
  }
}

/**
 * Calculates the resulting overall rate from the specified rates
 * and prescales.
 */
double RateLimiter::calcRate(std::vector<double> rates,
                             std::vector<double> prescales)
{
  double sum = 0.0;
  for (uint32 idx = 0; idx < rates.size(); ++idx) {
    sum += rates[idx] / prescales[idx];
  }
  return sum;
}

/**
 * Determines the target prescales for existing consumers based on the
 * allowed full rate and the rate at which consumers would like to
 * receive events.
 */
void RateLimiter::determineTargetPrescales(double fullRate,
                                           const std::vector<double>& rawRates,
                                           std::vector<double>& targetPrescales)
{
  uint32 rateCount = rawRates.size();

  // determine the target rates
  std::vector<double> targetRates;
  determineTargetRates(fullRate, rawRates, targetRates);
  assert(targetRates.size() == rateCount);

  // calculate the target prescales from the target rates
  targetPrescales.clear();
  for (uint32 idx = 0; idx < rateCount; ++idx) {
    if (targetRates[idx] > 0.0) {
      targetPrescales.push_back(rawRates[idx] / targetRates[idx]);
    }
    else {
      targetPrescales.push_back(1.0);
    }
  }
}

/**
 * Determines the target rates for existing consumers based on the
 * allowed full rate and the rate at which consumers would like to
 * receive events.
 */
void RateLimiter::determineTargetRates(double fullRate,
                                       const std::vector<double>& rawRates,
                                       std::vector<double>& targetRates)
{
  uint32 rateCount = rawRates.size();
  targetRates.clear();

  // easy check #1 - there is only one rate
  if (rateCount == 1) {
    targetRates.push_back(std::min(fullRate, rawRates[0]));
    return;
  }

  // easy check #2 - the sum of all rates fits within the full rate
  double sum = 0.0;
  for (uint32 idx = 0; idx < rateCount; ++idx) {
    sum += rawRates[idx];
  }
  if (sum <= fullRate) {
    targetRates.resize(rateCount);
    std::copy(rawRates.begin(), rawRates.end(), targetRates.begin());
    return;
  }

  // initialize the targetRates to negative values in preparation for
  // using the fairShareAlgo method
  for (uint32 idx = 0; idx < rateCount; ++idx) {
    targetRates.push_back(-1.0);
  }

  // run the fair share algorithm
  fairShareAlgo(fullRate, rawRates, targetRates);
}

/**
 * Fairly allocates the allowed full rate among the existing consumers
 * according to the rate at which each consumer would like to receive events.
 * The targetRates list should contain a -1.0 value for each consumer that
 * has not yet been assigned a fair share.  This method is called recursively
 * with varying full rates and varying targetRate values.
 */
void RateLimiter::fairShareAlgo(double fullRate,
                                const std::vector<double>& rawRates,
                                std::vector<double>& targetRates)
{
  uint32 rateCount = rawRates.size();

  // count the number of target values that still need to be determined
  int targetCount = 0;
  for (uint32 idx = 0; idx < rateCount; ++idx) {
    if (targetRates[idx] < 0.0) {
      ++targetCount;
    }
  }

  // sanity check
  if (targetCount == 0) {return;}

  // calculate the maximum fair share for the remaining targets
  double fairShareRate = fullRate / (double) targetCount;

  // check for input rates that are less than the fair share
  double accomodatedRate = 0.0;
  int graceCount = 0;
  for (uint32 idx = 0; idx < rateCount; ++idx) {
    if (targetRates[idx] < 0.0 && rawRates[idx] < fairShareRate) {
      targetRates[idx] = rawRates[idx];
      accomodatedRate += rawRates[idx];
      ++graceCount;
    }
  }

  // if we found rates that could be fully accomodated, iterate
  if (graceCount > 0) {
    fairShareAlgo((fullRate - accomodatedRate), rawRates, targetRates);
  }

  // otherwise, fill in the remaining target rates will the fair share amount
  else {
    for (uint32 idx = 0; idx < rateCount; ++idx) {
      if (targetRates[idx] < 0.0) {
        targetRates[idx] = fairShareRate;
      }
    }
  }
}

/**
 * This method attempts to loosen the specified prescales to allow
 * a better match between the raw consumer rates and the desired
 * overall rates.
 */
void RateLimiter::loosenPrescales(const std::vector<double>& rawRates1,
                                  double fullRate1,
                                  const std::vector<double>& rawRates2,
                                  double fullRate2,
                                  std::vector<double>& prescales,
                                  const std::vector<double>& minPrescales)
{
  // initialization
  double fom;
  double baseFOM = calcFigureOfMerit(rawRates1, fullRate1,
                                     rawRates2, fullRate2,
                                     prescales);

  // calculate how much a predefined change in each prescale makes to
  // the figure-of-merit
  std::vector<double> fomImpact;
  fomImpact.resize(prescales.size());
  for (uint32 idx = 0; idx < prescales.size(); ++idx) {
    if (prescales[idx] > 1.0) {
      prescales[idx] *= 0.99;
      fom = calcFigureOfMerit(rawRates1, fullRate1,
                              rawRates2, fullRate2,
                              prescales);
      if ((baseFOM - fom) >= 0.0) {
        fomImpact[idx] = baseFOM - fom;
      }
      else {
        fomImpact[idx] = fom - baseFOM;
      }
      prescales[idx] /= 0.99;
    }
    else {
      fomImpact[idx] = 0.0;
    }
  }
  //if ((debugCounter % 500) == 0) {
  //  printf("Impacts: ");
  //  for (uint32 pdx = 0; pdx < prescales.size(); pdx++) {
  //    printf("%8.4f", fomImpact[pdx]);
  //  }
  //  printf("\n");
  //}

  // determine a prescale ordering based on how much each prescale can impact
  // the figure of merit (ordered from most impact to least impact)
  std::vector<int> impactOrder;
  for (uint32 idx = 0; idx < prescales.size(); ++idx) {
    double maxValue = -1.0;
    int maxIndex = -1;
    for (uint32 jdx = 0; jdx < prescales.size(); ++jdx) {
      if (fomImpact[jdx] >= 0.0 && fomImpact[jdx] > maxValue) {
        maxValue = fomImpact[jdx];
        maxIndex = jdx;
      }
    }
    fomImpact[maxIndex] = -1.0;
    impactOrder.push_back(maxIndex);
  }
  //if ((debugCounter % 500) == 0) {
  //  printf("Impact order: ");
  //  for (uint32 pdx = 0; pdx < prescales.size(); pdx++) {
  //    printf(" %d", impactOrder[pdx]);
  //  }
  //  printf("\n");
  //}

  // loop through the list of prescales (in order of decreasing impact)
  // and vary each prescale until the figure-of-merit stops improving.  
  // Once we're reached an optimum value for all prescales, we're done.
  // (A watchdog counter is included to prevent infinite loops.)
  fom = baseFOM;
  bool allDone = false;
  int watchdogCounter = -1;
  while (! allDone && ++watchdogCounter < 10) {

    // assume that this will be the last pass through the set of prescales
    allDone = true;
    for (uint32 idx = 0; idx < prescales.size(); ++idx) {

      // fetch the index of the next prescale to consider
      uint32 psIdx = impactOrder[idx];

      // skip over prescale values of 1.0 (how could we improve on that?)
      if (prescales[psIdx] > 1.0) {

        // if changing this prescale made a difference, we should keep
        // going to see if other prescale changes can improve the FOM
        // even more
        if (loosenOnePrescale(rawRates1, fullRate1, rawRates2, fullRate2,
                              prescales, psIdx, minPrescales[psIdx])) {
          allDone = false;
        }
      }
    }
  }
}

/**
 * This method loosens a specified (single) prescale value within
 * specified bounds.  Returns a flag to indicate whether loosening the
 * specified prescale helped at all.
 */
bool RateLimiter::loosenOnePrescale(const std::vector<double>& rawRates1,
                                    double fullRate1,
                                    const std::vector<double>& rawRates2,
                                    double fullRate2,
                                    std::vector<double>& prescales,
                                    uint32 psIndex, double lowBound)
{
  // initialization
  bool resetToOneDone = false;
  double baseFOM = calcFigureOfMerit(rawRates1, fullRate1,
                                     rawRates2, fullRate2,
                                     prescales);
  double fom = baseFOM;
  double lastFOM = fom * 2;
  int loopCount = 0;
  if (lowBound < 1.0) {lowBound = 1.0;}
  if (prescales[psIndex] <= 1.0) {return false;}

  // loop as long as the FOM is getting better
  // (The check here on loopCount is simply to prevent infinite loops,
  // although loopCount is used for a different purpose below.)
  while (fom < lastFOM && prescales[psIndex] >= lowBound &&
         loopCount < 1000) {
    ++loopCount;
    lastFOM = fom;

    // reduce the prescale under test
    prescales[psIndex] *= 0.99;
    if (! resetToOneDone && prescales[psIndex] < 1.0) {
      resetToOneDone = true;
      prescales[psIndex] = 1.0;
    }
    fom = calcFigureOfMerit(rawRates1, fullRate1,
                            rawRates2, fullRate2,
                            prescales);
  }

  // restore the prescale under test to the value that it had
  // when the minimum FOM was reached
  prescales[psIndex] /= 0.99;

  //if ((debugCounter % 500) == 0) {
  //  fom = calcFigureOfMerit(rawRates1, fullRate1,
  //                          rawRates2, fullRate2,
  //                          prescales);
  //  printf("FOM value = %10.4f for prescales", fom);
  //  for (uint32 pdx = 0; pdx < prescales.size(); ++pdx) {
  //    printf("%5.2f", prescales[pdx]);
  //  }
  //  double rate1 = calcRate(rawRates1, prescales);
  //  double rate2 = calcRate(rawRates2, prescales);
  //  printf("\n  Event and data rate = %6.2f %7.2f\n", rate1, rate2);
  //}

  // if we only looped once, then loosening didn't have any effect
  return (loopCount > 1);
}

/**
 * Calculates a figure of merit for the specified rates and prescales.
 * This FOM is a measure of how close the expected rates (calculated from
 * the raw rates and the prescales) are to the desired rates.
 */
double RateLimiter::calcFigureOfMerit(const std::vector<double>& rawRates1,
                                      double fullRate1,
                                      const std::vector<double>& rawRates2,
                                      double fullRate2,
                                      const std::vector<double>& prescales)
{
  double sum1 = 0.0;
  for (uint32 idx = 0; idx < rawRates1.size(); ++idx) {
    sum1 += (rawRates1[idx] / prescales[idx]);
  }
  double delta1 = sum1 - fullRate1;
  double ratio1 = delta1 / fullRate1;
  // give an extra penalty to values more than the max
  if (ratio1 > 0.0) {ratio1 *= 5.0;}

  double sum2 = 0.0;
  for (uint32 idx = 0; idx < rawRates2.size(); ++idx) {
    sum2 += (rawRates2[idx] / prescales[idx]);
  }
  double delta2 = sum2 - fullRate2;
  double ratio2 = delta2 / fullRate2;
  // give an extra penalty to values more than the max
  if (ratio2 > 0.0) {ratio2 *= 5.0;}

  return ((ratio1 * ratio1) + (ratio2 * ratio2));
}
