#ifndef STOR_RATE_LIMITER_H
#define STOR_RATE_LIMITER_H

/**
 * This class provides functionality to limit events to consumers
 * based on maximum event and data rates.
 *
 * $Id: RateLimiter.h,v 1.12 2008/03/03 20:09:36 biery Exp $
 */

#include "IOPool/Streamer/interface/MsgTools.h"
#include "EventFilter/StorageManager/interface/RollingIntervalCounter.h"
#include "boost/random.hpp"
#include "boost/shared_ptr.hpp"
#include "boost/thread/mutex.hpp"
#include <map>
#include <vector>

namespace stor
{
  class RateLimiter
  {

   public:

    RateLimiter(double maxEventRate, double maxDataRate);

    void addConsumer(uint32 consumerId);
    void removeConsumer(uint32 consumerId);

    std::vector<uint32> getAllowedConsumersFromList(double dataSize,
                           const std::vector<uint32>& candidateList);

    void dumpData(std::ostream& outStream);

    static double calcRate(std::vector<double> rates,
                           std::vector<double> prescales);

   private:

    static void determineTargetPrescales(double fullRate,
                                         const std::vector<double>& rawRates,
                                         std::vector<double>& targetPrescales);
    static void determineTargetRates(double fullRate,
                                     const std::vector<double>& rawRates,
                                     std::vector<double>& targetRates);
    static void fairShareAlgo(double fullRate,
                              const std::vector<double>& rawRates,
                              std::vector<double>& targetRates);

    static void loosenPrescales(const std::vector<double>& rawRates1,
                                double fullRate1,
                                const std::vector<double>& rawRates2,
                                double fullRate2,
                                std::vector<double>& prescales,
                                const std::vector<double>& minPrescales);
    static bool loosenOnePrescale(const std::vector<double>& rawRates1,
                                  double fullRate1,
                                  const std::vector<double>& rawRates2,
                                  double fullRate2,
                                  std::vector<double>& prescales,
                                  uint32 psIndex, double lowBound);
    static double calcFigureOfMerit(const std::vector<double>& rawRates1,
                                    double fullRate1,
                                    const std::vector<double>& rawRates2,
                                    double fullRate2,
                                    const std::vector<double>& prescales);

    double maxEventRate_;
    double maxDataRate_;

    std::vector<uint32> consumerList_;
    std::map<uint32,boost::shared_ptr<RollingIntervalCounter> > dataRateTable_;

    boost::mt19937 baseGenerator_;
    boost::shared_ptr< boost::uniform_01<boost::mt19937> > generator_;

    boost::mutex dataMutex_;

  };
}

#endif
