#ifndef CondCore_OnlineDBOutputService_h
#define CondCore_OnlineDBOutputService_h
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include <string>
#include <map>
#include <fstream>
#include <mutex>
#include <chrono>

//
// Package:     DBOutputService
// Class  :     OnlineDBOutputService
//
/**\class OnlineDBOutputService OnlineDBOutputService.h CondCore/DBOutputService/interface/OnlineDBOutputService.h
   Description: edm service for writing conditions object to DB. Specific implementation for online lumi-based conditions.  
*/
//
// Author:      Giacomo Govi
//

namespace cond {

  cond::Time_t getLatestLumiFromFile(const std::string& fileName) {
    cond::Time_t lastLumiProcessed = cond::time::MIN_VAL;
    std::ifstream lastLumiFile(fileName);
    if (lastLumiFile) {
      lastLumiFile >> lastLumiProcessed;
    }
    return lastLumiProcessed;
  }

  namespace service {

    class OnlineDBOutputService : public PoolDBOutputService {
    public:
      OnlineDBOutputService(const edm::ParameterSet& iConfig, edm::ActivityRegistry& iAR);

      ~OnlineDBOutputService() override;

      cond::Iov_t preLoadIov(const std::string& recordName, cond::Time_t targetTime);

      //
      template <typename PayloadType>
      bool writeForNextLumisection(const PayloadType* payload, const std::string& recordName) {
        cond::Time_t targetTime = getLastLumiProcessed() + m_latencyInLumisections;
        auto t0 = std::chrono::high_resolution_clock::now();
        edm::LogInfo(MSGSOURCE) << "Updating lumisection " << targetTime;
        cond::Hash payloadId = PoolDBOutputService::writeOne<PayloadType>(payload, targetTime, recordName);
        bool ret = true;
        if (payloadId.empty()) {
          return false;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        auto w_lat = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        edm::LogInfo(MSGSOURCE) << "Update has taken " << w_lat << " microsecs.";
        // check for late updates...
        cond::Time_t lastProcessed = getLastLumiProcessed();
        edm::LogInfo(MSGSOURCE) << "Last lumisection processed after update: " << lastProcessed;
        // check the pre-loaded iov
        edm::LogInfo(MSGSOURCE) << "Preloading lumisection " << targetTime;
        auto t2 = std::chrono::high_resolution_clock::now();
        cond::Iov_t usedIov = preLoadIov(recordName, targetTime);
        auto t3 = std::chrono::high_resolution_clock::now();
        edm::LogInfo(MSGSOURCE) << "Iov for preloaded lumisection " << targetTime << " is " << usedIov.since;
        auto p_lat = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
        edm::LogInfo(MSGSOURCE) << "Preload has taken " << p_lat << " microsecs.";
        if (usedIov.since < targetTime) {
          edm::LogWarning(MSGSOURCE) << "Found a late update for lumisection " << targetTime << "(found since "
                                     << usedIov.since << "). A revert is required.";
          PoolDBOutputService::eraseSinceTime(payloadId, targetTime, recordName);
          PoolDBOutputService::commitTransaction();
          ret = false;
        }
        auto t4 = std::chrono::high_resolution_clock::now();
        auto t_lat = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t0).count();
        edm::LogInfo(MSGSOURCE) << "Total update time: " << t_lat << " microsecs.";
        return ret;
      }

    private:
      cond::Time_t getLastLumiProcessed();

      cond::persistency::Session getReadOnlyCache(cond::Time_t targetTime);

    private:
      static constexpr const char* const MSGSOURCE = "OnlineDBOuputService";
      cond::Time_t m_runNumber;
      size_t m_latencyInLumisections;
      std::string m_lastLumiUrl;
      std::string m_lastLumiFile;
      //std::chrono::time_point<std::chrono::steady_clock> m_startRunTime;
      std::string m_preLoadConnectionString;
      bool m_debug;

    };  //OnlineDBOutputService
  }     // namespace service
}  // namespace cond
#endif
