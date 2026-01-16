#pragma once

#include "CondFormats/RunInfo/interface/LHCInfoPerFill.h"
#include "CondTools/RunInfo/interface/LHCInfoPerFillPopConSourceHandler.h"
#include "CondTools/RunInfo/interface/OMSAccess.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <map>
#include <memory>
#include <string>
#include <tuple>

/*
* A mock PopCon source handler for testing LHCInfoPerFillPopCon logic without
* external dependencies (e.g. OMS, subsystem DBs, destination DB).
*/

class TestLHCInfoPerFillPopConSourceHandler : public LHCInfoPerFillPopConSourceHandler {
public:
  TestLHCInfoPerFillPopConSourceHandler(edm::ParameterSet const& pset);
  ~TestLHCInfoPerFillPopConSourceHandler() override = default;

  std::vector<std::pair<cond::Time_t /*timestamp*/, std::shared_ptr<LHCInfoPerFill>>> mockOmsFills;
  std::map<unsigned short /*fillNr*/, cond::OMSServiceResult> mockLumiData;
  boost::posix_time::ptime mockExecutionTime;

  const Container& iovs() const { return m_iovs; }

protected:
  std::unique_ptr<LHCInfoPerFill> findFillToProcess(cond::OMSService& oms,
                                                    const boost::posix_time::ptime& nextFillSearchTime,
                                                    bool inclusiveSearchTime) override;

  cond::Time_t handleIfNewTagAndGetLastSince() override;

  void fetchLastPayload() override {};

  boost::posix_time::ptime getExecutionTime() const override;

  std::tuple<cond::persistency::Session, cond::persistency::Session> createSubsystemDbSessions() const override;

  void getDipData(const cond::OMSService& oms,
                  const boost::posix_time::ptime& beginFillTime,
                  const boost::posix_time::ptime& endFillTime) override {};

  bool getCTPPSData(cond::persistency::Session& session,
                    const boost::posix_time::ptime& beginFillTime,
                    const boost::posix_time::ptime& endFillTime) override {
    return true;
  };

  bool getEcalData(cond::persistency::Session& session,
                   const boost::posix_time::ptime& lowerTime,
                   const boost::posix_time::ptime& upperTime) override {
    return true;
  };

  std::tuple<cond::OMSServiceResult, bool, std::unique_ptr<cond::OMSServiceQuery>> executeLumiQuery(
      const cond::OMSService& oms,
      unsigned short fillId,
      const boost::posix_time::ptime& beginFillTime,
      const boost::posix_time::ptime& endFillTime) const override;
};
