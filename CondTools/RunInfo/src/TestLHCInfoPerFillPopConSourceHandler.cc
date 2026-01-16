#include "CondTools/RunInfo/interface/TestLHCInfoPerFillPopConSourceHandler.h"
#include <tuple>

TestLHCInfoPerFillPopConSourceHandler::TestLHCInfoPerFillPopConSourceHandler(edm::ParameterSet const& pset)
    : LHCInfoPerFillPopConSourceHandler(pset) {}

std::unique_ptr<LHCInfoPerFill> TestLHCInfoPerFillPopConSourceHandler::findFillToProcess(
    cond::OMSService& oms, const boost::posix_time::ptime& nextFillSearchTime, bool inclusiveSearchTime) {
  for (auto& fill : mockOmsFills) {
    auto fillStartTime = cond::time::to_boost(fill.first);
    if (inclusiveSearchTime ? (fillStartTime >= nextFillSearchTime) : (fillStartTime > nextFillSearchTime)) {
      auto fillPayload = std::make_unique<LHCInfoPerFill>(*fill.second);
      return fillPayload;
    }
  }
  return nullptr;
}

cond::Time_t TestLHCInfoPerFillPopConSourceHandler::handleIfNewTagAndGetLastSince() {
  cond::Time_t lastSince = 0;
  if (m_endFillMode) {
    addEmptyPayload(1);
    lastSince = 1;
  } else {
    lastSince = 0;
  }
  return lastSince;
}

boost::posix_time::ptime TestLHCInfoPerFillPopConSourceHandler::getExecutionTime() const { return mockExecutionTime; }

std::tuple<cond::persistency::Session, cond::persistency::Session>
TestLHCInfoPerFillPopConSourceHandler::createSubsystemDbSessions() const {
  return std::make_tuple(cond::persistency::Session(), cond::persistency::Session());
}

std::tuple<cond::OMSServiceResult, bool, std::unique_ptr<cond::OMSServiceQuery>>
TestLHCInfoPerFillPopConSourceHandler::executeLumiQuery(const cond::OMSService& oms,
                                                        unsigned short fillId,
                                                        const boost::posix_time::ptime& beginFillTime,
                                                        const boost::posix_time::ptime& endFillTime) const {
  auto it = mockLumiData.find(fillId);
  if (it != mockLumiData.end()) {
    return std::make_tuple(it->second, true, nullptr);
  }
  return std::make_tuple(cond::OMSServiceResult(), false, nullptr);
}
