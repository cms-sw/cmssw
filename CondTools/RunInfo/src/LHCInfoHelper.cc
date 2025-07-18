#include "CondTools/RunInfo/interface/LHCInfoHelper.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//*****************************************************************
// Returns lumi-type IOV (packed using cond::time::lumiTime) from
// last LS of last Run of the specified Fill
//*****************************************************************

std::pair<int, unsigned short> cond::lhcInfoHelper::getFillLastRunAndLS(const cond::OMSService& oms,
                                                                        unsigned short fillId) {
  // Define query
  auto query = oms.query("lumisections");
  query->addOutputVars({"lumisection_number", "run_number"});
  query->filterEQ("fill_number", fillId);
  query->limit(cond::lhcInfoHelper::kLumisectionsQueryLimit);

  // Execute query
  if (!query->execute()) {
    throw cms::Exception("OMSQueryFailure")
        << "OMS query of fill " << fillId << " failed, status:" << query->status() << "\n";
  }

  // Get query result
  auto queryResult = query->result();
  if (queryResult.empty()) {
    throw cms::Exception("OMSQueryFailure") << "OMS query of fill " << fillId << " returned empty result!\n";
  }

  // Return the final IOV
  auto lastRun = queryResult.back().get<int>("run_number");
  auto lastLumi = queryResult.back().get<unsigned short>("lumisection_number");
  return std::make_pair(lastRun, lastLumi);
}

cond::Time_t cond::lhcInfoHelper::getFillLastLumiIOV(const cond::OMSService& oms, unsigned short fillId) {
  auto [lastRun, lastLumi] = getFillLastRunAndLS(oms, fillId);
  return cond::time::lumiTime(lastRun, lastLumi);
}