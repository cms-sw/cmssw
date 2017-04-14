#include "L1Trigger/L1TMuonEndCap/interface/ConditionHelper.hh"

#include "CondFormats/L1TObjects/interface/L1TMuonEndCapParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndcapParamsRcd.h"

#include "CondFormats/L1TObjects/interface/L1TMuonEndCapForest.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndCapForestRcd.h"

#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngine.hh"


ConditionHelper::ConditionHelper():
  params_cache_id_(0ULL), forest_cache_id_(0ULL) {
}

ConditionHelper::~ConditionHelper() {
}

void ConditionHelper::checkAndUpdateConditions(const edm::EventSetup& iSetup, PtAssignmentEngine& pt_assign_engine_) {
  // Pull configuration from the EventSetup
  auto& params_setup = iSetup.get<L1TMuonEndcapParamsRcd>();
  if (params_setup.cacheIdentifier() != params_cache_id_) {
    params_setup.get(params_);

    // with the magic above you can use params_->fwVersion to change emulator's behavior
    // ...

    // reset cache id
    params_cache_id_ = params_setup.cacheIdentifier();
  }

  // Pull pt LUT from the EventSetup
  auto& forest_setup = iSetup.get<L1TMuonEndCapForestRcd>();
  if (forest_setup.cacheIdentifier() != forest_cache_id_) {
    forest_setup.get(forest_);

    // at this point we want to reload the newly pulled pT LUT
    pt_assign_engine_.load(&(*forest_));

    // reset cache id
    forest_cache_id_ = forest_setup.cacheIdentifier();
  }
}
