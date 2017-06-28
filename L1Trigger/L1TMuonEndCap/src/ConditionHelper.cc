#include "L1Trigger/L1TMuonEndCap/interface/ConditionHelper.h"

#include "CondFormats/L1TObjects/interface/L1TMuonEndCapParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndcapParamsRcd.h"

#include "CondFormats/L1TObjects/interface/L1TMuonEndCapForest.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndCapForestRcd.h"

#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngine.h"

#include "FWCore/Framework/interface/EventSetup.h"

ConditionHelper::ConditionHelper():
  params_cache_id_(0ULL), forest_cache_id_(0ULL) {
}

ConditionHelper::~ConditionHelper() {
}

bool ConditionHelper::checkAndUpdateConditions(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  bool new_params = false;
  bool new_forests = false;

  // Pull configuration from the EventSetup
  auto& params_setup = iSetup.get<L1TMuonEndcapParamsRcd>();
  if (params_setup.cacheIdentifier() != params_cache_id_) {
    params_setup.get(params_);

    // with the magic above you can use params_->fwVersion to change emulator's behavior
    // ...

    // reset cache id
    params_cache_id_ = params_setup.cacheIdentifier();
    new_params = true;
  }

  // Pull pt LUT from the EventSetup
  auto& forest_setup = iSetup.get<L1TMuonEndCapForestRcd>();
  if (forest_setup.cacheIdentifier() != forest_cache_id_) {
    forest_setup.get(forest_);

    // at this point we want to reload the newly pulled pT LUT
    // ...

    // reset cache id
    forest_cache_id_ = forest_setup.cacheIdentifier();
    new_forests = true;
  }

  // Debug
  //std::cout << "Run number: " << iEvent.id().run() << " fw_version: " << get_fw_version()
  //    << " pt_lut_version: " << get_pt_lut_version() << " pc_lut_version: " << get_pc_lut_version()
  //    << std::endl;

  return (new_params || new_forests);
}

unsigned int ConditionHelper::get_fw_version() const {
  // std::cout << "Getting firmware version from ConditionHelper" << std::endl;
  return params_->firmwareVersion_;
}

unsigned int ConditionHelper::get_pt_lut_version() const {
  // std::cout << "Getting pT LUT version from ConditionHelper" << std::endl;
  return (params_->PtAssignVersion_ & 0xff);  // Version indicated by first two bytes
}

unsigned int ConditionHelper::get_pc_lut_version() const {
  // "PhiMatchWindowSt1" arbitrarily re-mapped to Primitive conversion (PC LUT) version
  // because of rigid CondFormats naming conventions - AWB 02.06.17
  // std::cout << "Getting PC LUT version from ConditionHelper" << std::endl;
  return params_->PhiMatchWindowSt1_;  
}
