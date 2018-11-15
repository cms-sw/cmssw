#include "L1Trigger/L1TMuonEndCap/interface/ConditionHelper.h"

#include "CondFormats/L1TObjects/interface/L1TMuonEndCapParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndCapParamsRcd.h"

#include "CondFormats/L1TObjects/interface/L1TMuonEndCapForest.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndCapForestRcd.h"

#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngine.h"

#include "FWCore/Framework/interface/EventSetup.h"

ConditionHelper::ConditionHelper():
  params_cache_id_(0ULL), forest_cache_id_(0ULL) {
}

ConditionHelper::~ConditionHelper() {
}

void ConditionHelper::checkAndUpdateConditions(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  bool new_params = false;
  bool new_forests = false;

  // Pull configuration from the EventSetup
  auto params_setup = iSetup.get<L1TMuonEndCapParamsRcd>();
  if (params_setup.cacheIdentifier() != params_cache_id_) {
    params_setup.get(params_);

    // with the magic above you can use params_->fwVersion to change emulator's behavior
    // ...

    // reset cache id
    params_cache_id_ = params_setup.cacheIdentifier();
    new_params = true;
  }

  // Pull pt LUT from the EventSetup
  auto forest_setup = iSetup.get<L1TMuonEndCapForestRcd>();
  if (forest_setup.cacheIdentifier() != forest_cache_id_) {
    forest_setup.get(forest_);

    // at this point we want to reload the newly pulled pT LUT
    // ...

    // reset cache id
    forest_cache_id_ = forest_setup.cacheIdentifier();
    new_forests = true;
  }

  bool new_conditions = (new_params || new_forests);
  if (new_conditions) {
    edm::LogInfo("L1T") << "EMTF updating conditions: pc_lut_ver: " << get_pc_lut_version() << " pt_lut_ver: " << get_pt_lut_version() << " fw_ver: " << get_fw_version();
  }

  // Debug
  //edm::LogWarning("L1T") << "EMTF new conditions? Yes (1) or no (0)? -- " << new_conditions << std::endl;
  //edm::LogWarning("L1T") << "EMTF updating conditions: pc_lut_ver: " << get_pc_lut_version() << " pt_lut_ver: " << get_pt_lut_version() << " fw_ver: " << get_fw_version();
}

unsigned int ConditionHelper::get_fw_version() const {
  // std::cout << "    - Getting firmware version from ConditionHelper: version = " << params_->firmwareVersion_ << std::endl;
  return params_->firmwareVersion_;
}

unsigned int ConditionHelper::get_pt_lut_version() const {
  // std::cout << "    - Getting pT LUT version from ConditionHelper: version = " << (params_->PtAssignVersion_ & 0xff);
  // std::cout << " (lowest bits of " << params_->PtAssignVersion_ << ")" << std::endl;
  if (params_->firmwareVersion_ < 50000)  // for 2016
    return 5;
  return (params_->PtAssignVersion_ & 0xff);  // Version indicated by first two bytes
}

unsigned int ConditionHelper::get_pc_lut_version() const {
  // "PhiMatchWindowSt1" arbitrarily re-mapped to Primitive conversion (PC LUT) version
  // because of rigid CondFormats naming conventions - AWB 02.06.17
  // std::cout << "    - Getting proper PC LUT version from ConditionHelper: version = " << params_->PhiMatchWindowSt1_ << std::endl;
  // return params_->PhiMatchWindowSt1_;

  // Hack until we figure out why the database is returning "0" for 2017 data - AWB 04.08.17
  // std::cout << "    - Getting hacked PC LUT version from ConditionHelper: version = " << (params_->firmwareVersion_ >= 50000) << std::endl;
  if        (params_->firmwareVersion_ < 50000) {       // For 2016
    return 0;
  } else if (params_->firmwareVersion_ < 1537467271) { // From the beginning of 2017
    return 1;                                          // Corresponding to FW timestamps before Sept. 20, 2018
  } else {
    return 2;                                          // Starting September 26, 2018 with run 323556 (data only, not in MC)
  }

}
