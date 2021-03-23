#include <memory>

#include "L1Trigger/L1TMuonEndCap/interface/EMTFSetup.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngine2016.h"
#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngine2017.h"

EMTFSetup::EMTFSetup(const edm::ParameterSet& iConfig)
    : geometry_translator_(),
      condition_helper_(),
      version_control_(iConfig),
      sector_processor_lut_(),
      pt_assign_engine_(nullptr),
      pt_assign_engine_dxy_(nullptr),
      fw_ver_(0),
      pt_lut_ver_(0),
      pc_lut_ver_(0) {
  // Set pt assignment engine according to Era
  if (era() == "Run2_2016") {
    pt_assign_engine_ = std::make_unique<PtAssignmentEngine2016>();
  } else if (era() == "Run2_2017" || era() == "Run2_2018") {
    pt_assign_engine_ = std::make_unique<PtAssignmentEngine2017>();
  } else if (era() == "Run3_2021") {
    pt_assign_engine_ = std::make_unique<PtAssignmentEngine2017>();  //TODO - implement ver 2021
  } else {
    throw cms::Exception("L1TMuonEndCap") << "Cannot recognize the era option: " << era();
  }

  // No era setup for displaced pT assignment engine
  pt_assign_engine_dxy_ = std::make_unique<PtAssignmentEngineDxy>();

  emtf_assert(pt_assign_engine_ != nullptr);
  emtf_assert(pt_assign_engine_dxy_ != nullptr);
}

EMTFSetup::~EMTFSetup() {}

void EMTFSetup::reload(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Get the geometry for TP conversions
  geometry_translator_.checkAndUpdateGeometry(iSetup);

  // Get the conditions, primarily the firmware version and the BDT forests
  condition_helper_.checkAndUpdateConditions(iEvent, iSetup);

  // Set version numbers
  fw_ver_ = condition_helper_.get_fw_version();
  pt_lut_ver_ = condition_helper_.get_pt_lut_version();
  pc_lut_ver_ = condition_helper_.get_pc_lut_version();

  if (!useO2O()) {
    // Currently, do not modify fw_ver_ and pt_lut_ver_
    pc_lut_ver_ = condition_helper_.get_pc_lut_version_unchecked();
  }

  // Do run-dependent configuration. This may overwrite the configurables passed by the python config file
  version_control_.configure_by_fw_version(get_fw_version());

  // Reload primitive conversion LUTs if necessary
  sector_processor_lut_.read(iEvent.isRealData(), get_pc_lut_version());

  // Reload pT LUT if necessary
  pt_assign_engine_->load(get_pt_lut_version(), condition_helper_.getForest());

  return;
}
