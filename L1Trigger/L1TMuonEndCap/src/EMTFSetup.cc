#include "L1Trigger/L1TMuonEndCap/interface/EMTFSetup.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngine2016.h"
#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngine2017.h"

EMTFSetup::EMTFSetup(const edm::ParameterSet& iConfig)
    : geometry_translator_(),
      condition_helper_(),
      version_control_(iConfig),
      sector_processor_lut_(),
      pt_assign_engine_(nullptr) {
  // Set pt assignment engine according to Era
  if (era() == "Run2_2016") {
    pt_assign_engine_.reset(new PtAssignmentEngine2016());
  } else if (era() == "Run2_2017" || era() == "Run2_2018") {
    pt_assign_engine_.reset(new PtAssignmentEngine2017());
  } else if (era() == "Run3_2021") {
    pt_assign_engine_.reset(new PtAssignmentEngine2017());  //TODO - implement ver 2021
  } else {
    throw cms::Exception("L1TMuonEndCap") << "Cannot recognize the era option: " << era();
  }

  emtf_assert(pt_assign_engine_ != nullptr);
}

EMTFSetup::~EMTFSetup() {}

void EMTFSetup::reload(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Get the geometry for TP conversions
  geometry_translator_.checkAndUpdateGeometry(iSetup);

  // Get the conditions, primarily the firmware version and the BDT forests
  condition_helper_.checkAndUpdateConditions(iEvent, iSetup);

  // Do run-dependent configuration. This may overwrite the configurables passed by the python config file
  version_control_.configure_by_fw_version(condition_helper_.get_fw_version());

  // Decide the best pc_lut_ver & pt_lut_ver
  unsigned pc_lut_ver = condition_helper_.get_pc_lut_version();  //TODO - check this
  unsigned pt_lut_ver = condition_helper_.get_pt_lut_version();  //TODO - check this

  // Reload primitive conversion LUTs if necessary
  sector_processor_lut_.read(iEvent.isRealData(), pc_lut_ver);

  // Reload pT LUT if necessary
  pt_assign_engine_->load(pt_lut_ver, &(condition_helper_.getForest()));

  return;
}
