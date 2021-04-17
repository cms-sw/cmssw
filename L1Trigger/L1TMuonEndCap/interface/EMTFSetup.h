#ifndef L1TMuonEndCap_EMTFSetup_h
#define L1TMuonEndCap_EMTFSetup_h

#include <memory>
#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "L1Trigger/L1TMuon/interface/GeometryTranslator.h"
#include "L1Trigger/L1TMuonEndCap/interface/ConditionHelper.h"
#include "L1Trigger/L1TMuonEndCap/interface/VersionControl.h"
#include "L1Trigger/L1TMuonEndCap/interface/SectorProcessorLUT.h"
#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngine.h"
#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngineDxy.h"

class EMTFSetup {
public:
  explicit EMTFSetup(const edm::ParameterSet& iConfig);
  ~EMTFSetup();

  // Check and update geometry, conditions, versions, sp LUTs, and pt assignment engine
  void reload(const edm::Event& iEvent, const edm::EventSetup& iSetup);

  // Getters
  const GeometryTranslator& getGeometryTranslator() const { return geometry_translator_; }

  const ConditionHelper& getConditionHelper() const { return condition_helper_; }

  const VersionControl& getVersionControl() const { return version_control_; }

  const SectorProcessorLUT& getSectorProcessorLUT() const { return sector_processor_lut_; }

  PtAssignmentEngine* getPtAssignmentEngine() const { return pt_assign_engine_.get(); }

  PtAssignmentEngineDxy* getPtAssignmentEngineDxy() const { return pt_assign_engine_dxy_.get(); }

  // Setters
  //void set_fw_version(unsigned version) { fw_ver_ = version; }
  //void set_pt_lut_version(unsigned version) { pt_lut_ver_ = version; }
  //void set_pc_lut_version(unsigned version) { pc_lut_ver_ = version; }

  // Getters
  unsigned get_fw_version() const { return fw_ver_; }
  unsigned get_pt_lut_version() const { return pt_lut_ver_; }
  unsigned get_pc_lut_version() const { return pc_lut_ver_; }

  // VersionControl getters
  const edm::ParameterSet& getConfig() const { return version_control_.getConfig(); }
  int verbose() const { return version_control_.verbose(); }
  bool useO2O() const { return version_control_.useO2O(); }
  std::string era() const { return version_control_.era(); }

private:
  GeometryTranslator geometry_translator_;

  ConditionHelper condition_helper_;

  VersionControl version_control_;

  SectorProcessorLUT sector_processor_lut_;

  // Polymorphic class
  std::unique_ptr<PtAssignmentEngine> pt_assign_engine_;
  // Displaced muon pT assignment
  std::unique_ptr<PtAssignmentEngineDxy> pt_assign_engine_dxy_;

  // Version numbers. Note: may be different from those in ConditionHelper
  unsigned fw_ver_;
  unsigned pt_lut_ver_;
  unsigned pc_lut_ver_;
};

#endif
