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
};

#endif
