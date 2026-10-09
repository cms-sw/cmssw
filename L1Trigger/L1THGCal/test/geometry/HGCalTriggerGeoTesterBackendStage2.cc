
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/ForwardDetId/interface/HGCalTriggerModuleDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalTriggerBackendDetId.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeoTesterBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"

class HGCalTriggerGeoTesterBackendStage2 : public HGCalTriggerGeoTesterBase {
public:
  HGCalTriggerGeoTesterBackendStage2(const edm::ParameterSet& conf);
  ~HGCalTriggerGeoTesterBackendStage2() override {};
  void initialize(TTree*, const edm::ParameterSet&) final;
  void check(const HGCalTriggerGeoTesterEventSetup& es) final;
  void fill(const HGCalTriggerGeoTesterEventSetup& es) final;

private:
  void clear() final;

  HGCalTriggerTools triggerTools_;

  unsigned id_ = 0;
  unsigned errorbits_ = 0;
  int label_ = 0;
  int zside_ = 0;
  int sector_ = 0;
  int type_ = 0;
  int stage1_n_ = 0;
  std::vector<uint32_t> stage1s_;
};

DEFINE_EDM_PLUGIN(HGCalTriggerGeoTesterFactory,
                  HGCalTriggerGeoTesterBackendStage2,
                  "HGCalTriggerGeoTesterBackendStage2");

HGCalTriggerGeoTesterBackendStage2::HGCalTriggerGeoTesterBackendStage2(const edm::ParameterSet& conf)
    : HGCalTriggerGeoTesterBase(conf) {}

void HGCalTriggerGeoTesterBackendStage2::initialize(TTree* tree, const edm::ParameterSet& conf) {
  tree_ = tree;

  tree_->Branch("id", &id_, "id/i");
  tree_->Branch("errorbits", &errorbits_, "errorbits/i");
  tree_->Branch("zside", &zside_, "zside/I");
  tree_->Branch("sector", &sector_, "sector/I");
  tree_->Branch("label", &label_, "label/I");
  tree_->Branch("type", &type_, "type/i");
  tree_->Branch("stage1_n", &stage1_n_, "stage1_n/I");
  tree_->Branch("stage1s", &stage1s_);
}

void HGCalTriggerGeoTesterBackendStage2::fill(const HGCalTriggerGeoTesterEventSetup& es) {
  clear();
  edm::LogPrint("TreeFilling") << "Filling Backend Stage 2 tree";
  // Create list of Stage 1 boards from valid cells
  std::unordered_set<uint32_t> modules;
  for (const auto& id : es.geometry->eeGeometry()->getValidDetIds()) {
    modules.insert(es.geometry->getModuleFromCell(id));
  }
  for (const auto& id : es.geometry->hsiGeometry()->getValidDetIds()) {
    modules.insert(es.geometry->getModuleFromCell(id));
  }
  for (const auto& id : es.geometry->hscGeometry()->getValidDetIds()) {
    modules.insert(es.geometry->getModuleFromCell(id));
  }
  std::unordered_set<uint32_t> stage2s;
  for (const auto& module : modules) {
    if (es.geometry->disconnectedModule(module))
      continue;
    auto s1 = es.geometry->getStage1FpgaFromModule(module);
    auto s2s = es.geometry->getStage2FpgasFromStage1Fpga(s1);
    for (auto s2 : s2s) {
      stage2s.insert(s2);
    }
  }
  for (const auto& id : stage2s) {
    HGCalTriggerBackendDetId detid(id);
    const auto error_itr = errors_.detids().find(id);
    if (error_itr != errors_.detids().end()) {
      for (const auto& error : error_itr->second) {
        errorbits_ |= (0x1 << error);
      }
    }
    id_ = id;
    zside_ = detid.zside();
    type_ = detid.type();
    sector_ = detid.sector();
    label_ = detid.label();
    auto stage1s = es.geometry->getStage1FpgasFromStage2Fpga(id);
    stage1_n_ = stage1s.size();
    stage1s_.resize(stage1s.size());
    std::copy(stage1s.begin(), stage1s.end(), stage1s_.begin());

    tree_->Fill();
    clear();
  }
}

void HGCalTriggerGeoTesterBackendStage2::check(const HGCalTriggerGeoTesterEventSetup& es) {
  edm::LogPrint("GeoTesterBackendStage2") << "Checking Stage 2";
  // Create list of modules from valid cells
  std::unordered_set<uint32_t> modules;
  for (const auto& id : es.geometry->eeGeometry()->getValidDetIds()) {
    modules.insert(es.geometry->getModuleFromCell(id));
  }
  for (const auto& id : es.geometry->hsiGeometry()->getValidDetIds()) {
    modules.insert(es.geometry->getModuleFromCell(id));
  }
  for (const auto& id : es.geometry->hscGeometry()->getValidDetIds()) {
    modules.insert(es.geometry->getModuleFromCell(id));
  }
  std::unordered_map<uint32_t, std::unordered_set<uint32_t>> stage2_to_stage1;
  for (const auto& id : modules) {
    if (es.geometry->disconnectedModule(id))
      continue;
    auto s1 = es.geometry->getStage1FpgaFromModule(id);
    auto s2s = es.geometry->getStage2FpgasFromStage1Fpga(s1);
    for (auto s2 : s2s) {
      auto itr_insert = stage2_to_stage1.emplace(s2, std::unordered_set<uint32_t>());
      itr_insert.first->second.emplace(s1);
    }
  }

  // Check consistency of Stage 1 <-> Stage 2 mapping
  for (const auto& [stage2id, stage1s] : stage2_to_stage1) {
    HGCalTriggerGeometryBase::geom_set stage1_from_stage2 = es.geometry->getStage1FpgasFromStage2Fpga(stage2id);
    for (auto stage1 : stage1s) {
      if (stage1_from_stage2.find(stage1) == stage1_from_stage2.end()) {
        errors_.fill(HGCalTriggerGeoTesterErrors::MissingStage1InStage2FPGA, stage2id);
      }
    }
    for (auto stage1 : stage1_from_stage2) {
      if (stage1s.find(stage1) == stage1s.end()) {
        errors_.fill(HGCalTriggerGeoTesterErrors::InvalidStage1InStage2FPGA, stage2id);
      }
    }
  }
}

void HGCalTriggerGeoTesterBackendStage2::clear() {
  id_ = 0;
  errorbits_ = 0;
  label_ = 0;
  zside_ = 0;
  sector_ = 0;
  type_ = 0;
  stage1_n_ = 0;
  stage1s_.clear();
}
