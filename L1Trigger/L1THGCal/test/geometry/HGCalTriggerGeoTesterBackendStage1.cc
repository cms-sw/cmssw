
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/ForwardDetId/interface/HGCalTriggerModuleDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalTriggerBackendDetId.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeoTesterBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"

class HGCalTriggerGeoTesterBackendStage1 : public HGCalTriggerGeoTesterBase {
public:
  HGCalTriggerGeoTesterBackendStage1(const edm::ParameterSet& conf);
  ~HGCalTriggerGeoTesterBackendStage1() override {};
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
  int lpgbts_n_ = 0;
  int modules_n_ = 0;
  std::vector<uint32_t> lpgbts_;
  std::vector<uint32_t> modules_;
};

DEFINE_EDM_PLUGIN(HGCalTriggerGeoTesterFactory,
                  HGCalTriggerGeoTesterBackendStage1,
                  "HGCalTriggerGeoTesterBackendStage1");

HGCalTriggerGeoTesterBackendStage1::HGCalTriggerGeoTesterBackendStage1(const edm::ParameterSet& conf)
    : HGCalTriggerGeoTesterBase(conf) {}

void HGCalTriggerGeoTesterBackendStage1::initialize(TTree* tree, const edm::ParameterSet& conf) {
  tree_ = tree;

  tree_->Branch("id", &id_, "id/i");
  tree_->Branch("errorbits", &errorbits_, "errorbits/i");
  tree_->Branch("zside", &zside_, "zside/I");
  tree_->Branch("sector", &sector_, "sector/I");
  tree_->Branch("label", &label_, "label/I");
  tree_->Branch("type", &type_, "type/i");
  tree_->Branch("lpgbts_n", &lpgbts_n_, "lpgbts_n/I");
  tree_->Branch("lpgbts", &lpgbts_);
  tree_->Branch("modules_n", &modules_n_, "modules_n/I");
  tree_->Branch("modules", &modules_);
}

void HGCalTriggerGeoTesterBackendStage1::fill(const HGCalTriggerGeoTesterEventSetup& es) {
  clear();
  edm::LogPrint("TreeFilling") << "Filling Backend Stage 1 tree";
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
  std::unordered_set<uint32_t> stage1s;
  for (const auto& module : modules) {
    if (es.geometry->disconnectedModule(module))
      continue;
    stage1s.insert(es.geometry->getStage1FpgaFromModule(module));
  }
  for (const auto& id : stage1s) {
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
    auto lpgbts = es.geometry->getLpgbtsFromStage1Fpga(id);
    lpgbts_n_ = lpgbts.size();
    lpgbts_.resize(lpgbts.size());
    std::copy(lpgbts.begin(), lpgbts.end(), lpgbts_.begin());
    auto modules = es.geometry->getModulesFromStage1Fpga(id);
    modules_n_ = modules.size();
    modules_.resize(modules.size());
    std::copy(modules.begin(), modules.end(), modules_.begin());

    tree_->Fill();
    clear();
  }
}

void HGCalTriggerGeoTesterBackendStage1::check(const HGCalTriggerGeoTesterEventSetup& es) {
  edm::LogPrint("GeoTesterBackendStage1") << "Checking Stage 1";
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
  std::unordered_map<uint32_t, std::unordered_set<uint32_t>> stage1_to_modules;
  for (const auto& id : modules) {
    if (es.geometry->disconnectedModule(id))
      continue;
    auto stage1 = es.geometry->getStage1FpgaFromModule(id);
    auto itr_insert = stage1_to_modules.emplace(stage1, std::unordered_set<uint32_t>());
    itr_insert.first->second.emplace(id);
  }

  // Check consistency of trigger cells included in modules
  for (const auto& [stage1id, modules] : stage1_to_modules) {
    HGCalTriggerGeometryBase::geom_set modules_from_stage1 = es.geometry->getModulesFromStage1Fpga(stage1id);
    for (auto module : modules) {
      if (modules_from_stage1.find(module) == modules_from_stage1.end()) {
        errors_.fill(HGCalTriggerGeoTesterErrors::MissingModuleInStage1FPGA, stage1id);
      }
    }
    for (auto module : modules_from_stage1) {
      if (modules.find(module) == modules.end()) {
        errors_.fill(HGCalTriggerGeoTesterErrors::InvalidModuleInStage1FPGA, stage1id);
      }
    }
  }
}

void HGCalTriggerGeoTesterBackendStage1::clear() {
  id_ = 0;
  errorbits_ = 0;
  label_ = 0;
  zside_ = 0;
  sector_ = 0;
  type_ = 0;
  lpgbts_n_ = 0;
  lpgbts_.clear();
  modules_n_ = 0;
  modules_.clear();
}
