
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/ForwardDetId/interface/HGCalTriggerModuleDetId.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeoTesterBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"

class HGCalTriggerGeoTesterModules : public HGCalTriggerGeoTesterBase {
public:
  HGCalTriggerGeoTesterModules(const edm::ParameterSet& conf);
  ~HGCalTriggerGeoTesterModules() override {};
  void initialize(TTree*, const edm::ParameterSet&) final;
  void check(const HGCalTriggerGeoTesterEventSetup& es) final;
  void fill(const HGCalTriggerGeoTesterEventSetup& es) final;

private:
  void clear() final;

  HGCalTriggerTools triggerTools_;

  unsigned id_ = 0;
  unsigned errorbits_ = 0;
  int disconnected_ = 0;
  int zside_ = 0;
  int subdet_ = 0;
  int sector_ = 0;
  int layer_ = 0;
  unsigned stage1_ = 0;
  unsigned elinks_n_ = 0;
  int u_ = 0;
  int v_ = 0;
  int ieta_ = 0;
  int iphi_ = 0;
  int type_ = 0;
  float x_ = 0.;
  float y_ = 0.;
  float z_ = 0.;
  float eta_ = 0.;
  float phi_ = 0.;
  int cells_n_ = 0;
  int triggercells_n_ = 0;
  int lpgbts_n_ = 0;
  std::vector<uint32_t> cells_;
  std::vector<uint32_t> triggercells_;
  std::vector<uint32_t> lpgbts_;
};

DEFINE_EDM_PLUGIN(HGCalTriggerGeoTesterFactory, HGCalTriggerGeoTesterModules, "HGCalTriggerGeoTesterModules");

HGCalTriggerGeoTesterModules::HGCalTriggerGeoTesterModules(const edm::ParameterSet& conf)
    : HGCalTriggerGeoTesterBase(conf) {}

void HGCalTriggerGeoTesterModules::initialize(TTree* tree, const edm::ParameterSet& conf) {
  tree_ = tree;

  tree_->Branch("id", &id_, "id/i");
  tree_->Branch("errorbits", &errorbits_, "errorbits/i");
  tree_->Branch("disconnected", &disconnected_, "disconnected/I");
  tree_->Branch("zside", &zside_, "zside/I");
  tree_->Branch("subdet", &subdet_, "subdet/I");
  tree_->Branch("sector", &sector_, "sector/I");
  tree_->Branch("layer", &layer_, "layer/I");
  tree_->Branch("stage1", &stage1_, "stage1/i");
  tree_->Branch("elinks_n", &elinks_n_, "elinks_n/I");
  tree_->Branch("u", &u_, "u/I");
  tree_->Branch("v", &v_, "v/I");
  tree_->Branch("ieta", &ieta_, "ieta/I");
  tree_->Branch("iphi", &iphi_, "iphi/I");
  tree_->Branch("type", &type_, "type/i");
  tree_->Branch("x", &x_, "x/F");
  tree_->Branch("y", &y_, "y/F");
  tree_->Branch("z", &z_, "z/F");
  tree_->Branch("eta", &eta_, "eta/F");
  tree_->Branch("phi", &phi_, "phi/F");
  tree_->Branch("cells_n", &cells_n_, "cells_n/I");
  tree_->Branch("triggercells_n", &triggercells_n_, "triggercells_n/I");
  tree_->Branch("lpgbts_n", &lpgbts_n_, "lpgbts_n/I");
  tree_->Branch("cells", &cells_);
  tree_->Branch("triggercells", &triggercells_);
  tree_->Branch("lpgbts", &lpgbts_);
}

void HGCalTriggerGeoTesterModules::fill(const HGCalTriggerGeoTesterEventSetup& es) {
  clear();
  edm::LogPrint("TreeFilling") << "Filling modules tree";
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
  for (const auto& id : modules) {
    HGCalTriggerModuleDetId detid(id);
    const auto error_itr = errors_.detids().find(id);
    if (error_itr != errors_.detids().end()) {
      for (const auto& error : error_itr->second) {
        errorbits_ |= (0x1 << error);
      }
    }
    disconnected_ = es.geometry->disconnectedModule(id);
    id_ = id;
    zside_ = detid.zside();
    subdet_ = detid.triggerSubdetId();
    type_ = detid.type();
    sector_ = detid.sector();
    layer_ = triggerTools_.layerWithOffset(id);
    if (!es.geometry->disconnectedModule(id)) {
      stage1_ = es.geometry->getStage1FpgaFromModule(id);
      elinks_n_ = es.geometry->getLinksInModule(id);
    }
    if (triggerTools_.isSilicon(id)) {
      u_ = detid.moduleU();
      v_ = detid.moduleV();
      ieta_ = -999;
      iphi_ = -999;
    } else if (triggerTools_.isScintillator(id)) {
      u_ = -999;
      v_ = -999;
      ieta_ = detid.eta();
      iphi_ = detid.phi();
    } else {
      throw cms::Exception("InvalidHGCalTriggerDetid")
          << "Found unexpected module detid to be filled in HGCal trigger module ntuple.";
    }
    auto cells = es.geometry->getCellsFromModule(id);
    cells_n_ = cells.size();
    cells_.resize(cells.size());
    std::copy(cells.begin(), cells.end(), cells_.begin());
    auto tcs = es.geometry->getTriggerCellsFromModule(id);
    triggercells_n_ = tcs.size();
    triggercells_.resize(tcs.size());
    std::copy(tcs.begin(), tcs.end(), triggercells_.begin());
    auto lpgbts = es.geometry->getLpgbtsFromModule(id);
    lpgbts_n_ = lpgbts.size();
    lpgbts_.resize(lpgbts.size());
    std::copy(lpgbts.begin(), lpgbts.end(), lpgbts_.begin());
    //
    GlobalPoint center = es.geometry->getModulePosition(id);
    x_ = center.x();
    y_ = center.y();
    z_ = center.z();
    eta_ = center.eta();
    phi_ = center.phi();

    tree_->Fill();
    clear();
  }
}

void HGCalTriggerGeoTesterModules::check(const HGCalTriggerGeoTesterEventSetup& es) {
  edm::LogPrint("GeoTesterModules") << "Checking trigger modules";
  // Create list of modules from valid cells
  std::unordered_map<uint32_t, std::unordered_set<uint32_t>> modules_to_triggercells;
  for (const auto& id : es.geometry->eeGeometry()->getValidDetIds()) {
    if (!es.geometry->eeTopology().valid(id))
      continue;
    unsigned tcid = es.geometry->getTriggerCellFromCell(id);
    unsigned moduleid = es.geometry->getModuleFromTriggerCell(tcid);
    auto itr_insert = modules_to_triggercells.emplace(moduleid, std::unordered_set<uint32_t>());
    itr_insert.first->second.emplace(tcid);
  }
  for (const auto& id : es.geometry->hsiGeometry()->getValidDetIds()) {
    if (!es.geometry->hsiTopology().valid(id))
      continue;
    unsigned tcid = es.geometry->getTriggerCellFromCell(id);
    unsigned moduleid = es.geometry->getModuleFromTriggerCell(tcid);
    auto itr_insert = modules_to_triggercells.emplace(moduleid, std::unordered_set<uint32_t>());
    itr_insert.first->second.emplace(tcid);
  }
  for (const auto& id : es.geometry->hscGeometry()->getValidDetIds()) {
    if (!es.geometry->hscTopology().valid(id))
      continue;
    unsigned tcid = es.geometry->getTriggerCellFromCell(id);
    unsigned moduleid = es.geometry->getModuleFromTriggerCell(tcid);
    auto itr_insert = modules_to_triggercells.emplace(moduleid, std::unordered_set<uint32_t>());
    itr_insert.first->second.emplace(tcid);
  }
  // Check consistency of trigger cells included in modules
  for (const auto& [moduleid, tcs] : modules_to_triggercells) {
    HGCalTriggerGeometryBase::geom_set tcs_from_module = es.geometry->getTriggerCellsFromModule(moduleid);
    for (auto tc : tcs) {
      if (tcs_from_module.find(tc) == tcs_from_module.end()) {
        errors_.fill(HGCalTriggerGeoTesterErrors::MissingTCInModule, moduleid);
      }
    }
    for (auto tc : tcs_from_module) {
      if (tcs.find(tc) == tcs.end()) {
        errors_.fill(HGCalTriggerGeoTesterErrors::InvalidTCInModule, moduleid);
      }
    }
    if (es.geometry->disconnectedModule(moduleid))
      continue;
    auto lpgbts = es.geometry->getLpgbtsFromModule(moduleid);
    if (lpgbts.size() == 0) {
      errors_.fill(HGCalTriggerGeoTesterErrors::ConnectedModuleWithoutLpgbt, moduleid);
    }
    uint32_t stage1 = 0;
    for (const auto& lpgbt : lpgbts) {
      uint32_t stage1_tmp = es.geometry->getStage1FpgaFromLpgbt(lpgbt);
      if (stage1 != 0 && stage1_tmp != stage1) {
        errors_.fill(HGCalTriggerGeoTesterErrors::ModuleSplitInStage1, moduleid);
      }
      stage1 = stage1_tmp;
    }
  }
}

void HGCalTriggerGeoTesterModules::clear() {
  id_ = 0;
  errorbits_ = 0;
  disconnected_ = 0;
  zside_ = 0;
  subdet_ = 0;
  layer_ = 0;
  stage1_ = 0;
  elinks_n_ = 0;
  sector_ = 0;
  u_ = 0;
  v_ = 0;
  ieta_ = 0;
  iphi_ = 0;
  type_ = 0;
  x_ = 0.;
  y_ = 0.;
  z_ = 0.;
  eta_ = 0.;
  phi_ = 0.;
  cells_n_ = 0;
  triggercells_n_ = 0;
  cells_.clear();
  triggercells_.clear();
}
