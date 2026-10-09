
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/ForwardDetId/interface/HGCalTriggerDetId.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeoTesterBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"

class HGCalTriggerGeoTesterTriggerCells : public HGCalTriggerGeoTesterBase {
public:
  HGCalTriggerGeoTesterTriggerCells(const edm::ParameterSet& conf);
  ~HGCalTriggerGeoTesterTriggerCells() override {};
  void initialize(TTree*, const edm::ParameterSet&) final;
  void check(const HGCalTriggerGeoTesterEventSetup& es) final;
  void fill(const HGCalTriggerGeoTesterEventSetup& es) final;

private:
  void clear() final;

  HGCalTriggerTools triggerTools_;

  unsigned id_ = 0;
  int valid_ = 0;
  unsigned errorbits_ = 0;
  int disconnected_ = 0;
  int zside_ = 0;
  int subdet_ = 0;
  int layer_ = 0;
  int u_ = 0;
  int v_ = 0;
  int ieta_ = 0;
  int iphi_ = 0;
  int waferU_ = 0;
  int waferV_ = 0;
  int type_ = 0;
  unsigned modid_ = 0;
  float x_ = 0.;
  float y_ = 0.;
  float z_ = 0.;
  float eta_ = 0.;
  float phi_ = 0.;
  int cells_n_ = 0;
  std::vector<uint32_t> cells_;
};

DEFINE_EDM_PLUGIN(HGCalTriggerGeoTesterFactory, HGCalTriggerGeoTesterTriggerCells, "HGCalTriggerGeoTesterTriggerCells");

HGCalTriggerGeoTesterTriggerCells::HGCalTriggerGeoTesterTriggerCells(const edm::ParameterSet& conf)
    : HGCalTriggerGeoTesterBase(conf) {}

void HGCalTriggerGeoTesterTriggerCells::initialize(TTree* tree, const edm::ParameterSet& conf) {
  tree_ = tree;

  tree_->Branch("valid", &valid_, "valid/I");
  tree_->Branch("disconnected", &disconnected_, "disconnected/I");
  tree_->Branch("id", &id_, "id/i");
  tree_->Branch("errorbits", &errorbits_, "errorbits/i");
  tree_->Branch("zside", &zside_, "zside/I");
  tree_->Branch("subdet", &subdet_, "subdet/I");
  tree_->Branch("layer", &layer_, "layer/I");
  tree_->Branch("u", &u_, "u/I");
  tree_->Branch("v", &v_, "v/I");
  tree_->Branch("ieta", &ieta_, "ieta/I");
  tree_->Branch("iphi", &iphi_, "iphi/I");
  tree_->Branch("waferU", &waferU_, "waferU/I");
  tree_->Branch("waferV", &waferV_, "waferV/I");
  tree_->Branch("type", &type_, "type/i");
  tree_->Branch("modid", &modid_, "modid/I");
  tree_->Branch("x", &x_, "x/F");
  tree_->Branch("y", &y_, "y/F");
  tree_->Branch("z", &z_, "z/F");
  tree_->Branch("eta", &eta_, "eta/F");
  tree_->Branch("phi", &phi_, "phi/F");
  tree_->Branch("cells_n", &cells_n_, "cells_n/I");
  tree_->Branch("cells", &cells_);
}

void HGCalTriggerGeoTesterTriggerCells::fill(const HGCalTriggerGeoTesterEventSetup& es) {
  clear();
  edm::LogPrint("TreeFilling") << "Filling trigger cells tree";
  // Create list of TCs from valid cells
  std::unordered_set<uint32_t> triggercells;
  for (const auto& id : es.geometry->eeGeometry()->getValidDetIds()) {
    triggercells.insert(es.geometry->getTriggerCellFromCell(id));
  }
  for (const auto& id : es.geometry->hsiGeometry()->getValidDetIds()) {
    triggercells.insert(es.geometry->getTriggerCellFromCell(id));
  }
  for (const auto& id : es.geometry->hscGeometry()->getValidDetIds()) {
    triggercells.insert(es.geometry->getTriggerCellFromCell(id));
  }
  if (es.geometry->isWithNoseGeometry()) {
    for (const auto& id : es.geometry->noseGeometry()->getValidDetIds()) {
      triggercells.insert(es.geometry->getTriggerCellFromCell(id));
    }
  }
  for (const auto& id : triggercells) {
    DetId detid(id);
    modid_ = es.geometry->getModuleFromTriggerCell(id);
    valid_ = es.geometry->validTriggerCell(id);
    disconnected_ = es.geometry->disconnectedModule(modid_);
    id_ = id;
    const auto error_itr = errors_.detids().find(id);
    if (error_itr != errors_.detids().end()) {
      for (const auto& error : error_itr->second) {
        errorbits_ |= (0x1 << error);
      }
    }
    zside_ = triggerTools_.zside(id);
    layer_ = triggerTools_.layerWithOffset(id);
    if (detid.det() == DetId::HGCalTrigger) {
      HGCalTriggerDetId idtrg(id);
      subdet_ = idtrg.subdet();
      waferU_ = idtrg.waferU();
      waferV_ = idtrg.waferV();
      type_ = idtrg.type();
      u_ = idtrg.triggerCellU();
      v_ = idtrg.triggerCellV();
      ieta_ = -999;
      iphi_ = -999;
    } else if (detid.det() == DetId::HGCalHSc) {
      HGCScintillatorDetId idsci(id);
      subdet_ = idsci.subdet();
      waferU_ = -999;
      waferV_ = -999;
      type_ = idsci.type();
      u_ = -999;
      v_ = -999;
      ieta_ = idsci.ietaAbs();
      iphi_ = idsci.iphi();
    } else {
      throw cms::Exception("InvalidHGCalTriggerDetid")
          << "Found unexpected trigger cell detid to be filled in HGCal Trigger Cell ntuple.";
    }
    auto cells = es.geometry->getCellsFromTriggerCell(id_);
    cells_n_ = cells.size();
    cells_.resize(cells.size());
    std::copy(cells.begin(), cells.end(), cells_.begin());
    //
    GlobalPoint center = es.geometry->getTriggerCellPosition(id);
    x_ = center.x();
    y_ = center.y();
    z_ = center.z();
    eta_ = center.eta();
    phi_ = center.phi();

    tree_->Fill();
    clear();
  }
}

void HGCalTriggerGeoTesterTriggerCells::check(const HGCalTriggerGeoTesterEventSetup& es) {
  edm::LogPrint("GeoTesterTriggerCells") << "Checking trigger cells";
  // Create list of TCs from valid cells
  std::unordered_map<uint32_t, std::unordered_set<uint32_t>> triggercells_to_cells;
  for (const auto& id : es.geometry->eeGeometry()->getValidDetIds()) {
    if (!es.geometry->eeTopology().valid(id))
      continue;
    unsigned tcid = es.geometry->getTriggerCellFromCell(id);
    auto itr_insert = triggercells_to_cells.emplace(tcid, std::unordered_set<uint32_t>());
    itr_insert.first->second.emplace(id);
  }
  for (const auto& id : es.geometry->hsiGeometry()->getValidDetIds()) {
    if (!es.geometry->hsiTopology().valid(id))
      continue;
    unsigned tcid = es.geometry->getTriggerCellFromCell(id);
    auto itr_insert = triggercells_to_cells.emplace(tcid, std::unordered_set<uint32_t>());
    itr_insert.first->second.emplace(id);
  }
  for (const auto& id : es.geometry->hscGeometry()->getValidDetIds()) {
    if (!es.geometry->hscTopology().valid(id))
      continue;
    unsigned tcid = es.geometry->getTriggerCellFromCell(id);
    auto itr_insert = triggercells_to_cells.emplace(tcid, std::unordered_set<uint32_t>());
    itr_insert.first->second.emplace(id);
  }
  if (es.geometry->isWithNoseGeometry()) {
    for (const auto& id : es.geometry->noseGeometry()->getValidDetIds()) {
      if (!es.geometry->noseTopology().valid(id))
        continue;
      unsigned tcid = es.geometry->getTriggerCellFromCell(id);
      auto itr_insert = triggercells_to_cells.emplace(tcid, std::unordered_set<uint32_t>());
      itr_insert.first->second.emplace(id);
    }
  }

  // Check consistency of cells included in trigger cell
  for (const auto& [tcid, cells] : triggercells_to_cells) {
    HGCalTriggerGeometryBase::geom_set cells_from_tc = es.geometry->getCellsFromTriggerCell(tcid);
    for (auto cell : cells) {
      if (cells_from_tc.find(cell) == cells_from_tc.end()) {
        errors_.fill(HGCalTriggerGeoTesterErrors::MissingCellInTC, tcid);
      }
    }
    for (auto cell : cells_from_tc) {
      if (cells.find(cell) == cells.end()) {
        errors_.fill(HGCalTriggerGeoTesterErrors::InvalidCellInTC, tcid);
      }
    }
  }
}

void HGCalTriggerGeoTesterTriggerCells::clear() {
  id_ = 0;
  valid_ = 0;
  errorbits_ = 0;
  disconnected_ = 0;
  zside_ = 0;
  subdet_ = 0;
  layer_ = 0;
  u_ = 0;
  v_ = 0;
  ieta_ = 0;
  iphi_ = 0;
  waferU_ = 0;
  waferV_ = 0;
  type_ = 0;
  modid_ = 0;
  x_ = 0.;
  y_ = 0.;
  z_ = 0.;
  eta_ = 0.;
  phi_ = 0.;
  cells_n_ = 0;
  cells_.clear();
}
