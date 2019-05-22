#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalTriggerDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetIdToROC.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <regex>

class HGCalTriggerGeometryV9Imp2 : public HGCalTriggerGeometryBase {
public:
  HGCalTriggerGeometryV9Imp2(const edm::ParameterSet& conf);

  void initialize(const edm::ESHandle<CaloGeometry>&) final;
  void initialize(const edm::ESHandle<HGCalGeometry>&,
                  const edm::ESHandle<HGCalGeometry>&,
                  const edm::ESHandle<HGCalGeometry>&) final;
  void reset() final;

  unsigned getTriggerCellFromCell(const unsigned) const final;
  unsigned getModuleFromCell(const unsigned) const final;
  unsigned getModuleFromTriggerCell(const unsigned) const final;

  geom_set getCellsFromTriggerCell(const unsigned) const final;
  geom_set getCellsFromModule(const unsigned) const final;
  geom_set getTriggerCellsFromModule(const unsigned) const final;

  geom_ordered_set getOrderedCellsFromModule(const unsigned) const final;
  geom_ordered_set getOrderedTriggerCellsFromModule(const unsigned) const final;

  geom_set getNeighborsFromTriggerCell(const unsigned) const final;

  GlobalPoint getTriggerCellPosition(const unsigned) const final;
  GlobalPoint getModulePosition(const unsigned) const final;

  bool validTriggerCell(const unsigned) const final;
  bool disconnectedModule(const unsigned) const final;
  unsigned triggerLayer(const unsigned) const final;

private:
  // HSc trigger cell grouping
  unsigned hSc_triggercell_size_ = 2;
  unsigned hSc_module_size_ = 12;  // in TC units (144 TC / panel = 36 e-links)

  edm::FileInPath l1tModulesMapping_;

  // module related maps
  std::unordered_map<unsigned, unsigned> wafer_to_module_;
  std::unordered_multimap<unsigned, unsigned> module_to_wafers_;

  // Disconnected modules and layers
  std::unordered_set<unsigned> disconnected_layers_;
  std::vector<unsigned> trigger_layers_;

  // layer offsets
  unsigned heOffset_;
  unsigned totalLayers_;

  void fillMaps();
  bool validCellId(unsigned det, unsigned cell_id) const;
  bool validTriggerCellFromCells(const unsigned) const;

  int detIdWaferType(unsigned det, unsigned layer, short waferU, short waferV) const;
  unsigned packWaferId(int waferU, int waferV) const;
  void unpackWaferId(unsigned wafer, int& waferU, int& waferV) const;

  unsigned layerWithOffset(unsigned) const;
};

HGCalTriggerGeometryV9Imp2::HGCalTriggerGeometryV9Imp2(const edm::ParameterSet& conf)
    : HGCalTriggerGeometryBase(conf),
      hSc_triggercell_size_(conf.getParameter<unsigned>("ScintillatorTriggerCellSize")),
      hSc_module_size_(conf.getParameter<unsigned>("ScintillatorModuleSize")),
      l1tModulesMapping_(conf.getParameter<edm::FileInPath>("L1TModulesMapping")),
      disconnected_layers_(conf.getParameter<std::vector<unsigned>>("DisconnectedLayers").begin(),
                           conf.getParameter<std::vector<unsigned>>("DisconnectedLayers").end()) {}

void HGCalTriggerGeometryV9Imp2::reset() {
  wafer_to_module_.clear();
  module_to_wafers_.clear();
}

void HGCalTriggerGeometryV9Imp2::initialize(const edm::ESHandle<CaloGeometry>& calo_geometry) {
  throw cms::Exception("BadGeometry")
      << "HGCalTriggerGeometryV9Imp2 geometry cannot be initialized with the V7/V8 HGCAL geometry";
}

void HGCalTriggerGeometryV9Imp2::initialize(const edm::ESHandle<HGCalGeometry>& hgc_ee_geometry,
                                            const edm::ESHandle<HGCalGeometry>& hgc_hsi_geometry,
                                            const edm::ESHandle<HGCalGeometry>& hgc_hsc_geometry) {
  setEEGeometry(hgc_ee_geometry);
  setHSiGeometry(hgc_hsi_geometry);
  setHScGeometry(hgc_hsc_geometry);
  heOffset_ = eeTopology().dddConstants().layers(true);
  totalLayers_ = heOffset_ + hsiTopology().dddConstants().layers(true);
  trigger_layers_.resize(totalLayers_ + 1);
  trigger_layers_[0] = 0;  // layer number 0 doesn't exist
  unsigned trigger_layer = 1;
  for (unsigned layer = 1; layer < trigger_layers_.size(); layer++) {
    if (disconnected_layers_.find(layer) == disconnected_layers_.end()) {
      // Increase trigger layer number if the layer is not disconnected
      trigger_layers_[layer] = trigger_layer;
      trigger_layer++;
    } else {
      trigger_layers_[layer] = 0;
    }
  }
  fillMaps();
}

unsigned HGCalTriggerGeometryV9Imp2::getTriggerCellFromCell(const unsigned cell_id) const {
  unsigned det = DetId(cell_id).det();
  unsigned trigger_cell_id = 0;
  // Scintillator
  if (det == DetId::HGCalHSc) {
    // Very rough mapping from cells to TC
    HGCScintillatorDetId cell_sc_id(cell_id);
    int ieta = ((cell_sc_id.ietaAbs() - 1) / hSc_triggercell_size_ + 1) * cell_sc_id.zside();
    int iphi = (cell_sc_id.iphi() - 1) / hSc_triggercell_size_ + 1;
    trigger_cell_id = HGCScintillatorDetId(cell_sc_id.type(), cell_sc_id.layer(), ieta, iphi);
  }
  // Silicon
  else if (det == DetId::HGCalEE || det == DetId::HGCalHSi) {
    HGCSiliconDetId cell_si_id(cell_id);
    trigger_cell_id = HGCalTriggerDetId(
        det == DetId::HGCalEE ? HGCalTriggerSubdetector::HGCalEETrigger : HGCalTriggerSubdetector::HGCalHSiTrigger,
        cell_si_id.zside(),
        cell_si_id.type(),
        cell_si_id.layer(),
        cell_si_id.waferU(),
        cell_si_id.waferV(),
        cell_si_id.triggerCellU(),
        cell_si_id.triggerCellV());
  }
  return trigger_cell_id;
}

unsigned HGCalTriggerGeometryV9Imp2::getModuleFromCell(const unsigned cell_id) const {
  return getModuleFromTriggerCell(getTriggerCellFromCell(cell_id));
}

unsigned HGCalTriggerGeometryV9Imp2::getModuleFromTriggerCell(const unsigned trigger_cell_id) const {
  unsigned det = DetId(trigger_cell_id).det();
  unsigned module = 0;
  unsigned subdet_old = 0;
  int zside = 0;
  unsigned tc_type = 1;
  unsigned layer = 0;
  unsigned module_id = 0;
  // Scintillator
  if (det == DetId::HGCalHSc) {
    HGCScintillatorDetId trigger_cell_sc_id(trigger_cell_id);
    tc_type = trigger_cell_sc_id.type();
    layer = trigger_cell_sc_id.layer();
    zside = trigger_cell_sc_id.zside();
    int ieta = ((trigger_cell_sc_id.ietaAbs() - 1) / hSc_module_size_ + 1) * zside;
    int iphi = (trigger_cell_sc_id.iphi() - 1) / hSc_module_size_ + 1;
    module_id = HGCScintillatorDetId(tc_type, layer, ieta, iphi);
  }
  // Silicon
  else {
    HGCalTriggerDetId trigger_cell_trig_id(trigger_cell_id);
    unsigned subdet = trigger_cell_trig_id.subdet();
    subdet_old =
        (subdet == HGCalTriggerSubdetector::HGCalEETrigger ? ForwardSubdetector::HGCEE : ForwardSubdetector::HGCHEF);
    layer = trigger_cell_trig_id.layer();
    zside = trigger_cell_trig_id.zside();
    if (subdet == HGCalTriggerSubdetector::HGCalEETrigger || subdet == HGCalTriggerSubdetector::HGCalHSiTrigger) {
      int waferu = trigger_cell_trig_id.waferU();
      int waferv = trigger_cell_trig_id.waferV();
      auto module_itr = wafer_to_module_.find(packWaferId(waferu, waferv));
      if (module_itr == wafer_to_module_.end()) {
        throw cms::Exception("BadGeometry")
            << trigger_cell_trig_id << "HGCalTriggerGeometry: Wafer (" << waferu << "," << waferv
            << ") is not mapped to any trigger module. The module mapping should be modified. \n";
      }
      module = module_itr->second;
    }
    module_id =
        HGCalDetId((ForwardSubdetector)subdet_old, zside, layer, tc_type, module, HGCalDetId::kHGCalCellMask).rawId();
  }
  return module_id;
}

HGCalTriggerGeometryBase::geom_set HGCalTriggerGeometryV9Imp2::getCellsFromTriggerCell(
    const unsigned trigger_cell_id) const {
  DetId trigger_cell_det_id(trigger_cell_id);
  unsigned det = trigger_cell_det_id.det();
  geom_set cell_det_ids;
  // Scintillator
  if (det == DetId::HGCalHSc) {
    HGCScintillatorDetId trigger_cell_sc_id(trigger_cell_id);
    int ieta0 = (trigger_cell_sc_id.ietaAbs() - 1) * hSc_triggercell_size_ + 1;
    int iphi0 = (trigger_cell_sc_id.iphi() - 1) * hSc_triggercell_size_ + 1;
    for (int ietaAbs = ieta0; ietaAbs < ieta0 + (int)hSc_triggercell_size_; ietaAbs++) {
      int ieta = ietaAbs * trigger_cell_sc_id.zside();
      for (int iphi = iphi0; iphi < iphi0 + (int)hSc_triggercell_size_; iphi++) {
        unsigned cell_id = HGCScintillatorDetId(trigger_cell_sc_id.type(), trigger_cell_sc_id.layer(), ieta, iphi);
        if (validCellId(DetId::HGCalHSc, cell_id))
          cell_det_ids.emplace(cell_id);
      }
    }
  }
  // Silicon
  else {
    HGCalTriggerDetId trigger_cell_trig_id(trigger_cell_id);
    unsigned subdet = trigger_cell_trig_id.subdet();
    if (subdet == HGCalTriggerSubdetector::HGCalEETrigger || subdet == HGCalTriggerSubdetector::HGCalHSiTrigger) {
      DetId::Detector cell_det = (subdet == HGCalTriggerSubdetector::HGCalEETrigger ? DetId::HGCalEE : DetId::HGCalHSi);
      int layer = trigger_cell_trig_id.layer();
      int zside = trigger_cell_trig_id.zside();
      int type = trigger_cell_trig_id.type();
      int waferu = trigger_cell_trig_id.waferU();
      int waferv = trigger_cell_trig_id.waferV();
      std::vector<int> cellus = trigger_cell_trig_id.cellU();
      std::vector<int> cellvs = trigger_cell_trig_id.cellV();
      for (unsigned ic = 0; ic < cellus.size(); ic++) {
        HGCSiliconDetId cell_det_id(cell_det, zside, type, layer, waferu, waferv, cellus[ic], cellvs[ic]);
        cell_det_ids.emplace(cell_det_id.rawId());
      }
    }
  }
  return cell_det_ids;
}

HGCalTriggerGeometryBase::geom_set HGCalTriggerGeometryV9Imp2::getCellsFromModule(const unsigned module_id) const {
  geom_set cell_det_ids;
  geom_set trigger_cells = getTriggerCellsFromModule(module_id);
  for (auto trigger_cell_id : trigger_cells) {
    geom_set cells = getCellsFromTriggerCell(trigger_cell_id);
    cell_det_ids.insert(cells.begin(), cells.end());
  }
  return cell_det_ids;
}

HGCalTriggerGeometryBase::geom_ordered_set HGCalTriggerGeometryV9Imp2::getOrderedCellsFromModule(
    const unsigned module_id) const {
  geom_ordered_set cell_det_ids;
  geom_ordered_set trigger_cells = getOrderedTriggerCellsFromModule(module_id);
  for (auto trigger_cell_id : trigger_cells) {
    geom_set cells = getCellsFromTriggerCell(trigger_cell_id);
    cell_det_ids.insert(cells.begin(), cells.end());
  }
  return cell_det_ids;
}

HGCalTriggerGeometryBase::geom_set HGCalTriggerGeometryV9Imp2::getTriggerCellsFromModule(
    const unsigned module_id) const {
  DetId module_det_id(module_id);
  unsigned det = module_det_id.det();
  geom_set trigger_cell_det_ids;
  // Scintillator
  if (det == DetId::HGCalHSc) {
    HGCScintillatorDetId module_sc_id(module_id);
    int ieta0 = (module_sc_id.ietaAbs() - 1) * hSc_module_size_ + 1;
    int iphi0 = (module_sc_id.iphi() - 1) * hSc_module_size_ + 1;
    for (int ietaAbs = ieta0; ietaAbs < ieta0 + (int)hSc_module_size_; ietaAbs++) {
      int ieta = ietaAbs * module_sc_id.zside();
      for (int iphi = iphi0; iphi < iphi0 + (int)hSc_module_size_; iphi++) {
        unsigned trigger_cell_id = HGCScintillatorDetId(module_sc_id.type(), module_sc_id.layer(), ieta, iphi);
        if (validTriggerCellFromCells(trigger_cell_id))
          trigger_cell_det_ids.emplace(trigger_cell_id);
      }
    }
  }
  // Silicon
  else {
    HGCalDetId module_si_id(module_id);
    unsigned module = module_si_id.wafer();
    HGCSiliconDetIdToROC tc2roc;
    auto wafer_itrs = module_to_wafers_.equal_range(module);
    // loop on the wafers included in the module
    for (auto wafer_itr = wafer_itrs.first; wafer_itr != wafer_itrs.second; wafer_itr++) {
      int waferu = 0;
      int waferv = 0;
      unpackWaferId(wafer_itr->second, waferu, waferv);
      DetId::Detector det = (module_si_id.subdetId() == ForwardSubdetector::HGCEE ? DetId::HGCalEE : DetId::HGCalHSi);
      HGCalTriggerSubdetector subdet =
          (module_si_id.subdetId() == ForwardSubdetector::HGCEE ? HGCalTriggerSubdetector::HGCalEETrigger
                                                                : HGCalTriggerSubdetector::HGCalHSiTrigger);
      unsigned layer = module_si_id.layer();
      unsigned wafer_type = detIdWaferType(det, layer, waferu, waferv);
      int nroc = (wafer_type == HGCSiliconDetId::HGCalFine ? 6 : 3);
      // Loop on ROCs in wafer
      for (int roc = 1; roc <= nroc; roc++) {
        // loop on TCs in ROC
        auto tc_uvs = tc2roc.getTriggerId(roc, wafer_type);
        for (const auto& tc_uv : tc_uvs) {
          HGCalTriggerDetId trigger_cell_id(
              subdet, module_si_id.zside(), wafer_type, layer, waferu, waferv, tc_uv.first, tc_uv.second);
          if (validTriggerCellFromCells(trigger_cell_id))
            trigger_cell_det_ids.emplace(trigger_cell_id.rawId());
        }
      }
    }
  }
  return trigger_cell_det_ids;
}

HGCalTriggerGeometryBase::geom_ordered_set HGCalTriggerGeometryV9Imp2::getOrderedTriggerCellsFromModule(
    const unsigned module_id) const {
  DetId module_det_id(module_id);
  unsigned det = module_det_id.det();
  geom_ordered_set trigger_cell_det_ids;
  // Scintillator
  if (det == DetId::HGCalHSc) {
    HGCScintillatorDetId module_sc_id(module_id);
    int ieta0 = (module_sc_id.ietaAbs() - 1) * hSc_module_size_ + 1;
    int iphi0 = (module_sc_id.iphi() - 1) * hSc_module_size_ + 1;
    for (int ietaAbs = ieta0; ietaAbs < ieta0 + (int)hSc_module_size_; ietaAbs++) {
      int ieta = ietaAbs * module_sc_id.zside();
      for (int iphi = iphi0; iphi < iphi0 + (int)hSc_module_size_; iphi++) {
        unsigned trigger_cell_id = HGCScintillatorDetId(module_sc_id.type(), module_sc_id.layer(), ieta, iphi);
        if (validTriggerCellFromCells(trigger_cell_id))
          trigger_cell_det_ids.emplace(trigger_cell_id);
      }
    }
  }
  // EE or FH
  else {
    HGCalDetId module_si_id(module_id);
    unsigned module = module_si_id.wafer();
    HGCSiliconDetIdToROC tc2roc;
    auto wafer_itrs = module_to_wafers_.equal_range(module);
    // loop on the wafers included in the module
    for (auto wafer_itr = wafer_itrs.first; wafer_itr != wafer_itrs.second; wafer_itr++) {
      int waferu = 0;
      int waferv = 0;
      unpackWaferId(wafer_itr->second, waferu, waferv);
      DetId::Detector det = (module_si_id.subdetId() == ForwardSubdetector::HGCEE ? DetId::HGCalEE : DetId::HGCalHSi);
      HGCalTriggerSubdetector subdet =
          (module_si_id.subdetId() == ForwardSubdetector::HGCEE ? HGCalTriggerSubdetector::HGCalEETrigger
                                                                : HGCalTriggerSubdetector::HGCalHSiTrigger);
      unsigned layer = module_si_id.layer();
      unsigned wafer_type = detIdWaferType(det, layer, waferu, waferv);
      int nroc = (wafer_type == HGCSiliconDetId::HGCalFine ? 6 : 3);
      // Loop on ROCs in wafer
      for (int roc = 1; roc <= nroc; roc++) {
        // loop on TCs in ROC
        auto tc_uvs = tc2roc.getTriggerId(roc, wafer_type);
        for (const auto& tc_uv : tc_uvs) {
          HGCalTriggerDetId trigger_cell_id(
              subdet, module_si_id.zside(), wafer_type, layer, waferu, waferv, tc_uv.first, tc_uv.second);
          trigger_cell_det_ids.emplace(trigger_cell_id.rawId());
        }
      }
    }
  }
  return trigger_cell_det_ids;
}

HGCalTriggerGeometryBase::geom_set HGCalTriggerGeometryV9Imp2::getNeighborsFromTriggerCell(
    const unsigned trigger_cell_id) const {
  HGCalDetId trigger_cell_det_id(trigger_cell_id);
  geom_set neighbor_detids;
  return neighbor_detids;
}

GlobalPoint HGCalTriggerGeometryV9Imp2::getTriggerCellPosition(const unsigned trigger_cell_det_id) const {
  unsigned det = DetId(trigger_cell_det_id).det();
  // Position: barycenter of the trigger cell.
  Basic3DVector<float> triggerCellVector(0., 0., 0.);
  const auto cell_ids = getCellsFromTriggerCell(trigger_cell_det_id);
  // Scintillator
  if (det == DetId::HGCalHSc) {
    for (const auto& cell : cell_ids) {
      HGCScintillatorDetId cellDetId(cell);
      triggerCellVector += hscGeometry()->getPosition(cellDetId).basicVector();
    }
  }
  // Silicon
  else {
    for (const auto& cell : cell_ids) {
      HGCSiliconDetId cellDetId(cell);
      triggerCellVector += (cellDetId.det() == DetId::HGCalEE ? eeGeometry()->getPosition(cellDetId)
                                                              : hsiGeometry()->getPosition(cellDetId))
                               .basicVector();
    }
  }
  return GlobalPoint(triggerCellVector / cell_ids.size());
}

GlobalPoint HGCalTriggerGeometryV9Imp2::getModulePosition(const unsigned module_det_id) const {
  unsigned det = DetId(module_det_id).det();
  // Position: barycenter of the module.
  Basic3DVector<float> moduleVector(0., 0., 0.);
  const auto cell_ids = getCellsFromModule(module_det_id);
  // Scintillator
  if (det == DetId::HGCalHSc) {
    for (const auto& cell : cell_ids) {
      HGCScintillatorDetId cellDetId(cell);
      moduleVector += hscGeometry()->getPosition(cellDetId).basicVector();
    }
  }
  // Silicon
  else {
    for (const auto& cell : cell_ids) {
      HGCSiliconDetId cellDetId(cell);
      moduleVector += (cellDetId.det() == DetId::HGCalEE ? eeGeometry()->getPosition(cellDetId)
                                                         : hsiGeometry()->getPosition(cellDetId))
                          .basicVector();
    }
  }
  return GlobalPoint(moduleVector / cell_ids.size());
}

void HGCalTriggerGeometryV9Imp2::fillMaps() {
  // read module mapping file
  std::ifstream l1tModulesMappingStream(l1tModulesMapping_.fullPath());
  if (!l1tModulesMappingStream.is_open()) {
    throw cms::Exception("MissingDataFile") << "Cannot open HGCalTriggerGeometry L1TModulesMapping file\n";
  }
  short waferu = 0;
  short waferv = 0;
  short module = 0;
  for (; l1tModulesMappingStream >> waferu >> waferv >> module;) {
    unsigned wafer_key = packWaferId(waferu, waferv);
    wafer_to_module_.emplace(wafer_key, module);
    module_to_wafers_.emplace(module, wafer_key);
  }
  if (!l1tModulesMappingStream.eof()) {
    throw cms::Exception("BadGeometryFile")
        << "Error reading L1TModulesMapping '" << waferu << " " << waferv << " " << module << "' \n";
  }
  l1tModulesMappingStream.close();
}

unsigned HGCalTriggerGeometryV9Imp2::packWaferId(int waferU, int waferV) const {
  unsigned packed_value = 0;
  unsigned waferUsign = (waferU >= 0) ? 0 : 1;
  unsigned waferVsign = (waferV >= 0) ? 0 : 1;
  packed_value |= ((std::abs(waferU) & HGCSiliconDetId::kHGCalWaferUMask) << HGCSiliconDetId::kHGCalWaferUOffset);
  packed_value |= ((waferUsign & HGCSiliconDetId::kHGCalWaferUSignMask) << HGCSiliconDetId::kHGCalWaferUSignOffset);
  packed_value |= ((std::abs(waferV) & HGCSiliconDetId::kHGCalWaferVMask) << HGCSiliconDetId::kHGCalWaferVOffset);
  packed_value |= ((waferVsign & HGCSiliconDetId::kHGCalWaferVSignMask) << HGCSiliconDetId::kHGCalWaferVSignOffset);
  return packed_value;
}

void HGCalTriggerGeometryV9Imp2::unpackWaferId(unsigned wafer, int& waferU, int& waferV) const {
  unsigned waferUAbs = (wafer >> HGCSiliconDetId::kHGCalWaferUOffset) & HGCSiliconDetId::kHGCalWaferUMask;
  unsigned waferVAbs = (wafer >> HGCSiliconDetId::kHGCalWaferVOffset) & HGCSiliconDetId::kHGCalWaferVMask;
  waferU = (((wafer >> HGCSiliconDetId::kHGCalWaferUSignOffset) & HGCSiliconDetId::kHGCalWaferUSignMask) ? -waferUAbs
                                                                                                         : waferUAbs);
  waferV = (((wafer >> HGCSiliconDetId::kHGCalWaferVSignOffset) & HGCSiliconDetId::kHGCalWaferVSignMask) ? -waferVAbs
                                                                                                         : waferVAbs);
}

bool HGCalTriggerGeometryV9Imp2::validTriggerCell(const unsigned trigger_cell_id) const {
  return validTriggerCellFromCells(trigger_cell_id);
}

bool HGCalTriggerGeometryV9Imp2::disconnectedModule(const unsigned module_id) const {
  bool disconnected = false;
  if (disconnected_layers_.find(layerWithOffset(module_id)) != disconnected_layers_.end())
    disconnected = true;
  return disconnected;
}

unsigned HGCalTriggerGeometryV9Imp2::triggerLayer(const unsigned id) const {
  unsigned layer = layerWithOffset(id);
  if (layer >= trigger_layers_.size())
    return 0;
  return trigger_layers_[layer];
}

bool HGCalTriggerGeometryV9Imp2::validTriggerCellFromCells(const unsigned trigger_cell_id) const {
  // Check the validity of a trigger cell with the
  // validity of the cells. One valid cell in the
  // trigger cell is enough to make the trigger cell
  // valid.
  const geom_set cells = getCellsFromTriggerCell(trigger_cell_id);
  bool is_valid = false;
  for (const auto cell_id : cells) {
    unsigned det = DetId(cell_id).det();
    is_valid |= validCellId(det, cell_id);
    if (is_valid)
      break;
  }
  return is_valid;
}

bool HGCalTriggerGeometryV9Imp2::validCellId(unsigned subdet, unsigned cell_id) const {
  bool is_valid = false;
  switch (subdet) {
    case DetId::HGCalEE:
      is_valid = eeTopology().valid(cell_id);
      break;
    case DetId::HGCalHSi:
      is_valid = hsiTopology().valid(cell_id);
      break;
    case DetId::HGCalHSc:
      is_valid = hscTopology().valid(cell_id);
      break;
    default:
      is_valid = false;
      break;
  }
  return is_valid;
}

int HGCalTriggerGeometryV9Imp2::detIdWaferType(unsigned det, unsigned layer, short waferU, short waferV) const {
  int wafer_type = 0;
  switch (det) {
    case DetId::HGCalEE:
      wafer_type = eeTopology().dddConstants().getTypeHex(layer, waferU, waferV);
      break;
    case DetId::HGCalHSi:
      wafer_type = hsiTopology().dddConstants().getTypeHex(layer, waferU, waferV);
      break;
    default:
      break;
  };
  return wafer_type;
}

unsigned HGCalTriggerGeometryV9Imp2::layerWithOffset(unsigned id) const {
  unsigned det = DetId(id).det();
  unsigned layer = 0;
  if (det == DetId::HGCalTrigger) {
    unsigned subdet = HGCalTriggerDetId(id).subdet();
    if (subdet == HGCalTriggerSubdetector::HGCalEETrigger) {
      layer = HGCalTriggerDetId(id).layer();
    } else if (subdet == HGCalTriggerSubdetector::HGCalHSiTrigger) {
      layer = heOffset_ + HGCalTriggerDetId(id).layer();
    }
  } else if (det == DetId::HGCalHSc) {
    layer = heOffset_ + HGCScintillatorDetId(id).layer();
  } else if (det == DetId::Forward) {
    unsigned subdet = HGCalDetId(id).subdetId();
    if (subdet == ForwardSubdetector::HGCEE) {
      layer = HGCalDetId(id).layer();
    } else if (subdet == ForwardSubdetector::HGCHEF || subdet == ForwardSubdetector::HGCHEB) {
      layer = heOffset_ + HGCalTriggerDetId(id).layer();
    }
  }
  return layer;
}

DEFINE_EDM_PLUGIN(HGCalTriggerGeometryFactory, HGCalTriggerGeometryV9Imp2, "HGCalTriggerGeometryV9Imp2");
