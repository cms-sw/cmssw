#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

#include <fstream>
#include <iostream>
#include <regex>
#include <vector>

class HGCalTriggerGeometryHexImp2 : public HGCalTriggerGeometryBase {
public:
  HGCalTriggerGeometryHexImp2(const edm::ParameterSet& conf);

  void initialize(const CaloGeometry*) final;
  void initialize(const HGCalGeometry*, const HGCalGeometry*, const HGCalGeometry*) final;
  void initialize(const HGCalGeometry*, const HGCalGeometry*, const HGCalGeometry*, const HGCalGeometry*) final;
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

  unsigned getLinksInModule(const unsigned module_id) const final;
  unsigned getModuleSize(const unsigned module_id) const final;

  GlobalPoint getTriggerCellPosition(const unsigned) const final;
  GlobalPoint getModulePosition(const unsigned) const final;

  bool validTriggerCell(const unsigned) const final;
  bool disconnectedModule(const unsigned) const final;
  unsigned lastTriggerLayer() const final;
  unsigned triggerLayer(const unsigned) const final;

private:
  edm::FileInPath l1tCellsMapping_;
  edm::FileInPath l1tCellNeighborsMapping_;
  edm::FileInPath l1tWaferNeighborsMapping_;
  edm::FileInPath l1tModulesMapping_;

  // module related maps
  std::unordered_map<short, short> wafer_to_module_ee_;
  std::unordered_map<short, short> wafer_to_module_fh_;
  std::unordered_multimap<short, short> module_to_wafers_ee_;
  std::unordered_multimap<short, short> module_to_wafers_fh_;

  // trigger cell related maps
  std::map<std::pair<short, short>, short> cells_to_trigger_cells_;       // FIXME: something else than map<pair,short>?
  std::multimap<std::pair<short, short>, short> trigger_cells_to_cells_;  // FIXME: something else than map<pair,short>?
  std::unordered_map<short, short> number_trigger_cells_in_wafers_;       // the map key is the wafer type
  std::unordered_map<short, short> number_cells_in_wafers_;               // the map key is the wafer type
  std::unordered_set<unsigned> invalid_triggercells_;

  // neighbor related maps
  // trigger cell neighbors:
  // - The key includes the trigger cell id and the wafer configuration.
  // The wafer configuration is a 7 bits word encoding the type
  // (small or large cells) of the wafer containing the trigger cell
  // (central wafer) as well as the type of the 6 surrounding wafers
  // - The value is a set of (wafer_idx, trigger_cell_id)
  // wafer_idx is a number between 0 and 7. 0=central wafer, 1..7=surrounding
  // wafers
  std::unordered_map<int, std::set<std::pair<short, short>>> trigger_cell_neighbors_;
  // wafer neighbors:
  // List of the 6 surrounding neighbors around each wafer
  std::unordered_map<short, std::vector<short>> wafer_neighbors_ee_;
  std::unordered_map<short, std::vector<short>> wafer_neighbors_fh_;

  void fillMaps();
  void fillNeighborMaps();
  void fillInvalidTriggerCells();
  unsigned packTriggerCell(unsigned, const std::vector<int>&) const;
  // returns transverse wafer type: -1=coarse, 1=fine, 0=undefined
  int detIdWaferType(unsigned subdet, short wafer) const;
  bool validCellId(unsigned subdet, unsigned cell_id) const;
  bool validTriggerCellFromCells(const unsigned) const;
};

HGCalTriggerGeometryHexImp2::HGCalTriggerGeometryHexImp2(const edm::ParameterSet& conf)
    : HGCalTriggerGeometryBase(conf),
      l1tCellsMapping_(conf.getParameter<edm::FileInPath>("L1TCellsMapping")),
      l1tCellNeighborsMapping_(conf.getParameter<edm::FileInPath>("L1TCellNeighborsMapping")),
      l1tWaferNeighborsMapping_(conf.getParameter<edm::FileInPath>("L1TWaferNeighborsMapping")),
      l1tModulesMapping_(conf.getParameter<edm::FileInPath>("L1TModulesMapping")) {}

void HGCalTriggerGeometryHexImp2::reset() {
  wafer_to_module_ee_.clear();
  wafer_to_module_fh_.clear();
  module_to_wafers_ee_.clear();
  module_to_wafers_fh_.clear();
  cells_to_trigger_cells_.clear();
  trigger_cells_to_cells_.clear();
  number_trigger_cells_in_wafers_.clear();
  number_cells_in_wafers_.clear();
}

void HGCalTriggerGeometryHexImp2::initialize(const CaloGeometry* calo_geometry) {
  setCaloGeometry(calo_geometry);
  fillMaps();
  fillNeighborMaps();
  fillInvalidTriggerCells();
}

void HGCalTriggerGeometryHexImp2::initialize(const HGCalGeometry* hgc_ee_geometry,
                                             const HGCalGeometry* hgc_hsi_geometry,
                                             const HGCalGeometry* hgc_hsc_geometry) {
  throw cms::Exception("BadGeometry")
      << "HGCalTriggerGeometryHexImp2 geometry cannot be initialized with the V9 HGCAL geometry";
}

void HGCalTriggerGeometryHexImp2::initialize(const HGCalGeometry* hgc_ee_geometry,
                                             const HGCalGeometry* hgc_hsi_geometry,
                                             const HGCalGeometry* hgc_hsc_geometry,
                                             const HGCalGeometry* hgc_nose_geometry) {
  throw cms::Exception("BadGeometry")
      << "HGCalTriggerGeometryHexImp2 geometry cannot be initialized with the V9 HGCAL+Nose geometry";
}

unsigned HGCalTriggerGeometryHexImp2::getTriggerCellFromCell(const unsigned cell_id) const {
  if (DetId(cell_id).det() == DetId::Hcal)
    return 0;
  HGCalDetId cell_det_id(cell_id);
  int wafer_type = cell_det_id.waferType();
  unsigned cell = cell_det_id.cell();
  // FIXME: better way to do this cell->TC mapping?
  auto trigger_cell_itr = cells_to_trigger_cells_.find(std::make_pair(wafer_type, cell));
  if (trigger_cell_itr == cells_to_trigger_cells_.end()) {
    throw cms::Exception("BadGeometry") << "HGCalTriggerGeometry: HGCal  cell " << cell
                                        << " is not mapped to any trigger cell for the wafer type " << wafer_type
                                        << ". The trigger cell mapping should be modified.\n";
  }
  unsigned trigger_cell = trigger_cell_itr->second;
  return HGCalDetId((ForwardSubdetector)cell_det_id.subdetId(),
                    cell_det_id.zside(),
                    cell_det_id.layer(),
                    cell_det_id.waferType(),
                    cell_det_id.wafer(),
                    trigger_cell)
      .rawId();
}

unsigned HGCalTriggerGeometryHexImp2::getModuleFromCell(const unsigned cell_id) const {
  if (DetId(cell_id).det() == DetId::Hcal)
    return 0;
  HGCalDetId cell_det_id(cell_id);
  unsigned wafer = cell_det_id.wafer();
  unsigned subdet = cell_det_id.subdetId();
  std::unordered_map<short, short>::const_iterator module_itr;
  bool out_of_range_error = false;
  switch (subdet) {
    case ForwardSubdetector::HGCEE:
      module_itr = wafer_to_module_ee_.find(wafer);
      if (module_itr == wafer_to_module_ee_.end())
        out_of_range_error = true;
      break;
    case ForwardSubdetector::HGCHEF:
      module_itr = wafer_to_module_fh_.find(wafer);
      if (module_itr == wafer_to_module_fh_.end())
        out_of_range_error = true;
      break;
    default:
      edm::LogError("HGCalTriggerGeometry") << "Unknown wafer->module mapping for subdet " << subdet << "\n";
      return 0;
  };
  if (out_of_range_error) {
    throw cms::Exception("BadGeometry") << "HGCalTriggerGeometry: Wafer " << wafer
                                        << " is not mapped to any trigger module for subdetector " << subdet
                                        << ". The module mapping should be modified. See "
                                           "https://twiki.cern.ch/twiki/bin/viewauth/CMS/"
                                           "HGCALTriggerPrimitivesSimulation#Trigger_geometry for details.\n";
  }
  unsigned module = module_itr->second;
  return HGCalDetId((ForwardSubdetector)cell_det_id.subdetId(),
                    cell_det_id.zside(),
                    cell_det_id.layer(),
                    cell_det_id.waferType(),
                    module,
                    HGCalDetId::kHGCalCellMask)
      .rawId();
}

unsigned HGCalTriggerGeometryHexImp2::getModuleFromTriggerCell(const unsigned trigger_cell_id) const {
  HGCalDetId trigger_cell_det_id(trigger_cell_id);
  unsigned wafer = trigger_cell_det_id.wafer();
  unsigned subdet = trigger_cell_det_id.subdetId();
  if (subdet == ForwardSubdetector::HGCHEB)
    return 0;
  std::unordered_map<short, short>::const_iterator module_itr;
  bool out_of_range_error = false;
  switch (subdet) {
    case ForwardSubdetector::HGCEE:
      module_itr = wafer_to_module_ee_.find(wafer);
      if (module_itr == wafer_to_module_ee_.end())
        out_of_range_error = true;
      break;
    case ForwardSubdetector::HGCHEF:
      module_itr = wafer_to_module_fh_.find(wafer);
      if (module_itr == wafer_to_module_fh_.end())
        out_of_range_error = true;
      break;
    default:
      edm::LogError("HGCalTriggerGeometry") << "Unknown wafer->module mapping for subdet " << subdet << "\n";
      return 0;
  };
  if (out_of_range_error) {
    throw cms::Exception("BadGeometry") << "HGCalTriggerGeometry: Wafer " << wafer
                                        << " is not mapped to any trigger module for subdetector " << subdet
                                        << ". The module mapping should be modified. See "
                                           "https://twiki.cern.ch/twiki/bin/viewauth/CMS/"
                                           "HGCALTriggerPrimitivesSimulation#Trigger_geometry for details.\n";
  }
  unsigned module = module_itr->second;
  return HGCalDetId((ForwardSubdetector)trigger_cell_det_id.subdetId(),
                    trigger_cell_det_id.zside(),
                    trigger_cell_det_id.layer(),
                    trigger_cell_det_id.waferType(),
                    module,
                    HGCalDetId::kHGCalCellMask)
      .rawId();
}

HGCalTriggerGeometryBase::geom_set HGCalTriggerGeometryHexImp2::getCellsFromTriggerCell(
    const unsigned trigger_cell_id) const {
  HGCalDetId trigger_cell_det_id(trigger_cell_id);
  unsigned subdet = trigger_cell_det_id.subdetId();
  if (subdet == ForwardSubdetector::HGCHEB)
    return geom_set();
  int wafer_type = trigger_cell_det_id.waferType();
  unsigned trigger_cell = trigger_cell_det_id.cell();
  // FIXME: better way to do this TC->cell mapping?
  const auto& cell_range = trigger_cells_to_cells_.equal_range(std::make_pair(wafer_type, trigger_cell));
  geom_set cell_det_ids;
  for (auto tc_c_itr = cell_range.first; tc_c_itr != cell_range.second; tc_c_itr++) {
    cell_det_ids.emplace(HGCalDetId((ForwardSubdetector)trigger_cell_det_id.subdetId(),
                                    trigger_cell_det_id.zside(),
                                    trigger_cell_det_id.layer(),
                                    trigger_cell_det_id.waferType(),
                                    trigger_cell_det_id.wafer(),
                                    tc_c_itr->second)
                             .rawId());
  }
  return cell_det_ids;
}

HGCalTriggerGeometryBase::geom_set HGCalTriggerGeometryHexImp2::getCellsFromModule(const unsigned module_id) const {
  HGCalDetId module_det_id(module_id);
  unsigned subdet = module_det_id.subdetId();
  if (subdet == ForwardSubdetector::HGCHEB)
    return geom_set();
  unsigned module = module_det_id.wafer();
  std::pair<std::unordered_multimap<short, short>::const_iterator, std::unordered_multimap<short, short>::const_iterator>
      wafer_itrs;
  switch (subdet) {
    case ForwardSubdetector::HGCEE:
      wafer_itrs = module_to_wafers_ee_.equal_range(module);
      break;
    case ForwardSubdetector::HGCHEF:
      wafer_itrs = module_to_wafers_fh_.equal_range(module);
      break;
    default:
      edm::LogError("HGCalTriggerGeometry") << "Unknown module->wafers mapping for subdet " << subdet << "\n";
      return geom_set();
  };
  geom_set cell_det_ids;
  for (auto wafer_itr = wafer_itrs.first; wafer_itr != wafer_itrs.second; wafer_itr++) {
    int wafer_type = detIdWaferType(subdet, wafer_itr->second);
    if (wafer_type == 0)
      wafer_type = module_det_id.waferType();
    // loop on the cells in each wafer and return valid ones
    for (int cell = 0; cell < number_cells_in_wafers_.at(wafer_type); cell++) {
      HGCalDetId cell_id((ForwardSubdetector)module_det_id.subdetId(),
                         module_det_id.zside(),
                         module_det_id.layer(),
                         wafer_type,
                         wafer_itr->second,
                         cell);
      if (validCellId(subdet, cell_id))
        cell_det_ids.emplace(cell_id.rawId());
    }
  }
  return cell_det_ids;
}

HGCalTriggerGeometryBase::geom_ordered_set HGCalTriggerGeometryHexImp2::getOrderedCellsFromModule(
    const unsigned module_id) const {
  HGCalDetId module_det_id(module_id);
  unsigned subdet = module_det_id.subdetId();
  if (subdet == ForwardSubdetector::HGCHEB)
    return geom_ordered_set();
  unsigned module = module_det_id.wafer();
  std::pair<std::unordered_multimap<short, short>::const_iterator, std::unordered_multimap<short, short>::const_iterator>
      wafer_itrs;
  switch (subdet) {
    case ForwardSubdetector::HGCEE:
      wafer_itrs = module_to_wafers_ee_.equal_range(module);
      break;
    case ForwardSubdetector::HGCHEF:
      wafer_itrs = module_to_wafers_fh_.equal_range(module);
      break;
    default:
      edm::LogError("HGCalTriggerGeometry") << "Unknown module->wafers mapping for subdet " << subdet << "\n";
      return geom_ordered_set();
  };
  geom_ordered_set cell_det_ids;
  for (auto wafer_itr = wafer_itrs.first; wafer_itr != wafer_itrs.second; wafer_itr++) {
    int wafer_type = detIdWaferType(subdet, wafer_itr->second);
    if (wafer_type == 0)
      wafer_type = module_det_id.waferType();
    // loop on the cells in each wafer
    for (int cell = 0; cell < number_cells_in_wafers_.at(wafer_type); cell++) {
      HGCalDetId cell_id((ForwardSubdetector)module_det_id.subdetId(),
                         module_det_id.zside(),
                         module_det_id.layer(),
                         wafer_type,
                         wafer_itr->second,
                         cell);
      if (validCellId(subdet, cell_id))
        cell_det_ids.emplace(cell_id.rawId());
    }
  }
  return cell_det_ids;
}

HGCalTriggerGeometryBase::geom_set HGCalTriggerGeometryHexImp2::getTriggerCellsFromModule(
    const unsigned module_id) const {
  HGCalDetId module_det_id(module_id);
  unsigned subdet = module_det_id.subdetId();
  if (subdet == ForwardSubdetector::HGCHEB)
    return geom_set();
  unsigned module = module_det_id.wafer();
  std::pair<std::unordered_multimap<short, short>::const_iterator, std::unordered_multimap<short, short>::const_iterator>
      wafer_itrs;
  switch (subdet) {
    case ForwardSubdetector::HGCEE:
      wafer_itrs = module_to_wafers_ee_.equal_range(module);
      break;
    case ForwardSubdetector::HGCHEF:
      wafer_itrs = module_to_wafers_fh_.equal_range(module);
      break;
    default:
      edm::LogError("HGCalTriggerGeometry") << "Unknown module->wafers mapping for subdet " << subdet << "\n";
      return geom_set();
  };
  geom_set trigger_cell_det_ids;
  // loop on the wafers included in the module
  for (auto wafer_itr = wafer_itrs.first; wafer_itr != wafer_itrs.second; wafer_itr++) {
    int wafer_type = detIdWaferType(subdet, wafer_itr->second);
    if (wafer_type == 0)
      wafer_type = module_det_id.waferType();
    // loop on the trigger cells in each wafer
    for (int trigger_cell = 0; trigger_cell < number_trigger_cells_in_wafers_.at(wafer_type); trigger_cell++) {
      HGCalDetId trigger_cell_id((ForwardSubdetector)module_det_id.subdetId(),
                                 module_det_id.zside(),
                                 module_det_id.layer(),
                                 wafer_type,
                                 wafer_itr->second,
                                 trigger_cell);
      if (validTriggerCell(trigger_cell_id))
        trigger_cell_det_ids.emplace(trigger_cell_id.rawId());
    }
  }
  return trigger_cell_det_ids;
}

HGCalTriggerGeometryBase::geom_ordered_set HGCalTriggerGeometryHexImp2::getOrderedTriggerCellsFromModule(
    const unsigned module_id) const {
  HGCalDetId module_det_id(module_id);
  unsigned subdet = module_det_id.subdetId();
  if (subdet == ForwardSubdetector::HGCHEB)
    return geom_ordered_set();
  unsigned module = module_det_id.wafer();
  std::pair<std::unordered_multimap<short, short>::const_iterator, std::unordered_multimap<short, short>::const_iterator>
      wafer_itrs;
  switch (subdet) {
    case ForwardSubdetector::HGCEE:
      wafer_itrs = module_to_wafers_ee_.equal_range(module);
      break;
    case ForwardSubdetector::HGCHEF:
      wafer_itrs = module_to_wafers_fh_.equal_range(module);
      break;
    default:
      edm::LogError("HGCalTriggerGeometry") << "Unknown module->wafers mapping for subdet " << subdet << "\n";
      return geom_ordered_set();
  };
  geom_ordered_set trigger_cell_det_ids;
  // loop on the wafers included in the module
  for (auto wafer_itr = wafer_itrs.first; wafer_itr != wafer_itrs.second; wafer_itr++) {
    int wafer_type = detIdWaferType(subdet, wafer_itr->second);
    if (wafer_type == 0)
      wafer_type = module_det_id.waferType();
    // loop on the trigger cells in each wafer
    for (int trigger_cell = 0; trigger_cell < number_trigger_cells_in_wafers_.at(wafer_type); trigger_cell++) {
      HGCalDetId trigger_cell_id((ForwardSubdetector)module_det_id.subdetId(),
                                 module_det_id.zside(),
                                 module_det_id.layer(),
                                 wafer_type,
                                 wafer_itr->second,
                                 trigger_cell);
      if (validTriggerCell(trigger_cell_id))
        trigger_cell_det_ids.emplace(trigger_cell_id.rawId());
    }
  }
  return trigger_cell_det_ids;
}

HGCalTriggerGeometryBase::geom_set HGCalTriggerGeometryHexImp2::getNeighborsFromTriggerCell(
    const unsigned trigger_cell_id) const {
  HGCalDetId trigger_cell_det_id(trigger_cell_id);
  unsigned wafer = trigger_cell_det_id.wafer();
  int wafer_type = trigger_cell_det_id.waferType();
  unsigned subdet = trigger_cell_det_id.subdetId();
  unsigned trigger_cell = trigger_cell_det_id.cell();
  if (subdet == ForwardSubdetector::HGCHEB)
    return geom_set();
  // Retrieve surrounding wafers (around the wafer containing
  // the trigger cell)
  std::unordered_map<short, std::vector<short>>::const_iterator surrounding_wafers_itr;
  bool out_of_range_error = false;
  switch (subdet) {
    case ForwardSubdetector::HGCEE:
      surrounding_wafers_itr = wafer_neighbors_ee_.find(wafer);
      if (surrounding_wafers_itr == wafer_neighbors_ee_.end())
        out_of_range_error = true;
      break;
    case ForwardSubdetector::HGCHEF:
      surrounding_wafers_itr = wafer_neighbors_fh_.find(wafer);
      if (surrounding_wafers_itr == wafer_neighbors_fh_.end())
        out_of_range_error = true;
      break;
    default:
      edm::LogError("HGCalTriggerGeometry") << "Unknown wafer neighbours for subdet " << subdet << "\n";
      return geom_set();
  }
  if (out_of_range_error) {
    throw cms::Exception("BadGeometry") << "HGCalTriggerGeometry: Neighbors are not defined for wafer " << wafer
                                        << " in subdetector " << subdet
                                        << ". The wafer neighbor mapping should be modified. \n";
  };
  const std::vector<short>& surrounding_wafers = surrounding_wafers_itr->second;
  // Find the types of the surrounding wafers
  std::vector<int> types;
  types.reserve(surrounding_wafers.size() + 1);  // includes the central wafer -> +1
  types.emplace_back(wafer_type);
  for (const auto w : surrounding_wafers) {
    // if no neighbor, use the same type as the central one
    // to create the wafer configuration
    int wt = wafer_type;
    if (w != -1)
      wt = detIdWaferType(subdet, w);
    if (wt == 0)
      return geom_set();  // invalid wafer type
    types.emplace_back(wt);
  }
  // retrieve neighbors
  unsigned trigger_cell_key = packTriggerCell(trigger_cell, types);
  geom_set neighbor_detids;
  auto neighbors_itr = trigger_cell_neighbors_.find(trigger_cell_key);
  if (neighbors_itr == trigger_cell_neighbors_.end()) {
    throw cms::Exception("BadGeometry") << "HGCalTriggerGeometry: Neighbors are not defined for trigger cell "
                                        << trigger_cell << " with  wafer configuration "
                                        << std::bitset<7>(trigger_cell_key >> 8)
                                        << ". The trigger cell neighbor mapping should be modified. \n";
  }
  const auto& neighbors = neighbors_itr->second;
  // create HGCalDetId of neighbors and check their validity
  neighbor_detids.reserve(neighbors.size());
  for (const auto& wafer_tc : neighbors) {
    if (wafer_tc.first - 1 >= (int)surrounding_wafers.size()) {
      throw cms::Exception("BadGeometry")
          << "HGCalTriggerGeometry: Undefined wafer neighbor number " << wafer_tc.first << " for wafer " << wafer
          << " and trigger cell " << trigger_cell << ". The neighbor mapping files should be modified.";
    }
    int neighbor_wafer = (wafer_tc.first == 0 ? wafer : surrounding_wafers.at(wafer_tc.first - 1));
    if (neighbor_wafer == -1)
      continue;  // non-existing wafer
    int type = types.at(wafer_tc.first);
    HGCalDetId neighbor_det_id((ForwardSubdetector)trigger_cell_det_id.subdetId(),
                               trigger_cell_det_id.zside(),
                               trigger_cell_det_id.layer(),
                               type,
                               neighbor_wafer,
                               wafer_tc.second);
    if (validTriggerCell(neighbor_det_id.rawId())) {
      neighbor_detids.emplace(neighbor_det_id.rawId());
    }
  }
  return neighbor_detids;
}

unsigned HGCalTriggerGeometryHexImp2::getLinksInModule(const unsigned module_id) const { return 1; }

unsigned HGCalTriggerGeometryHexImp2::getModuleSize(const unsigned module_id) const { return 1; }

GlobalPoint HGCalTriggerGeometryHexImp2::getTriggerCellPosition(const unsigned trigger_cell_det_id) const {
  // Position: barycenter of the trigger cell.
  Basic3DVector<float> triggerCellVector(0., 0., 0.);
  const auto cell_ids = getCellsFromTriggerCell(trigger_cell_det_id);
  if (cell_ids.empty())
    return GlobalPoint(0, 0, 0);
  for (const auto& cell : cell_ids) {
    HGCalDetId cellDetId(cell);
    triggerCellVector += (cellDetId.subdetId() == ForwardSubdetector::HGCEE ? eeGeometry()->getPosition(cellDetId)
                                                                            : fhGeometry()->getPosition(cellDetId))
                             .basicVector();
  }
  return GlobalPoint(triggerCellVector / cell_ids.size());
}

GlobalPoint HGCalTriggerGeometryHexImp2::getModulePosition(const unsigned module_det_id) const {
  // Position: barycenter of the module.
  Basic3DVector<float> moduleVector(0., 0., 0.);
  const auto cell_ids = getCellsFromModule(module_det_id);
  if (cell_ids.empty())
    return GlobalPoint(0, 0, 0);
  for (const auto& cell : cell_ids) {
    HGCalDetId cellDetId(cell);
    moduleVector += (cellDetId.subdetId() == ForwardSubdetector::HGCEE ? eeGeometry()->getPosition(cellDetId)
                                                                       : fhGeometry()->getPosition(cellDetId))
                        .basicVector();
  }
  return GlobalPoint(moduleVector / cell_ids.size());
}

void HGCalTriggerGeometryHexImp2::fillMaps() {
  //
  // read module mapping file
  std::ifstream l1tModulesMappingStream(l1tModulesMapping_.fullPath());
  if (!l1tModulesMappingStream.is_open()) {
    throw cms::Exception("MissingDataFile") << "Cannot open HGCalTriggerGeometry L1TModulesMapping file\n";
  }
  short subdet = 0;
  short wafer = 0;
  short module = 0;
  for (; l1tModulesMappingStream >> subdet >> wafer >> module;) {
    int wafer_type = detIdWaferType(subdet, wafer);
    switch (subdet) {
      case ForwardSubdetector::HGCEE: {
        // fill module <-> wafers mappings
        wafer_to_module_ee_.emplace(wafer, module);
        module_to_wafers_ee_.emplace(module, wafer);
        // fill number of cells for a given wafer type
        number_cells_in_wafers_.emplace(wafer_type, eeTopology().dddConstants().numberCellsHexagon(wafer));
        break;
      }
      case ForwardSubdetector::HGCHEF: {
        // fill module <-> wafers mappings
        wafer_to_module_fh_.emplace(wafer, module);
        module_to_wafers_fh_.emplace(module, wafer);
        // fill number of cells for a given wafer type
        number_cells_in_wafers_.emplace(wafer_type, fhTopology().dddConstants().numberCellsHexagon(wafer));
        break;
      }
      default:
        edm::LogWarning("HGCalTriggerGeometry")
            << "Unsupported subdetector number (" << subdet << ") in L1TModulesMapping file\n";
        break;
    }
  }
  if (!l1tModulesMappingStream.eof())
    edm::LogWarning("HGCalTriggerGeometry") << "Error reading L1TModulesMapping '" << wafer << " " << module << "' \n";
  l1tModulesMappingStream.close();
  // read trigger cell mapping file
  std::ifstream l1tCellsMappingStream(l1tCellsMapping_.fullPath());
  if (!l1tCellsMappingStream.is_open()) {
    throw cms::Exception("MissingDataFile") << "Cannot open HGCalTriggerGeometry L1TCellsMapping file\n";
  }
  short waferType = 0;
  short cell = 0;
  short triggerCell = 0;
  for (; l1tCellsMappingStream >> waferType >> cell >> triggerCell;) {
    // fill cell <-> trigger cell mappings
    cells_to_trigger_cells_.emplace(std::make_pair((waferType ? 1 : -1), cell), triggerCell);
    trigger_cells_to_cells_.emplace(std::make_pair((waferType ? 1 : -1), triggerCell), cell);
    // fill number of cells for a given wafer type
    auto itr_insert = number_trigger_cells_in_wafers_.emplace((waferType ? 1 : -1), 0);
    if (triggerCell + 1 > itr_insert.first->second)
      itr_insert.first->second = triggerCell + 1;
  }
  if (!l1tCellsMappingStream.eof())
    edm::LogWarning("HGCalTriggerGeometry")
        << "Error reading L1TCellsMapping'" << waferType << " " << cell << " " << triggerCell << "' \n";
  l1tCellsMappingStream.close();
}

void HGCalTriggerGeometryHexImp2::fillNeighborMaps() {
  // Fill trigger neighbor map
  std::ifstream l1tCellNeighborsMappingStream(l1tCellNeighborsMapping_.fullPath());
  if (!l1tCellNeighborsMappingStream.is_open()) {
    throw cms::Exception("MissingDataFile") << "Cannot open HGCalTriggerGeometry L1TCellNeighborsMapping file\n";
  }
  for (std::array<char, 512> buffer; l1tCellNeighborsMappingStream.getline(&buffer[0], 512);) {
    std::string line(&buffer[0]);
    // Extract keys consisting of the wafer configuration
    // and of the trigger cell id
    // Match patterns (X,Y)
    // where X is a set of 7 bits
    // and Y is a number with less than 4 digits
    std::regex key_regex("\\(\\s*[01]{7}\\s*,\\s*\\d{1,3}\\s*\\)");
    std::vector<std::string> key_tokens{std::sregex_token_iterator(line.begin(), line.end(), key_regex), {}};
    if (key_tokens.size() != 1) {
      throw cms::Exception("BadGeometry") << "Syntax error in the L1TCellNeighborsMapping:\n"
                                          << "  Cannot find the trigger cell key in line:\n"
                                          << "  '" << &buffer[0] << "'\n";
    }
    std::regex digits_regex("([01]{7})|(\\d{1,3})");
    std::vector<std::string> type_tc{
        std::sregex_token_iterator(key_tokens[0].begin(), key_tokens[0].end(), digits_regex), {}};
    // get cell id and wafer configuration
    int trigger_cell = std::stoi(type_tc[1]);
    std::vector<int> wafer_types;
    wafer_types.reserve(type_tc[0].size());
    // Convert waferType coarse=0, fine=1 to coarse=-1, fine=1
    for (const char c : type_tc[0])
      wafer_types.emplace_back((std::stoi(std::string(&c)) ? 1 : -1));
    unsigned map_key = packTriggerCell(trigger_cell, wafer_types);
    // Extract neighbors
    // Match patterns (X,Y)
    // where X is a number with less than 4 digits
    // and Y is one single digit (the neighbor wafer, between 0 and 6)
    std::regex neighbors_regex("\\(\\s*\\d{1,3}\\s*,\\s*\\d\\s*\\)");
    std::vector<std::string> neighbors_tokens{std::sregex_token_iterator(line.begin(), line.end(), neighbors_regex),
                                              {}};
    if (neighbors_tokens.empty()) {
      throw cms::Exception("BadGeometry") << "Syntax error in the L1TCellNeighborsMapping:\n"
                                          << "  Cannot find any neighbor in line:\n"
                                          << "  '" << &buffer[0] << "'\n";
    }
    auto itr_insert = trigger_cell_neighbors_.emplace(map_key, std::set<std::pair<short, short>>());
    for (const auto& neighbor : neighbors_tokens) {
      std::vector<std::string> pair_neighbor{std::sregex_token_iterator(neighbor.begin(), neighbor.end(), digits_regex),
                                             {}};
      short neighbor_wafer(std::stoi(pair_neighbor[1]));
      short neighbor_cell(std::stoi(pair_neighbor[0]));
      itr_insert.first->second.emplace(neighbor_wafer, neighbor_cell);
    }
  }
  if (!l1tCellNeighborsMappingStream.eof())
    edm::LogWarning("HGCalTriggerGeometry") << "Error reading L1TCellNeighborsMapping'\n";
  l1tCellNeighborsMappingStream.close();

  // Fill wafer neighbor map
  std::ifstream l1tWaferNeighborsMappingStream(l1tWaferNeighborsMapping_.fullPath());
  if (!l1tWaferNeighborsMappingStream.is_open()) {
    throw cms::Exception("MissingDataFile") << "Cannot open HGCalTriggerGeometry L1TWaferNeighborsMapping file\n";
  }
  for (std::array<char, 512> buffer; l1tWaferNeighborsMappingStream.getline(&buffer[0], 512);) {
    std::string line(&buffer[0]);
    // split line using spaces as delimiter
    std::regex delimiter("\\s+");
    std::vector<std::string> tokens{std::sregex_token_iterator(line.begin(), line.end(), delimiter, -1), {}};
    if (tokens.size() != 8) {
      throw cms::Exception("BadGeometry")
          << "Syntax error in the L1TWaferNeighborsMapping in line:\n"
          << "  '" << &buffer[0] << "'\n"
          << "  A line should be composed of 8 integers separated by spaces:\n"
          << "  subdet waferid neighbor1 neighbor2 neighbor3 neighbor4 neighbor5 neighbor6\n";
    }
    short subdet(std::stoi(tokens[0]));
    short wafer(std::stoi(tokens[1]));

    std::unordered_map<short, std::vector<short>>* wafer_neighbors;
    switch (subdet) {
      case ForwardSubdetector::HGCEE:
        wafer_neighbors = &wafer_neighbors_ee_;
        break;
      case ForwardSubdetector::HGCHEF:
        wafer_neighbors = &wafer_neighbors_fh_;
        break;
      default:
        throw cms::Exception("BadGeometry") << "Unknown subdet " << subdet << " in L1TWaferNeighborsMapping:\n"
                                            << "  '" << &buffer[0] << "'\n";
    };
    auto wafer_itr = wafer_neighbors->emplace(wafer, std::vector<short>());
    for (auto neighbor_itr = tokens.cbegin() + 2; neighbor_itr != tokens.cend(); ++neighbor_itr) {
      wafer_itr.first->second.emplace_back(std::stoi(*neighbor_itr));
    }
  }
}

void HGCalTriggerGeometryHexImp2::fillInvalidTriggerCells() {
  unsigned n_layers_ee = eeTopology().dddConstants().layers(true);
  for (unsigned layer = 1; layer <= n_layers_ee; layer++) {
    for (const auto& wafer_module : wafer_to_module_ee_) {
      unsigned wafer = wafer_module.first;
      int wafer_type = detIdWaferType(ForwardSubdetector::HGCEE, wafer);
      // loop on the trigger cells in each wafer
      for (int trigger_cell = 0; trigger_cell < number_trigger_cells_in_wafers_.at(wafer_type); trigger_cell++) {
        HGCalDetId trigger_cell_id_neg(ForwardSubdetector::HGCEE, -1, layer, wafer_type, wafer, trigger_cell);
        HGCalDetId trigger_cell_id_pos(ForwardSubdetector::HGCEE, 1, layer, wafer_type, wafer, trigger_cell);
        if (!validTriggerCellFromCells(trigger_cell_id_neg))
          invalid_triggercells_.emplace(trigger_cell_id_neg.rawId());
        if (!validTriggerCellFromCells(trigger_cell_id_pos))
          invalid_triggercells_.emplace(trigger_cell_id_pos.rawId());
      }
    }
  }
  unsigned n_layers_fh = fhTopology().dddConstants().layers(true);
  for (unsigned layer = 1; layer <= n_layers_fh; layer++) {
    for (const auto& wafer_module : wafer_to_module_fh_) {
      unsigned wafer = wafer_module.first;
      int wafer_type = detIdWaferType(ForwardSubdetector::HGCHEF, wafer);
      // loop on the trigger cells in each wafer
      for (int trigger_cell = 0; trigger_cell < number_trigger_cells_in_wafers_.at(wafer_type); trigger_cell++) {
        HGCalDetId trigger_cell_id_neg(ForwardSubdetector::HGCHEF, -1, layer, wafer_type, wafer, trigger_cell);
        HGCalDetId trigger_cell_id_pos(ForwardSubdetector::HGCHEF, 1, layer, wafer_type, wafer, trigger_cell);
        if (!validTriggerCellFromCells(trigger_cell_id_neg))
          invalid_triggercells_.emplace(trigger_cell_id_neg.rawId());
        if (!validTriggerCellFromCells(trigger_cell_id_pos))
          invalid_triggercells_.emplace(trigger_cell_id_pos.rawId());
      }
    }
  }
}

unsigned HGCalTriggerGeometryHexImp2::packTriggerCell(unsigned trigger_cell,
                                                      const std::vector<int>& wafer_types) const {
  unsigned packed_value = trigger_cell;
  for (unsigned i = 0; i < wafer_types.size(); i++) {
    // trigger cell id on 8 bits
    // wafer configuration bits: 0=coarse, 1=fine
    if (wafer_types.at(i) == 1)
      packed_value += (0x1 << (8 + i));
  }
  return packed_value;
}

int HGCalTriggerGeometryHexImp2::detIdWaferType(unsigned subdet, short wafer) const {
  int wafer_type = 0;
  switch (subdet) {
    // HGCalDDDConstants::waferTypeT() returns 2=coarse, 1=fine
    // HGCalDetId::waferType() returns -1=coarse, 1=fine
    // Convert to HGCalDetId waferType
    case ForwardSubdetector::HGCEE:
      wafer_type = (eeTopology().dddConstants().waferTypeT(wafer) == 2 ? -1 : 1);
      break;
    case ForwardSubdetector::HGCHEF:
      wafer_type = (fhTopology().dddConstants().waferTypeT(wafer) == 2 ? -1 : 1);
      break;
    default:
      break;
  };
  return wafer_type;
}

bool HGCalTriggerGeometryHexImp2::validTriggerCell(const unsigned trigger_cell_id) const {
  return invalid_triggercells_.find(trigger_cell_id) == invalid_triggercells_.end();
}

bool HGCalTriggerGeometryHexImp2::disconnectedModule(const unsigned module_id) const { return false; }

unsigned HGCalTriggerGeometryHexImp2::lastTriggerLayer() const { return eeTopology().dddConstants().layers(true); }

unsigned HGCalTriggerGeometryHexImp2::triggerLayer(const unsigned id) const { return HGCalDetId(id).layer(); }

bool HGCalTriggerGeometryHexImp2::validTriggerCellFromCells(const unsigned trigger_cell_id) const {
  // Check the validity of a trigger cell with the
  // validity of the cells. One valid cell in the
  // trigger cell is enough to make the trigger cell
  // valid.
  HGCalDetId trigger_cell_det_id(trigger_cell_id);
  unsigned subdet = trigger_cell_det_id.subdetId();
  const geom_set cells = getCellsFromTriggerCell(trigger_cell_id);
  bool is_valid = false;
  for (const auto cell_id : cells) {
    is_valid |= validCellId(subdet, cell_id);
    if (is_valid)
      break;
  }
  return is_valid;
}

bool HGCalTriggerGeometryHexImp2::validCellId(unsigned subdet, unsigned cell_id) const {
  bool is_valid = false;
  switch (subdet) {
    case ForwardSubdetector::HGCEE:
      is_valid = eeTopology().valid(cell_id);
      break;
    case ForwardSubdetector::HGCHEF:
      is_valid = fhTopology().valid(cell_id);
      break;
    default:
      is_valid = false;
      break;
  }
  return is_valid;
}

DEFINE_EDM_PLUGIN(HGCalTriggerGeometryFactory, HGCalTriggerGeometryHexImp2, "HGCalTriggerGeometryHexImp2");
