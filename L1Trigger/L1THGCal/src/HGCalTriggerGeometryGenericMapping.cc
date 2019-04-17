#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryGenericMapping.h"

using namespace HGCalTriggerGeometry;

namespace {
  std::unique_ptr<const TriggerCell> null_tc;
  std::unique_ptr<const Module> null_mod;
}  // namespace

HGCalTriggerGeometryGenericMapping::HGCalTriggerGeometryGenericMapping(const edm::ParameterSet& conf)
    : HGCalTriggerGeometryBase(conf) {}

void HGCalTriggerGeometryGenericMapping::reset() {
  geom_map().swap(cells_to_trigger_cells_);
  geom_map().swap(trigger_cells_to_modules_);
  module_map().swap(modules_);
  trigger_cell_map().swap(trigger_cells_);
}

unsigned HGCalTriggerGeometryGenericMapping::getTriggerCellFromCell(const unsigned cell_det_id) const {
  auto found_tc = cells_to_trigger_cells_.find(cell_det_id);
  if (found_tc == cells_to_trigger_cells_.end()) {
    return 0;
  }
  return trigger_cells_.find(found_tc->second)->second->triggerCellId();
}

unsigned HGCalTriggerGeometryGenericMapping::getModuleFromCell(const unsigned cell_det_id) const {
  auto found_tc = cells_to_trigger_cells_.find(cell_det_id);
  if (found_tc == cells_to_trigger_cells_.end()) {
    return 0;
  }
  auto found_mod = trigger_cells_to_modules_.find(found_tc->second);
  if (found_mod == trigger_cells_to_modules_.end()) {
    return 0;
  }
  return modules_.find(found_mod->second)->second->moduleId();
}

unsigned HGCalTriggerGeometryGenericMapping::getModuleFromTriggerCell(const unsigned trigger_cell_det_id) const {
  auto found_mod = trigger_cells_to_modules_.find(trigger_cell_det_id);
  if (found_mod == trigger_cells_to_modules_.end()) {
    return 0;
  }
  return modules_.find(found_mod->second)->second->moduleId();
}

HGCalTriggerGeometryBase::geom_set HGCalTriggerGeometryGenericMapping::getCellsFromTriggerCell(
    const unsigned trigger_cell_det_id) const {
  return trigger_cells_.find(trigger_cell_det_id)->second->components();
}

HGCalTriggerGeometryBase::geom_set HGCalTriggerGeometryGenericMapping::getCellsFromModule(
    const unsigned module_det_id) const {
  const auto& triggercell_cells = modules_.find(module_det_id)->second->triggerCellComponents();
  HGCalTriggerGeometryBase::geom_set cells;
  for (const auto& tc_c : triggercell_cells) {
    cells.emplace(tc_c.second);
  }
  return cells;
}

HGCalTriggerGeometryBase::geom_ordered_set HGCalTriggerGeometryGenericMapping::getOrderedCellsFromModule(
    const unsigned module_det_id) const {
  const auto& triggercell_cells = modules_.find(module_det_id)->second->triggerCellComponents();
  HGCalTriggerGeometryBase::geom_ordered_set cells;
  for (const auto& tc_c : triggercell_cells) {
    cells.emplace(tc_c.second);
  }
  return cells;
}

HGCalTriggerGeometryBase::geom_set HGCalTriggerGeometryGenericMapping::getTriggerCellsFromModule(
    const unsigned module_det_id) const {
  return modules_.find(module_det_id)->second->components();
}

HGCalTriggerGeometryBase::geom_ordered_set HGCalTriggerGeometryGenericMapping::getOrderedTriggerCellsFromModule(
    const unsigned module_det_id) const {
  // Build set from unordered_set. Maybe a more efficient to do it
  HGCalTriggerGeometryBase::geom_ordered_set trigger_cells;
  for (const auto& tc : modules_.find(module_det_id)->second->components()) {
    trigger_cells.emplace(tc);
  }
  return trigger_cells;
}

HGCalTriggerGeometryBase::geom_set HGCalTriggerGeometryGenericMapping::getNeighborsFromTriggerCell(
    const unsigned trigger_cell_id) const {
  // empty neighbors
  return geom_set();
}

GlobalPoint HGCalTriggerGeometryGenericMapping::getTriggerCellPosition(const unsigned trigger_cell_det_id) const {
  return trigger_cells_.find(trigger_cell_det_id)->second->position();
}

GlobalPoint HGCalTriggerGeometryGenericMapping::getModulePosition(const unsigned module_det_id) const {
  return modules_.find(module_det_id)->second->position();
}

bool HGCalTriggerGeometryGenericMapping::validTriggerCell(const unsigned trigger_cell_det_id) const {
  return (trigger_cells_.find(trigger_cell_det_id) != trigger_cells_.end());
}

bool HGCalTriggerGeometryGenericMapping::disconnectedModule(const unsigned module_id) const { return false; }

unsigned HGCalTriggerGeometryGenericMapping::triggerLayer(const unsigned id) const { return HGCalDetId(id).layer(); }
