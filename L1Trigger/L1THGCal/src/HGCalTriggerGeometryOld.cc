#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryOld.h"

using namespace HGCalTriggerGeometry;

namespace {
  std::unique_ptr<const TriggerCell> null_tc;
  std::unique_ptr<const Module>      null_mod;
}

HGCalTriggerGeometryOld::
HGCalTriggerGeometryOld(const edm::ParameterSet& conf) : 
    HGCalTriggerGeometryBase(conf){
}

void HGCalTriggerGeometryOld::reset() {
  geom_map().swap(cells_to_trigger_cells_);
  geom_map().swap(trigger_cells_to_modules_);
  module_map().swap(modules_);
  trigger_cell_map().swap(trigger_cells_);
}

unsigned 
HGCalTriggerGeometryOld::
getTriggerCellFromCell( const unsigned cell_det_id ) const {
  auto found_tc = cells_to_trigger_cells_.find(cell_det_id);
  if( found_tc == cells_to_trigger_cells_.end() ) {
    return 0;
  }
  return trigger_cells_.find(found_tc->second)->second->triggerCellId();
}

unsigned 
HGCalTriggerGeometryOld::
getModuleFromCell( const unsigned cell_det_id ) const {
  auto found_tc = cells_to_trigger_cells_.find(cell_det_id);
  if( found_tc == cells_to_trigger_cells_.end() ) {
    return 0;
  }
  auto found_mod = trigger_cells_to_modules_.find(found_tc->second);
  if( found_mod == trigger_cells_to_modules_.end() ) {
    return 0;
  }
  return modules_.find(found_mod->second)->second->moduleId();
}

unsigned
HGCalTriggerGeometryOld::
getModuleFromTriggerCell( const unsigned trigger_cell_det_id ) const {
  auto found_mod = trigger_cells_to_modules_.find(trigger_cell_det_id);
  if( found_mod == trigger_cells_to_modules_.end() ) {
    return 0;
  }
  return modules_.find(found_mod->second)->second->moduleId();
}

HGCalTriggerGeometryBase::geom_set 
HGCalTriggerGeometryOld::
getCellsFromTriggerCell( const unsigned trigger_cell_det_id ) const {
  return trigger_cells_.find(trigger_cell_det_id)->second->components();
}

HGCalTriggerGeometryBase::geom_set 
HGCalTriggerGeometryOld::
getCellsFromModule( const unsigned module_det_id ) const {
  const auto& triggercell_cells = modules_.find(module_det_id)->second->triggerCellComponents();
  HGCalTriggerGeometryBase::geom_set cells;
  for(const auto& tc_c : triggercell_cells) {
    cells.emplace(tc_c.second);
  }
  return cells; 
}

HGCalTriggerGeometryBase::geom_set 
HGCalTriggerGeometryOld::
getTriggerCellsFromModule( const unsigned module_det_id ) const {
  return modules_.find(module_det_id)->second->components(); 
}

GlobalPoint 
HGCalTriggerGeometryOld::
getTriggerCellPosition(const unsigned trigger_cell_det_id) const {
   return trigger_cells_.find(trigger_cell_det_id)->second->position(); 
}

GlobalPoint 
HGCalTriggerGeometryOld::
getModulePosition(const unsigned module_det_id) const {
  return modules_.find(module_det_id)->second->position();
}


