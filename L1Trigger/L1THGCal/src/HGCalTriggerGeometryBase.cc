#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

using namespace HGCalTriggerGeometry;

namespace {
  std::unique_ptr<const TriggerCell> null_tc;
  std::unique_ptr<const Module>      null_mod;
}

HGCalTriggerGeometryBase::
HGCalTriggerGeometryBase(const edm::ParameterSet& conf) : 
  name_(conf.getParameter<std::string>("TriggerGeometryName")),
  ee_sd_name_(conf.getParameter<std::string>("eeSDName")),
  fh_sd_name_(conf.getParameter<std::string>("fhSDName")),
  bh_sd_name_(conf.getParameter<std::string>("bhSDName")) {
}

void HGCalTriggerGeometryBase::reset() {
  geom_map().swap(cells_to_trigger_cells_);
  geom_map().swap(trigger_cells_to_modules_);
  module_map().swap(modules_);
  trigger_cell_map().swap(trigger_cells_);
}

const std::unique_ptr<const TriggerCell>& 
HGCalTriggerGeometryBase::
getTriggerCellFromCell( const unsigned cell_det_id ) const {
  auto found_tc = cells_to_trigger_cells_.find(cell_det_id);
  if( found_tc == cells_to_trigger_cells_.end() ) {
    return null_tc;
  }
  return trigger_cells_.find(found_tc->second)->second;
}

const std::unique_ptr<const Module>& 
HGCalTriggerGeometryBase::
getModuleFromCell( const unsigned cell_det_id ) const {
  auto found_tc = cells_to_trigger_cells_.find(cell_det_id);
  if( found_tc == cells_to_trigger_cells_.end() ) {
    return null_mod;
  }
  auto found_mod = trigger_cells_to_modules_.find(found_tc->second);
  if( found_mod == trigger_cells_to_modules_.end() ) {
    return null_mod;
  }
  return modules_.find(found_mod->second)->second;
}

const std::unique_ptr<const Module>&
HGCalTriggerGeometryBase::
getModuleFromTriggerCell( const unsigned trigger_cell_det_id ) const {
  auto found_mod = trigger_cells_to_modules_.find(trigger_cell_det_id);
  if( found_mod == trigger_cells_to_modules_.end() ) {
    return null_mod;
  }
  return modules_.find(found_mod->second)->second;
}

EDM_REGISTER_PLUGINFACTORY(HGCalTriggerGeometryFactory,
			   "HGCalTriggerGeometryFactory");
