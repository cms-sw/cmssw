#ifndef __L1Trigger_L1THGCal_HGCalTriggerGeometryGenericMapping_h__
#define __L1Trigger_L1THGCal_HGCalTriggerGeometryGenericMapping_h__

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

/*******
 *
 * class: HGCalTriggerGeometryGenericMapping
 * author: L.Gray (FNAL)
 * date: 26 July, 2015
 *
 * This class is the base class for all HGCal trigger geometry definitions.
 * Typically this is just a ganging of cells and nearest-neighbour relationships.
 * 
 * Classes for TriggerCells and Modules are defined. They are containers for
 * maps relating modules to modules and trigger cells to trigger cells and
 * raw HGC cells.
 * 
 * It is up to the user of the class to define what these mappings are through the
 * initialize() function. This function takes the full HGC geometry as input and
 * creates all the necessary maps for navigating the trigger system. All of the 
 * containers for parts of the map are available through protected members of the base
 * class.
 * 
 * All unsigned ints used here are DetIds, or will become them once we sort out the
 * full nature of the trigger detids.
 *******/

class HGCalTriggerGeometryGenericMapping;

namespace HGCalTriggerGeometry {
  class TriggerCell {
  public:
    typedef std::unordered_set<unsigned> list_type;

    TriggerCell(
        unsigned tc_id, unsigned mod_id, const GlobalPoint& pos, const list_type& neighbs, const list_type& comps)
        : trigger_cell_id_(tc_id), module_id_(mod_id), position_(pos), neighbours_(neighbs), components_(comps) {}
    ~TriggerCell() {}

    unsigned triggerCellId() const { return trigger_cell_id_; }
    unsigned moduleId() const { return module_id_; }

    bool containsCell(const unsigned cell) const { return (components_.find(cell) != components_.end()); }

    const GlobalPoint& position() const { return position_; }

    const std::unordered_set<unsigned>& neighbours() const { return neighbours_; }
    const std::unordered_set<unsigned>& components() const { return components_; }

  private:
    unsigned trigger_cell_id_;  // the ID of this trigger cell
    unsigned module_id_;        // module this TC belongs to
    GlobalPoint position_;
    list_type neighbours_;  // neighbouring trigger cells
    list_type components_;  // contained HGC cells
  };

  class Module {
  public:
    typedef std::unordered_set<unsigned> list_type;
    typedef std::unordered_multimap<unsigned, unsigned> tc_map_type;

    Module(unsigned mod_id,
           const GlobalPoint& pos,
           const list_type& neighbs,
           const list_type& comps,
           const tc_map_type& tc_comps)
        : module_id_(mod_id), position_(pos), neighbours_(neighbs), components_(comps), tc_components_(tc_comps) {}
    ~Module() {}

    unsigned moduleId() const { return module_id_; }

    bool containsTriggerCell(const unsigned trig_cell) const {
      return (components_.find(trig_cell) != components_.end());
    }

    bool containsCell(const unsigned cell) const {
      for (const auto& value : tc_components_) {
        if (value.second == cell)
          return true;
      }
      return false;
    }

    const GlobalPoint& position() const { return position_; }

    const list_type& neighbours() const { return neighbours_; }
    const list_type& components() const { return components_; }

    const tc_map_type& triggerCellComponents() const { return tc_components_; }

  private:
    unsigned module_id_;  // module this TC belongs to
    GlobalPoint position_;
    list_type neighbours_;       // neighbouring Modules
    list_type components_;       // contained HGC trigger cells
    tc_map_type tc_components_;  // cells contained by trigger cells
  };
}  // namespace HGCalTriggerGeometry

class HGCalTriggerGeometryGenericMapping : public HGCalTriggerGeometryBase {
public:
  typedef std::unordered_map<unsigned, std::unique_ptr<const HGCalTriggerGeometry::Module> > module_map;
  typedef std::unordered_map<unsigned, std::unique_ptr<const HGCalTriggerGeometry::TriggerCell> > trigger_cell_map;

  HGCalTriggerGeometryGenericMapping(const edm::ParameterSet& conf);
  ~HGCalTriggerGeometryGenericMapping() override {}

  // non-const access to the geometry class
  void reset() final;

  unsigned getTriggerCellFromCell(const unsigned cell_det_id) const final;
  unsigned getModuleFromCell(const unsigned cell_det_id) const final;
  unsigned getModuleFromTriggerCell(const unsigned trigger_cell_det_id) const final;

  geom_set getCellsFromTriggerCell(const unsigned cell_det_id) const final;
  geom_set getCellsFromModule(const unsigned cell_det_id) const final;
  geom_set getTriggerCellsFromModule(const unsigned trigger_cell_det_id) const final;

  geom_ordered_set getOrderedCellsFromModule(const unsigned cell_det_id) const final;
  geom_ordered_set getOrderedTriggerCellsFromModule(const unsigned trigger_cell_det_id) const final;

  geom_set getNeighborsFromTriggerCell(const unsigned trigger_cell_det_id) const final;

  GlobalPoint getTriggerCellPosition(const unsigned trigger_cell_det_id) const final;
  GlobalPoint getModulePosition(const unsigned module_det_id) const final;

  bool validTriggerCell(const unsigned trigger_cell_det_id) const final;
  bool disconnectedModule(const unsigned module_id) const final;
  unsigned triggerLayer(const unsigned id) const final;

protected:
  geom_map cells_to_trigger_cells_;
  geom_map trigger_cells_to_modules_;

  module_map modules_;
  trigger_cell_map trigger_cells_;
};

#endif
