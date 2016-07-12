#ifndef __L1Trigger_L1THGCal_HGCalTriggerGeometryBase_h__
#define __L1Trigger_L1THGCal_HGCalTriggerGeometryBase_h__

#include <iostream>
#include <unordered_set>
#include <unordered_map>

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/CaloTopology/interface/HGCalTopology.h"

/*******
 *
 * class: HGCalTriggerGeometryBase
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

class HGCalTriggerGeometryBase;

namespace HGCalTriggerGeometry {
  class TriggerCell {
  public:
    typedef std::unordered_set<unsigned> list_type;

    TriggerCell(unsigned tc_id, unsigned mod_id, const GlobalPoint& pos,
                const list_type& neighbs, const list_type& comps) :
      trigger_cell_id_(tc_id),
      module_id_(mod_id),
      position_(pos),
      neighbours_(neighbs),
      components_(comps)
      {}
    ~TriggerCell() {}
   
    
    unsigned triggerCellId() const { return trigger_cell_id_; }
    unsigned moduleId()      const { return module_id_; }
    
    bool containsCell(const unsigned cell) const {
      return ( components_.find(cell) != components_.end() );
    }

    const GlobalPoint& position() const { return position_; }

    const std::unordered_set<unsigned>& neighbours() const { return neighbours_; }
    const std::unordered_set<unsigned>& components() const { return components_; }

  private:
    unsigned trigger_cell_id_; // the ID of this trigger cell
    unsigned module_id_; // module this TC belongs to
    GlobalPoint position_;
    list_type neighbours_; // neighbouring trigger cells
    list_type components_; // contained HGC cells
  };
  
  class Module {
  public:
    typedef std::unordered_set<unsigned> list_type;
    typedef std::unordered_multimap<unsigned,unsigned> tc_map_type;
    
    Module(unsigned mod_id, const GlobalPoint& pos,
           const list_type& neighbs, const list_type& comps,
           const tc_map_type& tc_comps):
      module_id_(mod_id),
      position_(pos),
      neighbours_(neighbs),
      components_(comps),
      tc_components_(tc_comps)  
      {}
    ~Module() {}
    
    unsigned moduleId()      const { return module_id_; }

    bool containsTriggerCell(const unsigned trig_cell) const {
      return ( components_.find(trig_cell) != components_.end() );
    }

    bool containsCell(const unsigned cell) const {
      for( const auto& value : tc_components_ ) {
        if( value.second == cell ) return true;
      }
      return false;
    }

    const GlobalPoint& position() const { return position_; }

    const list_type& neighbours() const { return neighbours_; }
    const list_type& components() const { return components_; }

    const tc_map_type& triggerCellComponents() const { return tc_components_; }

  private:    
    unsigned module_id_; // module this TC belongs to
    GlobalPoint position_;
    list_type neighbours_; // neighbouring Modules
    list_type components_; // contained HGC trigger cells
    tc_map_type tc_components_; // cells contained by trigger cells
  };
}  

class HGCalTriggerGeometryBase { 
 public:  
  struct es_info {
    edm::ESHandle<HGCalGeometry> geom_ee, geom_fh, geom_bh;
    edm::ESHandle<HGCalTopology> topo_ee, topo_fh, topo_bh;
  };

  typedef std::unordered_map<unsigned,unsigned> geom_map;
  typedef std::unordered_map<unsigned,std::unique_ptr<const HGCalTriggerGeometry::Module> > module_map;
  typedef std::unordered_map<unsigned,std::unique_ptr<const HGCalTriggerGeometry::TriggerCell> > trigger_cell_map;

  HGCalTriggerGeometryBase(const edm::ParameterSet& conf);
  virtual ~HGCalTriggerGeometryBase() {}

  const std::string& name() const { return name_; } 

  const std::string& eeSDName() const { return ee_sd_name_; } 
  const std::string& fhSDName() const { return fh_sd_name_; } 
  const std::string& bhSDName() const { return bh_sd_name_; } 

  // non-const access to the geometry class
  virtual void initialize( const es_info& ) = 0;
  void reset();
  
  // const access to the geometry class  
  // all of the get*From* functions return nullptr if the thing you
  // ask for doesn't exist
  const std::unique_ptr<const HGCalTriggerGeometry::TriggerCell>& getTriggerCellFromCell( const unsigned cell_det_id ) const;
  const std::unique_ptr<const HGCalTriggerGeometry::Module>& getModuleFromCell( const unsigned cell_det_id ) const;
  const std::unique_ptr<const HGCalTriggerGeometry::Module>& getModuleFromTriggerCell( const unsigned trigger_cell_det_id ) const;

  const geom_map& cellsToTriggerCellsMap() const { return cells_to_trigger_cells_; }
  const geom_map& triggerCellsToModulesMap() const { return trigger_cells_to_modules_; }
  
  const module_map& modules() const { return modules_; }
  const trigger_cell_map& triggerCells() const { return trigger_cells_; }

 protected:
  geom_map cells_to_trigger_cells_;
  geom_map trigger_cells_to_modules_;
  
  module_map modules_;
  trigger_cell_map trigger_cells_;
  
 private:
  const std::string name_;
  const std::string ee_sd_name_;
  const std::string fh_sd_name_;
  const std::string bh_sd_name_;  
};

#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory< HGCalTriggerGeometryBase* (const edm::ParameterSet&) > HGCalTriggerGeometryFactory;


#endif
