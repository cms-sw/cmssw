#ifndef __L1Trigger_L1THGCal_HGCalTriggerLightweightGeometryBase_h__
#define __L1Trigger_L1THGCal_HGCalTriggerLightweightGeometryBase_h__

#include <iostream>
#include <unordered_set>
#include <unordered_map>

//#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/CaloTopology/interface/HGCalTopology.h"



// Pure virtual trigger geometry class
// Provides the interface to access trigger cell and module mappings
class HGCalTriggerLightweightGeometryBase 
{ 
    public:  
        struct es_info 
        {
            edm::ESHandle<HGCalGeometry> geom_ee, geom_fh, geom_bh;
            edm::ESHandle<HGCalTopology> topo_ee, topo_fh, topo_bh;
        };

        typedef std::unordered_map<unsigned,unsigned> geom_map;
        typedef std::unordered_set<unsigned> geom_set;

        HGCalTriggerLightweightGeometryBase(const edm::ParameterSet& conf);
        virtual ~HGCalTriggerLightweightGeometryBase() {}

        const std::string& name() const { return name_; } 

        const std::string& eeSDName() const { return ee_sd_name_; } 
        const std::string& fhSDName() const { return fh_sd_name_; } 
        const std::string& bhSDName() const { return bh_sd_name_; } 

        // non-const access to the geometry class
        virtual void initialize( const es_info& ) = 0;
        void reset();

        // const access to the geometry class
        virtual unsigned getTriggerCellFromCell( const unsigned cell_det_id ) const = 0;
        virtual unsigned getModuleFromCell( const unsigned cell_det_id ) const = 0;
        virtual unsigned getModuleFromTriggerCell( const unsigned trigger_cell_det_id ) const = 0;

        virtual geom_set getCellsFromTriggerCell( const unsigned cell_det_id ) const = 0;
        virtual geom_set getCellsFromModule( const unsigned cell_det_id ) const = 0;
        virtual geom_set getTriggerCellsFromModule( const unsigned trigger_cell_det_id ) const = 0;

        virtual GlobalPoint getTriggerCellPosition(const unsigned trigger_cell_det_id) const = 0;
        virtual GlobalPoint getModulePosition(const unsigned module_det_id) const = 0;

        virtual const geom_set& getValidTriggerCellIds() const = 0;
        virtual const geom_set& getValidModuleIds() const = 0;


    private:
        const std::string name_;
        const std::string ee_sd_name_;
        const std::string fh_sd_name_;
        const std::string bh_sd_name_;  


};

#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory< HGCalTriggerLightweightGeometryBase* (const edm::ParameterSet&) > HGCalTriggerLightweightGeometryFactory;


#endif
