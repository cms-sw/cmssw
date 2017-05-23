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



// Pure virtual trigger geometry class
// Provides the interface to access trigger cell and module mappings
class HGCalTriggerGeometryBase 
{ 
    public:  
        struct es_info 
        {
            edm::ESHandle<HGCalGeometry> geom_ee, geom_fh, geom_bh;
            edm::ESHandle<HGCalTopology> topo_ee, topo_fh, topo_bh;
        };

        typedef std::unordered_map<unsigned,unsigned> geom_map;
        typedef std::unordered_set<unsigned> geom_set;
        typedef std::set<unsigned> geom_ordered_set;

        HGCalTriggerGeometryBase(const edm::ParameterSet& conf);
        virtual ~HGCalTriggerGeometryBase() {}

        const std::string& name() const { return name_; } 

        const std::string& eeSDName() const { return ee_sd_name_; } 
        const std::string& fhSDName() const { return fh_sd_name_; } 
        const std::string& bhSDName() const { return bh_sd_name_; } 
        const es_info& cellInfo() const {return es_info_;}

        // non-const access to the geometry class
        virtual void initialize( const es_info& ) = 0;
        virtual void reset();

        // const access to the geometry class
        virtual unsigned getTriggerCellFromCell( const unsigned cell_det_id ) const = 0;
        virtual unsigned getModuleFromCell( const unsigned cell_det_id ) const = 0;
        virtual unsigned getModuleFromTriggerCell( const unsigned trigger_cell_det_id ) const = 0;

        virtual geom_set getCellsFromTriggerCell( const unsigned cell_det_id ) const = 0;
        virtual geom_set getCellsFromModule( const unsigned cell_det_id ) const = 0;
        virtual geom_set getTriggerCellsFromModule( const unsigned trigger_cell_det_id ) const = 0;

        virtual geom_ordered_set getOrderedCellsFromModule( const unsigned cell_det_id ) const = 0;
        virtual geom_ordered_set getOrderedTriggerCellsFromModule( const unsigned trigger_cell_det_id ) const = 0;

        virtual geom_set getNeighborsFromTriggerCell( const unsigned trigger_cell_det_id ) const = 0;

        virtual GlobalPoint getTriggerCellPosition(const unsigned trigger_cell_det_id) const = 0;
        virtual GlobalPoint getModulePosition(const unsigned module_det_id) const = 0;

        virtual bool validTriggerCell( const unsigned trigger_cell_id) const = 0;

    protected:
        void setCellInfo(const es_info& es) {es_info_=es;}


    private:
        const std::string name_;
        const std::string ee_sd_name_;
        const std::string fh_sd_name_;
        const std::string bh_sd_name_;  

        es_info es_info_;

};

#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory< HGCalTriggerGeometryBase* (const edm::ParameterSet&) > HGCalTriggerGeometryFactory;


#endif
