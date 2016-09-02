#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerLightweightGeometryBase.h"
#include "DataFormats/ForwardDetId/interface/HGCTriggerHexDetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

#include <vector>
#include <iostream>
#include <fstream>


class HGCalTriggerGeometryHexImp2 : public HGCalTriggerLightweightGeometryBase
{
    public:
        HGCalTriggerGeometryHexImp2(const edm::ParameterSet& conf);

        virtual void initialize(const es_info& ) override final;

        virtual const unsigned getTriggerCellFromCell( const unsigned cell_det_id ) const override final;
        virtual const unsigned getModuleFromCell( const unsigned cell_det_id ) const override final;
        virtual const unsigned getModuleFromTriggerCell( const unsigned trigger_cell_det_id ) const override final;

        virtual const geom_set getCellsFromTriggerCell( const unsigned cell_det_id ) const override final;
        virtual const geom_set getCellsFromModule( const unsigned cell_det_id ) const override final;
        virtual const geom_set getTriggerCellsFromModule( const unsigned trigger_cell_det_id ) const override final;

        virtual const geom_set& getValidTriggerCellIds() const override final;
        virtual const geom_set& getValidModuleIds() const override final;

    private:
        edm::FileInPath l1tCellsMapping_;
        edm::FileInPath l1tModulesMapping_;

        geom_set validTriggerCellIds_;
        geom_set validModuleIds_;

        std::unordered_map<short, short> wafer_to_module_ee_;
        std::unordered_map<short, short> wafer_to_module_fh_;
        std::unordered_multimap<short, short> module_to_wafers_ee_;
        std::unordered_multimap<short, short> module_to_wafers_fh_;

        std::map<std::pair<short,short>, short> cells_to_trigger_cells_; // FIXME: something else than map<pair,short>?
        std::multimap<std::pair<short,short>, short> trigger_cells_to_cells_;// FIXME: something else than map<pair,short>?
        std::unordered_map<short, short> number_trigger_cells_in_wafers_; // the map key is the wafer type
        std::unordered_map<short, short> number_cells_in_wafers_; // the map key is the wafer type

        void fillMaps(const es_info&);
};


HGCalTriggerGeometryHexImp2::
HGCalTriggerGeometryHexImp2(const edm::ParameterSet& conf):
    HGCalTriggerLightweightGeometryBase(conf),
    l1tCellsMapping_(conf.getParameter<edm::FileInPath>("L1TCellsMapping")),
    l1tModulesMapping_(conf.getParameter<edm::FileInPath>("L1TModulesMapping"))
{
}


void
HGCalTriggerGeometryHexImp2::
initialize(const es_info& esInfo)
{
    edm::LogWarning("HGCalTriggerGeometry") << "WARNING: This HGCal trigger geometry is incomplete.\n"\
                                            << "WARNING: There is no positions implemented.\n"\
                                            << "WARNING: There is no neighbor information.\n";

    fillMaps(esInfo);

}

const unsigned 
HGCalTriggerGeometryHexImp2::
getTriggerCellFromCell( const unsigned cell_det_id ) const
{
    int wafer_type = HGCTriggerHexDetId::waferType(cell_det_id);
    unsigned cell = HGCTriggerHexDetId::cell(cell_det_id);
    // FIXME: better way to do this cell->TC mapping?
    unsigned trigger_cell = cells_to_trigger_cells_.at(std::make_pair(wafer_type,cell));
    // This is possible because HGCTriggerHexDetId has the same structure as HGCalDetId
    // Maybe enforce this by deriving HGCTriggerHexDetId from HGCalDetId
    unsigned trigger_cell_det_id = cell_det_id;
    HGCTriggerHexDetId::setCell(trigger_cell_det_id, trigger_cell);
    return trigger_cell_det_id;
}

const unsigned 
HGCalTriggerGeometryHexImp2::
getModuleFromCell( const unsigned cell_det_id ) const
{
    unsigned wafer = HGCTriggerHexDetId::wafer(cell_det_id);
    unsigned subdet = HGCTriggerHexDetId::subdet(cell_det_id);
    unsigned module = 0;
    switch(subdet)
    {
        case ForwardSubdetector::HGCEE:
            module = wafer_to_module_ee_.at(wafer);
            break;
        case ForwardSubdetector::HGCHEF:
            module = wafer_to_module_fh_.at(wafer);
            break;
        default:
            edm::LogError("HGCalTriggerGeometry") << "Unknown wafer->module mapping for subdet "<<subdet<<"\n";
            return 0;
    };
    // This is possible because HGCTriggerHexDetId has the same structure as HGCalDetId
    // Maybe enforce this by deriving HGCTriggerHexDetId from HGCalDetId
    unsigned module_id = cell_det_id;
    HGCTriggerHexDetId::setWafer(module_id, module);
    HGCTriggerHexDetId::setCell(module_id, HGCTriggerHexDetId::UndefinedCell());
    return module_id;
}

const unsigned 
HGCalTriggerGeometryHexImp2::
getModuleFromTriggerCell( const unsigned trigger_cell_det_id ) const
{
    unsigned wafer = HGCTriggerHexDetId::wafer(trigger_cell_det_id);
    unsigned subdet = HGCTriggerHexDetId::subdet(trigger_cell_det_id);
    unsigned module = 0;
    switch(subdet)
    {
        case ForwardSubdetector::HGCEE:
            module = wafer_to_module_ee_.at(wafer);
            break;
        case ForwardSubdetector::HGCHEF:
            module = wafer_to_module_fh_.at(wafer);
            break;
        default:
            edm::LogError("HGCalTriggerGeometry") << "Unknown wafer->module mapping for subdet "<<subdet<<"\n";
            return 0;
    };
    unsigned module_id = trigger_cell_det_id;
    HGCTriggerHexDetId::setWafer(module_id, module);
    HGCTriggerHexDetId::setCell(module_id, HGCTriggerHexDetId::UndefinedCell());
    return module_id;
}

const HGCalTriggerLightweightGeometryBase::geom_set 
HGCalTriggerGeometryHexImp2::
getCellsFromTriggerCell( const unsigned trigger_cell_det_id ) const
{
    int wafer_type = HGCTriggerHexDetId::waferType(trigger_cell_det_id);
    unsigned trigger_cell = HGCTriggerHexDetId::cell(trigger_cell_det_id);
    // FIXME: better way to do this TC->cell mapping?
    const auto& cell_range = trigger_cells_to_cells_.equal_range(std::make_pair(wafer_type,trigger_cell));
    geom_set cell_det_ids;
    for(auto tc_c_itr=cell_range.first; tc_c_itr!=cell_range.second; tc_c_itr++)
    {
        unsigned cell_det_id = trigger_cell_det_id;
        HGCTriggerHexDetId::setCell(cell_det_id, tc_c_itr->second);
        cell_det_ids.emplace(cell_det_id);
    }
    return cell_det_ids;
}

const HGCalTriggerLightweightGeometryBase::geom_set 
HGCalTriggerGeometryHexImp2::
getCellsFromModule( const unsigned module_det_id ) const
{
    unsigned module = HGCTriggerHexDetId::wafer(module_det_id);
    int wafer_type = HGCTriggerHexDetId::waferType(module_det_id);
    unsigned subdet = HGCTriggerHexDetId::subdet(module_det_id);
    std::pair<std::unordered_multimap<short, short>::const_iterator,
        std::unordered_multimap<short, short>::const_iterator> wafer_itrs;
    switch(subdet)
    {
        case ForwardSubdetector::HGCEE:
            wafer_itrs = module_to_wafers_ee_.equal_range(module);
            break;
        case ForwardSubdetector::HGCHEF:
            wafer_itrs = module_to_wafers_fh_.equal_range(module);
            break;
        default:
            edm::LogError("HGCalTriggerGeometry") << "Unknown module->wafers mapping for subdet "<<subdet<<"\n";
            return geom_set();
    };
    geom_set cell_det_ids;
    for(auto wafer_itr=wafer_itrs.first; wafer_itr!=wafer_itrs.second; wafer_itr++)
    {
        // loop on the cells in each wafer
        for(int cell=0; cell<number_cells_in_wafers_.at(wafer_type); cell++)
        {
            unsigned cell_det_id = module_det_id;
            HGCTriggerHexDetId::setWafer(cell_det_id, wafer_itr->second);
            HGCTriggerHexDetId::setCell(cell_det_id, cell);
            cell_det_ids.emplace(cell_det_id);
        }
    }
    return cell_det_ids;
}

const HGCalTriggerLightweightGeometryBase::geom_set 
HGCalTriggerGeometryHexImp2::
getTriggerCellsFromModule( const unsigned module_det_id ) const
{
    unsigned module = HGCTriggerHexDetId::wafer(module_det_id);
    unsigned wafer_type = HGCTriggerHexDetId::waferType(module_det_id);
    unsigned subdet = HGCTriggerHexDetId::subdet(module_det_id);
    std::pair<std::unordered_multimap<short, short>::const_iterator,
        std::unordered_multimap<short, short>::const_iterator> wafer_itrs;
    switch(subdet)
    {
        case ForwardSubdetector::HGCEE:
            wafer_itrs = module_to_wafers_ee_.equal_range(module);
            break;
        case ForwardSubdetector::HGCHEF:
            wafer_itrs = module_to_wafers_fh_.equal_range(module);
            break;
        default:
            edm::LogError("HGCalTriggerGeometry") << "Unknown module->wafers mapping for subdet "<<subdet<<"\n";
            return geom_set();
    };
    geom_set trigger_cell_det_ids;
    // loop on the wafers included in the module
    for(auto wafer_itr=wafer_itrs.first; wafer_itr!=wafer_itrs.second; wafer_itr++)
    {
        // loop on the trigger cells in each wafer
        for(int trigger_cell=0; trigger_cell<number_trigger_cells_in_wafers_.at(wafer_type); trigger_cell++)
        {
            unsigned trigger_cell_det_id = module_det_id;
            HGCTriggerHexDetId::setWafer(trigger_cell_det_id, wafer_itr->second);
            HGCTriggerHexDetId::setCell(trigger_cell_det_id, trigger_cell);
            trigger_cell_det_ids.emplace(trigger_cell_det_id);
        }
    }
    return trigger_cell_det_ids;
}

// FIXME: empty valid IDs
const HGCalTriggerLightweightGeometryBase::geom_set& 
HGCalTriggerGeometryHexImp2::
getValidTriggerCellIds() const
{
    return validTriggerCellIds_;
}

// FIXME: empty valid IDs
const HGCalTriggerLightweightGeometryBase::geom_set& 
HGCalTriggerGeometryHexImp2::
getValidModuleIds() const
{
    return validModuleIds_;
}



void 
HGCalTriggerGeometryHexImp2::
fillMaps(const es_info& esInfo)
{
    //
    // read module mapping file
    std::ifstream l1tModulesMappingStream(l1tModulesMapping_.fullPath());
    if(!l1tModulesMappingStream.is_open()) edm::LogError("HGCalTriggerGeometry") << "Cannot open L1TModulesMapping file\n";
    short subdet  = 0;
    short wafer   = 0;
    short module  = 0;
    for(; l1tModulesMappingStream>>subdet>>wafer>>module; )
    {
        switch(subdet)
        {
            case ForwardSubdetector::HGCEE:
            {
                // fill module <-> wafers mappings
                wafer_to_module_ee_.emplace(wafer,module);
                module_to_wafers_ee_.emplace(module, wafer);
                // fill number of cells for a given wafer type
                // translate wafer type 1/2 to 1/-1 
                int wafer_type = esInfo.topo_ee->dddConstants().waferTypeT(wafer)==1?1:-1;
                number_cells_in_wafers_.emplace(wafer_type, esInfo.topo_ee->dddConstants().numberCellsHexagon(wafer));
                break;
            }
            case ForwardSubdetector::HGCHEF:
            {
                // fill module <-> wafers mappings
                wafer_to_module_fh_.emplace(wafer,module);
                module_to_wafers_fh_.emplace(module, wafer);
                // fill number of cells for a given wafer type
                // translate wafer type 1/2 to 1/-1
                int wafer_type = esInfo.topo_fh->dddConstants().waferTypeT(wafer)==1?1:-1;
                number_cells_in_wafers_.emplace(wafer_type, esInfo.topo_fh->dddConstants().numberCellsHexagon(wafer));
                break;
            }
            default:
                edm::LogWarning("HGCalTriggerGeometry") << "Unsupported subdetector number ("<<subdet<<") in L1TModulesMapping file\n";
                break;
        }
    }
    if(!l1tModulesMappingStream.eof()) edm::LogWarning("HGCalTriggerGeometry") << "Error reading L1TModulesMapping '"<<wafer<<" "<<module<<"' \n";
    l1tModulesMappingStream.close();
    // read trigger cell mapping file
    std::ifstream l1tCellsMappingStream(l1tCellsMapping_.fullPath());
    if(!l1tCellsMappingStream.is_open()) edm::LogError("HGCalTriggerGeometry") << "Cannot open L1TCellsMapping file\n";
    short waferType   = 0;
    short cell        = 0;
    short triggerCell = 0;
    for(; l1tCellsMappingStream>>waferType>>cell>>triggerCell; )
    {
        // fill cell <-> trigger cell mappings
        cells_to_trigger_cells_.emplace(std::make_pair((waferType?1:-1),cell), triggerCell);
        trigger_cells_to_cells_.emplace(std::make_pair((waferType?1:-1),triggerCell), cell);
        // fill number of cells for a given wafer type
        auto itr_insert = number_trigger_cells_in_wafers_.emplace((waferType?1:-1), 0);
        if(triggerCell+1 > itr_insert.first->second) itr_insert.first->second = triggerCell+1;
    }
    if(!l1tCellsMappingStream.eof()) edm::LogWarning("HGCalTriggerGeometry") << "Error reading L1TCellsMapping'"<<waferType<<" "<<cell<<" "<<triggerCell<<"' \n";
    l1tCellsMappingStream.close();
}




DEFINE_EDM_PLUGIN(HGCalTriggerLightweightGeometryFactory, 
        HGCalTriggerGeometryHexImp2,
        "HGCalTriggerGeometryHexImp2");
