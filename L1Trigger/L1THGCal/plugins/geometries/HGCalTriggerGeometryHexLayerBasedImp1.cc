#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <regex>


class HGCalTriggerGeometryHexLayerBasedImp1 : public HGCalTriggerGeometryBase
{
    public:
        HGCalTriggerGeometryHexLayerBasedImp1(const edm::ParameterSet& conf);

        virtual void initialize(const es_info& ) override final;
        virtual void reset() override final;

        virtual unsigned getTriggerCellFromCell( const unsigned ) const override final;
        virtual unsigned getModuleFromCell( const unsigned ) const override final;
        virtual unsigned getModuleFromTriggerCell( const unsigned ) const override final;

        virtual geom_set getCellsFromTriggerCell( const unsigned ) const override final;
        virtual geom_set getCellsFromModule( const unsigned ) const override final;
        virtual geom_set getTriggerCellsFromModule( const unsigned ) const override final;

        virtual geom_ordered_set getOrderedCellsFromModule( const unsigned ) const override final;
        virtual geom_ordered_set getOrderedTriggerCellsFromModule( const unsigned ) const override final;

        virtual geom_set getNeighborsFromTriggerCell( const unsigned ) const override final;

        virtual GlobalPoint getTriggerCellPosition(const unsigned ) const override final;
        virtual GlobalPoint getModulePosition(const unsigned ) const override final;

        virtual bool validTriggerCell( const unsigned ) const override final;

    private:
        edm::FileInPath l1tCellsMapping_;
        edm::FileInPath l1tModulesMapping_;

        es_info es_info_;

        // module related maps
        std::unordered_map<unsigned, unsigned> wafer_to_module_;
        std::unordered_multimap<unsigned, unsigned> module_to_wafers_;

        // trigger cell related maps
        std::unordered_map<unsigned, unsigned> cells_to_trigger_cells_;
        std::unordered_multimap<unsigned, unsigned> trigger_cells_to_cells_;
        std::unordered_map<unsigned, unsigned short> number_trigger_cells_in_wafers_; 
        std::unordered_set<unsigned> invalid_triggercells_;

        void fillMaps(const es_info&);
        void fillInvalidTriggerCells(const es_info&);
        bool validCellId(unsigned subdet, unsigned cell_id) const;
        bool validTriggerCellFromCells( const unsigned ) const;

        // returns transverse wafer type: -1=coarse, 1=fine, 0=undefined
        int detIdWaferType(unsigned subdet, short wafer) const;
        unsigned packWaferCellId(unsigned subdet, unsigned wafer, unsigned cell) const;
        void unpackWaferCellId(unsigned wafer_cell, unsigned& wafer, unsigned& cell) const;
};


HGCalTriggerGeometryHexLayerBasedImp1::
HGCalTriggerGeometryHexLayerBasedImp1(const edm::ParameterSet& conf):
    HGCalTriggerGeometryBase(conf),
    l1tCellsMapping_(conf.getParameter<edm::FileInPath>("L1TCellsMapping")),
    l1tModulesMapping_(conf.getParameter<edm::FileInPath>("L1TModulesMapping"))
{
}

void
HGCalTriggerGeometryHexLayerBasedImp1::
reset()
{
    cells_to_trigger_cells_.clear();
    trigger_cells_to_cells_.clear();
    wafer_to_module_.clear();
    module_to_wafers_.clear();
    number_trigger_cells_in_wafers_.clear();
}

void
HGCalTriggerGeometryHexLayerBasedImp1::
initialize(const es_info& esInfo)
{
    es_info_ = esInfo;
    fillMaps(esInfo);
    fillInvalidTriggerCells(esInfo);

}

unsigned 
HGCalTriggerGeometryHexLayerBasedImp1::
getTriggerCellFromCell( const unsigned cell_id ) const
{
    HGCalDetId cell_det_id(cell_id);
    unsigned subdet = cell_det_id.subdetId();
    unsigned wafer = cell_det_id.wafer();
    unsigned cell = cell_det_id.cell();
    auto trigger_cell_itr = cells_to_trigger_cells_.find(packWaferCellId(subdet, wafer, cell));
    if(trigger_cell_itr==cells_to_trigger_cells_.end())
    {
        throw cms::Exception("BadGeometry")
            << "HGCalTriggerGeometry: HGCal  cell " << cell << " in wafer "<<wafer<<" is not mapped to any trigger cell. The trigger cell mapping should be modified.\n";
    }
    unsigned trigger_cell = 0;
    unsigned wafer_trigger_cell = 0;
    unpackWaferCellId(trigger_cell_itr->second, wafer_trigger_cell, trigger_cell);
    return HGCalDetId((ForwardSubdetector)cell_det_id.subdetId(), cell_det_id.zside(), cell_det_id.layer(), 1, wafer_trigger_cell, trigger_cell).rawId();
}

unsigned 
HGCalTriggerGeometryHexLayerBasedImp1::
getModuleFromCell( const unsigned cell_id ) const
{
    return getModuleFromTriggerCell(getTriggerCellFromCell(cell_id));
}

unsigned 
HGCalTriggerGeometryHexLayerBasedImp1::
getModuleFromTriggerCell( const unsigned trigger_cell_id ) const
{
    HGCalDetId trigger_cell_det_id(trigger_cell_id);
    auto module_itr = wafer_to_module_.find(trigger_cell_det_id.wafer());
    if(module_itr==wafer_to_module_.end())
    {
        throw cms::Exception("BadGeometry")
            << "HGCalTriggerGeometry: Wafer " << trigger_cell_det_id.wafer() << " is not mapped to any trigger module. The module mapping should be modified. See https://twiki.cern.ch/twiki/bin/viewauth/CMS/HGCALTriggerPrimitivesSimulation#Trigger_geometry for details.\n";
    }
    unsigned module = module_itr->second;
    return HGCalDetId((ForwardSubdetector)trigger_cell_det_id.subdetId(), trigger_cell_det_id.zside(), trigger_cell_det_id.layer(), trigger_cell_det_id.waferType(), module, HGCalDetId::kHGCalCellMask).rawId();
}

HGCalTriggerGeometryBase::geom_set 
HGCalTriggerGeometryHexLayerBasedImp1::
getCellsFromTriggerCell( const unsigned trigger_cell_id ) const
{
    HGCalDetId trigger_cell_det_id(trigger_cell_id);
    unsigned subdet = trigger_cell_det_id.subdetId();
    unsigned trigger_wafer = trigger_cell_det_id.wafer();
    unsigned trigger_cell = trigger_cell_det_id.cell();
    const auto& cell_range = trigger_cells_to_cells_.equal_range(packWaferCellId(subdet, trigger_wafer, trigger_cell));
    geom_set cell_det_ids;
    for(auto tc_c_itr=cell_range.first; tc_c_itr!=cell_range.second; tc_c_itr++)
    {
        unsigned wafer = 0;
        unsigned cell = 0;
        unpackWaferCellId(tc_c_itr->second, wafer, cell);
        unsigned wafer_type = detIdWaferType(subdet, wafer);
        cell_det_ids.emplace(HGCalDetId((ForwardSubdetector)trigger_cell_det_id.subdetId(), trigger_cell_det_id.zside(), trigger_cell_det_id.layer(), wafer_type, wafer, cell).rawId());
    }
    return cell_det_ids;
}

HGCalTriggerGeometryBase::geom_set 
HGCalTriggerGeometryHexLayerBasedImp1::
getCellsFromModule( const unsigned module_id ) const
{
    geom_set cell_det_ids;
    geom_set trigger_cells = getTriggerCellsFromModule(module_id);
    for(auto trigger_cell_id : trigger_cells)
    {
        geom_set cells = getCellsFromTriggerCell(trigger_cell_id);
        cell_det_ids.insert(cells.begin(), cells.end());
    }
    return cell_det_ids;
}

HGCalTriggerGeometryBase::geom_ordered_set 
HGCalTriggerGeometryHexLayerBasedImp1::
getOrderedCellsFromModule( const unsigned module_id ) const
{
    geom_ordered_set cell_det_ids;
    geom_ordered_set trigger_cells = getOrderedTriggerCellsFromModule(module_id);
    for(auto trigger_cell_id : trigger_cells)
    {
        geom_set cells = getCellsFromTriggerCell(trigger_cell_id);
        cell_det_ids.insert(cells.begin(), cells.end());
    }
    return cell_det_ids;
}

HGCalTriggerGeometryBase::geom_set 
HGCalTriggerGeometryHexLayerBasedImp1::
getTriggerCellsFromModule( const unsigned module_id ) const
{
    HGCalDetId module_det_id(module_id);
    unsigned module = module_det_id.wafer();
    auto wafer_itrs = module_to_wafers_.equal_range(module);
    geom_set trigger_cell_det_ids;
    // loop on the wafers included in the module
    for(auto wafer_itr=wafer_itrs.first; wafer_itr!=wafer_itrs.second; wafer_itr++)
    {
        unsigned wafer = wafer_itr->second;
        // loop on the trigger cells in each wafer
        for(int trigger_cell=0; trigger_cell<number_trigger_cells_in_wafers_.at(wafer); trigger_cell++)
        {
            HGCalDetId trigger_cell_id((ForwardSubdetector)module_det_id.subdetId(), module_det_id.zside(), module_det_id.layer(), 1, wafer, trigger_cell);
            if(validTriggerCell(trigger_cell_id)) trigger_cell_det_ids.emplace(trigger_cell_id.rawId());
        }
    }
    return trigger_cell_det_ids;
}

HGCalTriggerGeometryBase::geom_ordered_set 
HGCalTriggerGeometryHexLayerBasedImp1::
getOrderedTriggerCellsFromModule( const unsigned module_id ) const
{
    HGCalDetId module_det_id(module_id);
    unsigned module = module_det_id.wafer();
    auto wafer_itrs = module_to_wafers_.equal_range(module);
    geom_ordered_set trigger_cell_det_ids;
    // loop on the wafers included in the module
    for(auto wafer_itr=wafer_itrs.first; wafer_itr!=wafer_itrs.second; wafer_itr++)
    {
        unsigned wafer = wafer_itr->second;
        // loop on the trigger cells in each wafer
        for(int trigger_cell=0; trigger_cell<number_trigger_cells_in_wafers_.at(wafer); trigger_cell++)
        {
            HGCalDetId trigger_cell_id((ForwardSubdetector)module_det_id.subdetId(), module_det_id.zside(), module_det_id.layer(), 1, wafer, trigger_cell);
            if(validTriggerCell(trigger_cell_id)) trigger_cell_det_ids.emplace(trigger_cell_id.rawId());
        }
    }
    return trigger_cell_det_ids;
}



HGCalTriggerGeometryBase::geom_set
HGCalTriggerGeometryHexLayerBasedImp1::
getNeighborsFromTriggerCell( const unsigned trigger_cell_id ) const
{
    geom_set neighbor_detids;
    return neighbor_detids;
}


GlobalPoint 
HGCalTriggerGeometryHexLayerBasedImp1::
getTriggerCellPosition(const unsigned trigger_cell_det_id) const
{
    // Position: barycenter of the trigger cell.
    Basic3DVector<float> triggerCellVector(0.,0.,0.);
    const auto cell_ids = getCellsFromTriggerCell(trigger_cell_det_id);
    for(const auto& cell : cell_ids)
    {
        HGCalDetId cellDetId(cell);
        triggerCellVector += (cellDetId.subdetId()==ForwardSubdetector::HGCEE ? es_info_.geom_ee->getPosition(cellDetId) :  es_info_.geom_fh->getPosition(cellDetId)).basicVector();
    }
    return GlobalPoint( triggerCellVector/cell_ids.size() );

}

GlobalPoint 
HGCalTriggerGeometryHexLayerBasedImp1::
getModulePosition(const unsigned module_det_id) const
{
    // Position: barycenter of the module.
    Basic3DVector<float> moduleVector(0.,0.,0.);
    const auto cell_ids = getCellsFromModule(module_det_id);
    for(const auto& cell : cell_ids)
    {
        HGCalDetId cellDetId(cell);
        moduleVector += (cellDetId.subdetId()==ForwardSubdetector::HGCEE ? es_info_.geom_ee->getPosition(cellDetId) :  es_info_.geom_fh->getPosition(cellDetId)).basicVector();
    }
    return GlobalPoint( moduleVector/cell_ids.size() );
}


void 
HGCalTriggerGeometryHexLayerBasedImp1::
fillMaps(const es_info& esInfo)
{
    //
    // read module mapping file
    std::ifstream l1tModulesMappingStream(l1tModulesMapping_.fullPath());
    if(!l1tModulesMappingStream.is_open()) edm::LogError("HGCalTriggerGeometry") << "Cannot open L1TModulesMapping file\n";
    short trigger_wafer   = 0;
    short module  = 0;
    for(; l1tModulesMappingStream>>trigger_wafer>>module; )
    {
        wafer_to_module_.emplace(trigger_wafer,module);
        module_to_wafers_.emplace(module, trigger_wafer);
    }
    if(!l1tModulesMappingStream.eof()) edm::LogWarning("HGCalTriggerGeometry") << "Error reading L1TModulesMapping '"<<trigger_wafer<<" "<<module<<"' \n";
    l1tModulesMappingStream.close();
    // read trigger cell mapping file
    std::ifstream l1tCellsMappingStream(l1tCellsMapping_.fullPath());
    if(!l1tCellsMappingStream.is_open()) edm::LogError("HGCalTriggerGeometry") << "Cannot open L1TCellsMapping file\n";
    short subdet = 0;
    short wafer = 0;
    short cell = 0;
    trigger_wafer = 0;
    short trigger_cell = 0;
    for(; l1tCellsMappingStream>>subdet>>wafer>>cell>>trigger_wafer>>trigger_cell; )
    { 
        unsigned cell_key = packWaferCellId(subdet,wafer,cell);
        unsigned trigger_cell_key = packWaferCellId(subdet,trigger_wafer,trigger_cell);
        // fill cell <-> trigger cell mappings
        cells_to_trigger_cells_.emplace(cell_key, trigger_cell_key);
        trigger_cells_to_cells_.emplace(trigger_cell_key, cell_key);
        // fill number of trigger cells in wafers
        auto itr_insert = number_trigger_cells_in_wafers_.emplace(trigger_wafer, 0);
        if(trigger_cell+1 > itr_insert.first->second) itr_insert.first->second = trigger_cell+1;
    }
    if(!l1tCellsMappingStream.eof()) edm::LogWarning("HGCalTriggerGeometry") << "Error reading L1TCellsMapping '"<<subdet<<" "<<wafer<<" "<<cell<<" "<<trigger_wafer<<" "<<trigger_cell<<"' \n";
    l1tCellsMappingStream.close();
}



void 
HGCalTriggerGeometryHexLayerBasedImp1::
fillInvalidTriggerCells(const es_info& esInfo)
{
}

unsigned 
HGCalTriggerGeometryHexLayerBasedImp1::
packWaferCellId(unsigned subdet, unsigned wafer, unsigned cell) const
{
    unsigned packed_value = 0;
    packed_value |= ((cell & HGCalDetId::kHGCalCellMask) << HGCalDetId::kHGCalCellOffset);
    packed_value |= ((wafer & HGCalDetId::kHGCalWaferMask) << HGCalDetId::kHGCalWaferOffset);
    packed_value |= ((subdet & 0x7) << (HGCalDetId::kHGCalWaferOffset+3));
    return packed_value;
}

void
HGCalTriggerGeometryHexLayerBasedImp1::
unpackWaferCellId(unsigned wafer_cell, unsigned& wafer, unsigned& cell) const
{
    cell =  wafer_cell & HGCalDetId::kHGCalCellMask;
    wafer = (wafer_cell>>HGCalDetId::kHGCalWaferOffset) & HGCalDetId::kHGCalWaferMask;
}



bool 
HGCalTriggerGeometryHexLayerBasedImp1::
validTriggerCell(const unsigned trigger_cell_id) const
{
    return invalid_triggercells_.find(trigger_cell_id)==invalid_triggercells_.end();
}

bool 
HGCalTriggerGeometryHexLayerBasedImp1::
validTriggerCellFromCells(const unsigned trigger_cell_id) const
{
    // Check the validity of a trigger cell with the
    // validity of the cells. One valid cell in the 
    // trigger cell is enough to make the trigger cell
    // valid.
    HGCalDetId trigger_cell_det_id(trigger_cell_id);
    unsigned subdet = trigger_cell_det_id.subdetId();
    const geom_set cells = getCellsFromTriggerCell(trigger_cell_id);
    bool is_valid = false;
    for(const auto cell_id : cells)
    {
        is_valid |= validCellId(subdet, cell_id);
        if(is_valid) break;
    }
    return is_valid;
}

bool
HGCalTriggerGeometryHexLayerBasedImp1::
validCellId(unsigned subdet, unsigned cell_id) const
{
    bool is_valid = false;
    switch(subdet)
    {
        case ForwardSubdetector::HGCEE:
            is_valid = es_info_.topo_ee->valid(cell_id);
            break;
        case ForwardSubdetector::HGCHEF:
            is_valid = es_info_.topo_fh->valid(cell_id);
            break;
        default:
            is_valid = false;
            break;
    } 
    return is_valid;
}

int 
HGCalTriggerGeometryHexLayerBasedImp1::
detIdWaferType(unsigned subdet, short wafer) const
{
    int wafer_type = 0;
    switch(subdet)
    {
        // HGCalDDDConstants::waferTypeT() returns 2=coarse, 1=fine
        // HGCalDetId::waferType() returns -1=coarse, 1=fine
        // Convert to HGCalDetId waferType
        case ForwardSubdetector::HGCEE:
            wafer_type = (es_info_.topo_ee->dddConstants().waferTypeT(wafer)==2?-1:1);
            break;
        case ForwardSubdetector::HGCHEF:
            wafer_type = (es_info_.topo_fh->dddConstants().waferTypeT(wafer)==2?-1:1);
            break;
        default:
            break;
    };
    return wafer_type;
}



DEFINE_EDM_PLUGIN(HGCalTriggerGeometryFactory, 
        HGCalTriggerGeometryHexLayerBasedImp1,
        "HGCalTriggerGeometryHexLayerBasedImp1");
