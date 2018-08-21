#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <regex>


class HGCalTriggerGeometryHexLayerBasedImp1 : public HGCalTriggerGeometryBase
{
    public:
        HGCalTriggerGeometryHexLayerBasedImp1(const edm::ParameterSet& conf);

        void initialize(const edm::ESHandle<CaloGeometry>& ) final;
        void reset() final;

        unsigned getTriggerCellFromCell( const unsigned ) const final;
        unsigned getModuleFromCell( const unsigned ) const final;
        unsigned getModuleFromTriggerCell( const unsigned ) const final;

        geom_set getCellsFromTriggerCell( const unsigned ) const final;
        geom_set getCellsFromModule( const unsigned ) const final;
        geom_set getTriggerCellsFromModule( const unsigned ) const final;

        geom_ordered_set getOrderedCellsFromModule( const unsigned ) const final;
        geom_ordered_set getOrderedTriggerCellsFromModule( const unsigned ) const final;

        geom_set getNeighborsFromTriggerCell( const unsigned ) const final;

        GlobalPoint getTriggerCellPosition(const unsigned ) const final;
        GlobalPoint getModulePosition(const unsigned ) const final;

        bool validTriggerCell( const unsigned ) const final;
        bool disconnectedModule(const unsigned) const final;
        unsigned triggerLayer(const unsigned) const final;

    private:
        edm::FileInPath l1tCellsMapping_;
        edm::FileInPath l1tCellsBHMapping_;
        edm::FileInPath l1tModulesMapping_;
        edm::FileInPath l1tCellNeighborsMapping_;
        edm::FileInPath l1tCellNeighborsBHMapping_;

        // module related maps
        std::unordered_map<unsigned, unsigned> wafer_to_module_;
        std::unordered_multimap<unsigned, unsigned> module_to_wafers_;

        // trigger cell related maps
        std::unordered_map<unsigned, unsigned> cells_to_trigger_cells_;
        std::unordered_multimap<unsigned, unsigned> trigger_cells_to_cells_;
        std::unordered_map<unsigned, unsigned> cells_to_trigger_cells_bh_;
        std::unordered_multimap<unsigned, unsigned> trigger_cells_to_cells_bh_;
        std::unordered_map<unsigned, unsigned short> number_trigger_cells_in_wafers_; 
        std::unordered_map<unsigned, unsigned short> number_trigger_cells_in_wafers_bh_; 
        std::unordered_set<unsigned> invalid_triggercells_;

        // neighbor related maps
        // trigger cell neighbors:
        // - The key includes the module and trigger cell id
        // - The value is a set of (module_id, trigger_cell_id)
        std::unordered_map<int, std::set<std::pair<short,short>>> trigger_cell_neighbors_;
        std::unordered_map<int, std::set<std::pair<short,short>>> trigger_cell_neighbors_bh_;

        // Disconnected modules and layers
        std::unordered_set<unsigned> disconnected_modules_;
        std::unordered_set<unsigned> disconnected_layers_;
        std::vector<unsigned> trigger_layers_;

        // layer offsets 
        unsigned fhOffset_;
        unsigned bhOffset_;
        unsigned totalLayers_;

        void fillMaps();
        void fillNeighborMaps(const edm::FileInPath&,  std::unordered_map<int, std::set<std::pair<short,short>>>&);
        void fillInvalidTriggerCells();
        unsigned packTriggerCell(unsigned, unsigned) const;
        bool validCellId(unsigned subdet, unsigned cell_id) const;
        bool validTriggerCellFromCells( const unsigned ) const;

        // returns transverse wafer type: -1=coarse, 1=fine, 0=undefined
        int detIdWaferType(unsigned subdet, short wafer) const;
        unsigned packWaferCellId(unsigned subdet, unsigned wafer, unsigned cell) const;
        unsigned packIetaIphi(unsigned ieta, unsigned iphi) const;
        void unpackWaferCellId(unsigned wafer_cell, unsigned& wafer, unsigned& cell) const;
        void unpackIetaIphi(unsigned ieta_iphi, unsigned& ieta, unsigned& iphi) const;

        unsigned layerWithOffset(unsigned) const;
};


HGCalTriggerGeometryHexLayerBasedImp1::
HGCalTriggerGeometryHexLayerBasedImp1(const edm::ParameterSet& conf):
    HGCalTriggerGeometryBase(conf),
    l1tCellsMapping_(conf.getParameter<edm::FileInPath>("L1TCellsMapping")),
    l1tCellsBHMapping_(conf.getParameter<edm::FileInPath>("L1TCellsBHMapping")),
    l1tModulesMapping_(conf.getParameter<edm::FileInPath>("L1TModulesMapping")),
    l1tCellNeighborsMapping_(conf.getParameter<edm::FileInPath>("L1TCellNeighborsMapping")),
    l1tCellNeighborsBHMapping_(conf.getParameter<edm::FileInPath>("L1TCellNeighborsBHMapping"))
{
    std::vector<unsigned> tmp_vector = conf.getParameter<std::vector<unsigned>>("DisconnectedModules");
    std::move(tmp_vector.begin(), tmp_vector.end(), std::inserter(disconnected_modules_, disconnected_modules_.end()));
    tmp_vector = conf.getParameter<std::vector<unsigned>>("DisconnectedLayers");
    std::move(tmp_vector.begin(), tmp_vector.end(), std::inserter(disconnected_layers_, disconnected_layers_.end()));
}

void
HGCalTriggerGeometryHexLayerBasedImp1::
reset()
{
    cells_to_trigger_cells_.clear();
    trigger_cells_to_cells_.clear();
    cells_to_trigger_cells_bh_.clear();
    trigger_cells_to_cells_bh_.clear();
    wafer_to_module_.clear();
    module_to_wafers_.clear();
    number_trigger_cells_in_wafers_.clear();
    number_trigger_cells_in_wafers_bh_.clear();
    trigger_cell_neighbors_.clear();
    trigger_cell_neighbors_bh_.clear();
}

void
HGCalTriggerGeometryHexLayerBasedImp1::
initialize(const edm::ESHandle<CaloGeometry>& calo_geometry)
{
    setCaloGeometry(calo_geometry);
    fhOffset_ = eeTopology().dddConstants().layers(true);
    bhOffset_ = fhOffset_ + fhTopology().dddConstants().layers(true);
    totalLayers_ =  bhOffset_ + bhTopology().dddConstants()->getMaxDepth(1);
    trigger_layers_.resize(totalLayers_+1);
    unsigned trigger_layer = 0;
    for(unsigned layer=0; layer<trigger_layers_.size(); layer++)
    {
        if(disconnected_layers_.find(layer)==disconnected_layers_.end())
        {
            // Increase trigger layer number if the layer is not disconnected
            trigger_layers_[layer] = trigger_layer;
            trigger_layer++;
        }
        else
        {
            trigger_layers_[layer] = 0;
        }
    }
    fillMaps();
    fillNeighborMaps(l1tCellNeighborsMapping_, trigger_cell_neighbors_);
    fillNeighborMaps(l1tCellNeighborsBHMapping_, trigger_cell_neighbors_bh_);
    fillInvalidTriggerCells();

}

unsigned 
HGCalTriggerGeometryHexLayerBasedImp1::
getTriggerCellFromCell( const unsigned cell_id ) const
{
    unsigned subdet = 0;
    int zside = 0;
    unsigned layer = 0;
    unsigned wafer_trigger_cell = 0;
    unsigned trigger_cell = 0;
    // BH
    if(DetId(cell_id).det() == DetId::Hcal)
    {
        HcalDetId cell_det_id(cell_id);
        if(cell_det_id.subdetId()!=HcalEndcap) return 0;
        unsigned ieta = cell_det_id.ietaAbs();
        unsigned iphi = cell_det_id.iphi();
        layer = cell_det_id.depth();
        subdet = ForwardSubdetector::HGCHEB;
        zside = cell_det_id.zside();
        auto trigger_cell_itr = cells_to_trigger_cells_bh_.find(packIetaIphi(ieta, iphi));
        if(trigger_cell_itr==cells_to_trigger_cells_bh_.end())
        {
            throw cms::Exception("BadGeometry")
                << "HGCalTriggerGeometry: Hcal  cell ieta=" << ieta << ", iphi="<<iphi<<" is not mapped to any trigger cell. The trigger cell mapping should be modified.\n";
        }
        trigger_cell = 0;
        wafer_trigger_cell = 0;
        unpackWaferCellId(trigger_cell_itr->second, wafer_trigger_cell, trigger_cell);
    }
    // EE or FH
    else if(DetId(cell_id).det() == DetId::Forward)
    {
        HGCalDetId cell_det_id(cell_id);
        subdet = cell_det_id.subdetId();
        layer = cell_det_id.layer();
        zside = cell_det_id.zside();
        unsigned wafer = cell_det_id.wafer();
        unsigned cell = cell_det_id.cell();
        auto trigger_cell_itr = cells_to_trigger_cells_.find(packWaferCellId(subdet, wafer, cell));
        if(trigger_cell_itr==cells_to_trigger_cells_.end())
        {
            throw cms::Exception("BadGeometry")
                << "HGCalTriggerGeometry: HGCal  cell " << cell << " in wafer "<<wafer<<" is not mapped to any trigger cell. The trigger cell mapping should be modified.\n";
        }
        trigger_cell = 0;
        wafer_trigger_cell = 0;
        unpackWaferCellId(trigger_cell_itr->second, wafer_trigger_cell, trigger_cell);
    }
    return HGCalDetId((ForwardSubdetector)subdet, zside, layer, 1, wafer_trigger_cell, trigger_cell).rawId();

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
    unsigned module = 0;
    // BH
    if(trigger_cell_det_id.subdetId()==ForwardSubdetector::HGCHEB)
    {
        // For BH, the module ID is currently encoded as the wafer in HGCalDetId
        module = trigger_cell_det_id.wafer();
    }
    // EE or FH
    else
    {
        auto module_itr = wafer_to_module_.find(trigger_cell_det_id.wafer());
        if(module_itr==wafer_to_module_.end())
        {
            throw cms::Exception("BadGeometry")
                <<trigger_cell_det_id
                << "HGCalTriggerGeometry: Wafer " << trigger_cell_det_id.wafer() << " is not mapped to any trigger module. The module mapping should be modified. See https://twiki.cern.ch/twiki/bin/viewauth/CMS/HGCALTriggerPrimitivesSimulation#Trigger_geometry for details.\n";
        }
        module = module_itr->second;
    }
    return HGCalDetId((ForwardSubdetector)trigger_cell_det_id.subdetId(), trigger_cell_det_id.zside(), trigger_cell_det_id.layer(), (trigger_cell_det_id.waferType()==1 ? 1:0), module, HGCalDetId::kHGCalCellMask).rawId();
}

HGCalTriggerGeometryBase::geom_set 
HGCalTriggerGeometryHexLayerBasedImp1::
getCellsFromTriggerCell( const unsigned trigger_cell_id ) const
{
    HGCalDetId trigger_cell_det_id(trigger_cell_id);
    geom_set cell_det_ids;
    // BH
    if(trigger_cell_det_id.subdetId()==ForwardSubdetector::HGCHEB)
    {
        unsigned subdet = trigger_cell_det_id.subdetId();
        unsigned trigger_wafer = trigger_cell_det_id.wafer();
        unsigned trigger_cell = trigger_cell_det_id.cell();
        const auto& cell_range = trigger_cells_to_cells_bh_.equal_range(packWaferCellId(subdet, trigger_wafer, trigger_cell));
        for(auto tc_c_itr=cell_range.first; tc_c_itr!=cell_range.second; tc_c_itr++)
        {
            unsigned ieta = 0;
            unsigned iphi = 0;
            unpackIetaIphi(tc_c_itr->second, ieta, iphi);
            unsigned cell_det_id = HcalDetId(HcalEndcap, trigger_cell_det_id.zside()*ieta, iphi, trigger_cell_det_id.layer()).rawId();
            if(validCellId(subdet, cell_det_id)) cell_det_ids.emplace(cell_det_id);
        }
    }
    // EE or FH
    else
    {
        unsigned subdet = trigger_cell_det_id.subdetId();
        unsigned trigger_wafer = trigger_cell_det_id.wafer();
        unsigned trigger_cell = trigger_cell_det_id.cell();
        const auto& cell_range = trigger_cells_to_cells_.equal_range(packWaferCellId(subdet, trigger_wafer, trigger_cell));
        for(auto tc_c_itr=cell_range.first; tc_c_itr!=cell_range.second; tc_c_itr++)
        {
            unsigned wafer = 0;
            unsigned cell = 0;
            unpackWaferCellId(tc_c_itr->second, wafer, cell);
            unsigned wafer_type = (detIdWaferType(subdet, wafer)==1 ? 1 : 0);
            unsigned cell_det_id = HGCalDetId((ForwardSubdetector)trigger_cell_det_id.subdetId(), trigger_cell_det_id.zside(), trigger_cell_det_id.layer(), wafer_type, wafer, cell).rawId();
            if(validCellId(subdet, cell_det_id)) cell_det_ids.emplace(cell_det_id);
        }
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
    geom_set trigger_cell_det_ids;
    // BH
    if(module_det_id.subdetId()==ForwardSubdetector::HGCHEB)
    {
        unsigned module = module_det_id.wafer();
        // loop on the trigger cells in each module
        for(int trigger_cell=0; trigger_cell<number_trigger_cells_in_wafers_bh_.at(module); trigger_cell++)
        {
            HGCalDetId trigger_cell_id((ForwardSubdetector)module_det_id.subdetId(), module_det_id.zside(), module_det_id.layer(), 1, module, trigger_cell);
            if(validTriggerCell(trigger_cell_id)) trigger_cell_det_ids.emplace(trigger_cell_id.rawId());
        }
    }
    // EE or FH
    else
    {
        unsigned module = module_det_id.wafer();
        auto wafer_itrs = module_to_wafers_.equal_range(module);
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
    }
    return trigger_cell_det_ids;
}

HGCalTriggerGeometryBase::geom_ordered_set 
HGCalTriggerGeometryHexLayerBasedImp1::
getOrderedTriggerCellsFromModule( const unsigned module_id ) const
{
    HGCalDetId module_det_id(module_id);
    geom_ordered_set trigger_cell_det_ids;
    // BH
    if(module_det_id.subdetId()==ForwardSubdetector::HGCHEB)
    {
        unsigned module = module_det_id.wafer();
        // loop on the trigger cells in each module
        for(int trigger_cell=0; trigger_cell<number_trigger_cells_in_wafers_bh_.at(module); trigger_cell++)
        {
            HGCalDetId trigger_cell_id((ForwardSubdetector)module_det_id.subdetId(), module_det_id.zside(), module_det_id.layer(), 1, module, trigger_cell);
            if(validTriggerCell(trigger_cell_id)) trigger_cell_det_ids.emplace(trigger_cell_id.rawId());
        }
    }
    // EE or FH
    else
    {
        unsigned module = module_det_id.wafer();
        auto wafer_itrs = module_to_wafers_.equal_range(module);
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
    }
    return trigger_cell_det_ids;
}



HGCalTriggerGeometryBase::geom_set
HGCalTriggerGeometryHexLayerBasedImp1::
getNeighborsFromTriggerCell( const unsigned trigger_cell_id ) const
{
    HGCalDetId trigger_cell_det_id(trigger_cell_id);
    // Choose scintillator or silicon map
    const auto& neighbors_map = (trigger_cell_det_id.subdetId()==ForwardSubdetector::HGCHEB ? trigger_cell_neighbors_bh_ : trigger_cell_neighbors_);
    unsigned module = trigger_cell_det_id.wafer();
    unsigned trigger_cell = trigger_cell_det_id.cell();
    // retrieve neighbors
    unsigned trigger_cell_key = packTriggerCell(module, trigger_cell);
    geom_set neighbor_detids;
    auto neighbors_itr = neighbors_map.find(trigger_cell_key);
    if(neighbors_itr==neighbors_map.end())
    {
        throw cms::Exception("BadGeometry")
            << "HGCalTriggerGeometry: Neighbors are not defined for trigger cell " << trigger_cell << " in module "
            << module << ". The trigger cell neighbor mapping should be modified. \n";
    }
    const auto& neighbors = neighbors_itr->second;
    // create HGCalDetId of neighbors and check their validity
    neighbor_detids.reserve(neighbors.size());
    for(const auto& module_tc : neighbors)
    {
        HGCalDetId neighbor_det_id((ForwardSubdetector)trigger_cell_det_id.subdetId(), trigger_cell_det_id.zside(), trigger_cell_det_id.layer(), 1, module_tc.first, module_tc.second);
        if(validTriggerCell(neighbor_det_id.rawId()))
        {
            neighbor_detids.emplace(neighbor_det_id.rawId());
        }
    }
    return neighbor_detids;
}


GlobalPoint 
HGCalTriggerGeometryHexLayerBasedImp1::
getTriggerCellPosition(const unsigned trigger_cell_det_id) const
{
    unsigned subdet = HGCalDetId(trigger_cell_det_id).subdetId();
    // Position: barycenter of the trigger cell.
    Basic3DVector<float> triggerCellVector(0.,0.,0.);
    const auto cell_ids = getCellsFromTriggerCell(trigger_cell_det_id);
    // BH
    if(subdet==ForwardSubdetector::HGCHEB)
    {
        for(const auto& cell : cell_ids)
        {
            HcalDetId cellDetId(cell);
            triggerCellVector += bhGeometry()->getPosition(cellDetId).basicVector();
        }
    }
    // EE or FH
    else
    {
        for(const auto& cell : cell_ids)
        {
            HGCalDetId cellDetId(cell);
            triggerCellVector += (cellDetId.subdetId()==ForwardSubdetector::HGCEE ? eeGeometry()->getPosition(cellDetId) :  fhGeometry()->getPosition(cellDetId)).basicVector();
        }
    }
    return GlobalPoint( triggerCellVector/cell_ids.size() );

}

GlobalPoint 
HGCalTriggerGeometryHexLayerBasedImp1::
getModulePosition(const unsigned module_det_id) const
{
    unsigned subdet = HGCalDetId(module_det_id).subdetId();
    // Position: barycenter of the module.
    Basic3DVector<float> moduleVector(0.,0.,0.);
    const auto cell_ids = getCellsFromModule(module_det_id);
    // BH
    if(subdet==ForwardSubdetector::HGCHEB)
    {
        for(const auto& cell : cell_ids)
        {
            HcalDetId cellDetId(cell);
            moduleVector += bhGeometry()->getPosition(cellDetId).basicVector();
        }
    }
    // EE or FH
    else
    {
        for(const auto& cell : cell_ids)
        {
            HGCalDetId cellDetId(cell);
            moduleVector += (cellDetId.subdetId()==ForwardSubdetector::HGCEE ? eeGeometry()->getPosition(cellDetId) :  fhGeometry()->getPosition(cellDetId)).basicVector();
        }
    }
    return GlobalPoint( moduleVector/cell_ids.size() );
}


void 
HGCalTriggerGeometryHexLayerBasedImp1::
fillMaps()
{
    //
    // read module mapping file
    std::ifstream l1tModulesMappingStream(l1tModulesMapping_.fullPath());
    if(!l1tModulesMappingStream.is_open())
    {
        throw cms::Exception("MissingDataFile")
            << "Cannot open HGCalTriggerGeometry L1TModulesMapping file\n";
    }
    short trigger_wafer   = 0;
    short module  = 0;
    for(; l1tModulesMappingStream>>trigger_wafer>>module; )
    {
        wafer_to_module_.emplace(trigger_wafer,module);
        module_to_wafers_.emplace(module, trigger_wafer);
        // Default number of trigger cell in wafer is 0
        number_trigger_cells_in_wafers_.emplace(trigger_wafer, 0);
    }
    if(!l1tModulesMappingStream.eof()) edm::LogWarning("HGCalTriggerGeometry") << "Error reading L1TModulesMapping '"<<trigger_wafer<<" "<<module<<"' \n";
    l1tModulesMappingStream.close();
    // read trigger cell mapping file
    std::ifstream l1tCellsMappingStream(l1tCellsMapping_.fullPath());
    if(!l1tCellsMappingStream.is_open()) 
    {
        throw cms::Exception("MissingDataFile")
            << "Cannot open HGCalTriggerGeometry L1TCellsMapping file\n";
    }
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
    // read BH trigger cell mapping file
    std::ifstream l1tCellsBHMappingStream(l1tCellsBHMapping_.fullPath());
    if(!l1tCellsBHMappingStream.is_open())
    {
        throw cms::Exception("MissingDataFile")
            << "Cannot open HGCalTriggerGeometry L1TCellsBHMapping file\n";
    }
    short ieta = 0;
    short iphi = 0;
    trigger_wafer = 0;
    trigger_cell = 0;
    for(; l1tCellsBHMappingStream>>ieta>>iphi>>trigger_wafer>>trigger_cell; )
    { 
        unsigned cell_key = packIetaIphi(ieta,iphi);
        unsigned trigger_cell_key = packWaferCellId(ForwardSubdetector::HGCHEB,trigger_wafer,trigger_cell);
        // fill cell <-> trigger cell mappings
        cells_to_trigger_cells_bh_.emplace(cell_key, trigger_cell_key);
        trigger_cells_to_cells_bh_.emplace(trigger_cell_key, cell_key);
        // fill number of trigger cells in wafers
        auto itr_insert = number_trigger_cells_in_wafers_bh_.emplace(trigger_wafer, 0);
        if(trigger_cell+1 > itr_insert.first->second) itr_insert.first->second = trigger_cell+1;
    }
    if(!l1tCellsBHMappingStream.eof()) edm::LogWarning("HGCalTriggerGeometry") << "Error reading L1TCellsBHMapping '"<<ieta<<" "<<iphi<<" "<<trigger_wafer<<" "<<trigger_cell<<"' \n";
    l1tCellsBHMappingStream.close();
}


void 
HGCalTriggerGeometryHexLayerBasedImp1::
fillNeighborMaps(const edm::FileInPath& file,  std::unordered_map<int, std::set<std::pair<short,short>>>& neighbors_map)
{
    // Fill trigger neighbor map
    std::ifstream l1tCellNeighborsMappingStream(file.fullPath());
    if(!l1tCellNeighborsMappingStream.is_open()) 
    {
        throw cms::Exception("MissingDataFile")
            << "Cannot open HGCalTriggerGeometry L1TCellNeighborsMapping file\n";
    }
    for(std::array<char,512> buffer; l1tCellNeighborsMappingStream.getline(&buffer[0], 512); )
    {
        std::string line(&buffer[0]);
        // Extract keys consisting of the module id
        // and of the trigger cell id
        // Match patterns (X,Y) 
        // where X is a number with less than 4 digis
        // and Y is a number with less than 4 digits
        std::regex key_regex("\\(\\s*\\d{1,3}\\s*,\\s*\\d{1,3}\\s*\\)");
        std::vector<std::string> key_tokens {
            std::sregex_token_iterator(line.begin(), line.end(), key_regex), {}
        };
        if(key_tokens.empty())
        {
            throw cms::Exception("BadGeometry")
                << "Syntax error in the L1TCellNeighborsMapping:\n"
                << "  Cannot find the trigger cell key in line:\n"
                << "  '"<<&buffer[0]<<"'\n";
        }
        std::regex digits_regex("\\d{1,3}");
        std::vector<std::string>  module_tc {
            std::sregex_token_iterator(key_tokens[0].begin(), key_tokens[0].end(), digits_regex), {}
        };
        // get module and cell id 
        int module = std::stoi(module_tc[0]);
        int trigger_cell = std::stoi(module_tc[1]);
        unsigned map_key = packTriggerCell(module, trigger_cell);
        // Extract neighbors
        // Match patterns (X,Y) 
        // where X is a number with less than 4 digits
        // and Y is a number with less than 4 digits
        std::regex neighbors_regex("\\(\\s*\\d{1,3}\\s*,\\s*\\d{1,3}\\s*\\)");
        std::vector<std::string> neighbors_tokens {
            std::sregex_token_iterator(line.begin(), line.end(), neighbors_regex), {}
        };
        if(neighbors_tokens.size()<2)
        {
            throw cms::Exception("BadGeometry")
                << "Syntax error in the L1TCellNeighborsMapping:\n"
                << "  Cannot find any neighbor in line:\n"
                << "  '"<<&buffer[0]<<"'\n";
        }
        auto itr_insert = neighbors_map.emplace(map_key, std::set<std::pair<short,short>>());
        // The first element is the key, so start at index 1
        for(unsigned i=1; i<neighbors_tokens.size(); i++)
        {
            const auto& neighbor = neighbors_tokens[i];
            std::vector<std::string>  pair_neighbor {
                std::sregex_token_iterator(neighbor.begin(), neighbor.end(), digits_regex), {}
            };
            short neighbor_module(std::stoi(pair_neighbor[0]));
            short neighbor_cell(std::stoi(pair_neighbor[1]));
            itr_insert.first->second.emplace(neighbor_module, neighbor_cell);
        }
    }
    if(!l1tCellNeighborsMappingStream.eof()) edm::LogWarning("HGCalTriggerGeometry") << "Error reading L1TCellNeighborsMapping'\n";
    l1tCellNeighborsMappingStream.close();

}



void 
HGCalTriggerGeometryHexLayerBasedImp1::
fillInvalidTriggerCells()
{
    unsigned n_layers_ee = eeTopology().dddConstants().layers(true);
    for(unsigned layer=1; layer<=n_layers_ee; layer++)
    {
        for(const auto& wafer_module : wafer_to_module_)
        {
            unsigned wafer = wafer_module.first;
            // loop on the trigger cells in each wafer
            for(int trigger_cell=0; trigger_cell<number_trigger_cells_in_wafers_.at(wafer); trigger_cell++)
            {
                std::set<unsigned> trigger_cell_ids;
                trigger_cell_ids.emplace(HGCalDetId(ForwardSubdetector::HGCEE, -1, layer, 1, wafer, trigger_cell));
                trigger_cell_ids.emplace(HGCalDetId(ForwardSubdetector::HGCEE, 1, layer, 1, wafer, trigger_cell));
                trigger_cell_ids.emplace(HGCalDetId(ForwardSubdetector::HGCHEF, -1, layer, 1, wafer, trigger_cell));
                trigger_cell_ids.emplace(HGCalDetId(ForwardSubdetector::HGCHEF, 1, layer, 1, wafer, trigger_cell));
                for(unsigned trigger_cell : trigger_cell_ids)
                {
                    if(!validTriggerCellFromCells(trigger_cell)) invalid_triggercells_.emplace(trigger_cell);
                    for(unsigned neighbor : getNeighborsFromTriggerCell(trigger_cell))
                    {
                        if(!validTriggerCellFromCells(neighbor)) invalid_triggercells_.emplace(neighbor);
                    }
                }
            }
        }
    }
}

unsigned 
HGCalTriggerGeometryHexLayerBasedImp1::
packWaferCellId(unsigned subdet, unsigned wafer, unsigned cell) const
{
    unsigned packed_value = 0;
    const int kSubdetMask = 0x7;
    packed_value |= ((cell & HGCalDetId::kHGCalCellMask) << HGCalDetId::kHGCalCellOffset);
    packed_value |= ((wafer & HGCalDetId::kHGCalWaferMask) << HGCalDetId::kHGCalWaferOffset);
    packed_value |= ((subdet & kSubdetMask) << (HGCalDetId::kHGCalWaferTypeOffset));
    return packed_value;
}


unsigned 
HGCalTriggerGeometryHexLayerBasedImp1::
packIetaIphi(unsigned ieta, unsigned iphi) const
{
    unsigned packed_value = 0;
    packed_value |= (iphi & HcalDetId::kHcalPhiMask2);
    packed_value |= ((ieta & HcalDetId::kHcalEtaMask2) << HcalDetId::kHcalEtaOffset2);
    return packed_value;
}

void
HGCalTriggerGeometryHexLayerBasedImp1::
unpackWaferCellId(unsigned wafer_cell, unsigned& wafer, unsigned& cell) const
{
    cell =  wafer_cell & HGCalDetId::kHGCalCellMask;
    wafer = (wafer_cell>>HGCalDetId::kHGCalWaferOffset) & HGCalDetId::kHGCalWaferMask;
}


void
HGCalTriggerGeometryHexLayerBasedImp1::
unpackIetaIphi(unsigned ieta_iphi, unsigned& ieta, unsigned& iphi) const
{
    iphi =  ieta_iphi & HcalDetId::kHcalPhiMask2;
    ieta = (ieta_iphi>>HcalDetId::kHcalEtaOffset2) & HcalDetId::kHcalEtaMask2;
}

bool 
HGCalTriggerGeometryHexLayerBasedImp1::
validTriggerCell(const unsigned trigger_cell_id) const
{
    return invalid_triggercells_.find(trigger_cell_id)==invalid_triggercells_.end();
}

bool 
HGCalTriggerGeometryHexLayerBasedImp1::
disconnectedModule(const unsigned module_id) const
{
    bool disconnected = false;
    if(disconnected_modules_.find(HGCalDetId(module_id).wafer())!=disconnected_modules_.end()) disconnected = true;
    if(disconnected_layers_.find(layerWithOffset(module_id))!=disconnected_layers_.end()) disconnected = true;
    return disconnected;
}

unsigned 
HGCalTriggerGeometryHexLayerBasedImp1::
triggerLayer(const unsigned id) const
{
    unsigned layer = layerWithOffset(id);
    if(layer>=trigger_layers_.size()) return 0;
    return trigger_layers_[layer];
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
            is_valid = eeTopology().valid(cell_id);
            break;
        case ForwardSubdetector::HGCHEF:
            is_valid = fhTopology().valid(cell_id);
            break;
        case ForwardSubdetector::HGCHEB:
            is_valid = bhTopology().valid(cell_id);
            break;
        default:
            is_valid = false;
            break;
    } 
    return is_valid;
}


unsigned 
HGCalTriggerGeometryHexLayerBasedImp1::
packTriggerCell(unsigned module, unsigned trigger_cell) const
{
    unsigned packed_value = 0;
    packed_value |= ((trigger_cell & HGCalDetId::kHGCalCellMask) << HGCalDetId::kHGCalCellOffset);
    packed_value |= ((module & HGCalDetId::kHGCalWaferMask) << HGCalDetId::kHGCalWaferOffset);
    return packed_value;
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
            wafer_type = (eeTopology().dddConstants().waferTypeT(wafer)==2?-1:1);
            break;
        case ForwardSubdetector::HGCHEF:
            wafer_type = (fhTopology().dddConstants().waferTypeT(wafer)==2?-1:1);
            break;
        default:
            break;
    };
    return wafer_type;
}


unsigned
HGCalTriggerGeometryHexLayerBasedImp1::
layerWithOffset(unsigned id) const
{
    HGCalDetId detid(id);
    unsigned layer = 0;
    switch(detid.subdetId())
    {
        case ForwardSubdetector::HGCEE:
            layer = detid.layer();
            break;
        case ForwardSubdetector::HGCHEF:
            layer = fhOffset_ + detid.layer();
            break;
        case ForwardSubdetector::HGCHEB:
            layer = bhOffset_ + detid.layer();
            break;
    };
    return layer;
}



DEFINE_EDM_PLUGIN(HGCalTriggerGeometryFactory, 
        HGCalTriggerGeometryHexLayerBasedImp1,
        "HGCalTriggerGeometryHexLayerBasedImp1");
