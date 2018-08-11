#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <regex>


class HGCalTriggerGeometryV9Imp1 : public HGCalTriggerGeometryBase
{
    public:
        HGCalTriggerGeometryV9Imp1(const edm::ParameterSet& conf);

        void initialize(const edm::ESHandle<CaloGeometry>& ) final;
        void initialize(const edm::ESHandle<HGCalGeometry>&,
                const edm::ESHandle<HGCalGeometry>&,
                const edm::ESHandle<HGCalGeometry>&) final;
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
        edm::FileInPath l1tCellsSciMapping_;
        edm::FileInPath l1tWafersMapping_;
        edm::FileInPath l1tModulesMapping_;
        edm::FileInPath l1tCellNeighborsMapping_;
        edm::FileInPath l1tCellNeighborsSciMapping_;

        // module related maps
        std::unordered_map<unsigned, unsigned> wafer_to_module_;
        std::unordered_multimap<unsigned, unsigned> module_to_wafers_;

        // trigger cell related maps
        std::unordered_map<unsigned, unsigned> cells_to_trigger_cells_;
        std::unordered_multimap<unsigned, unsigned> trigger_cells_to_cells_;
        std::unordered_map<unsigned, unsigned> cells_to_trigger_cells_sci_;
        std::unordered_multimap<unsigned, unsigned> trigger_cells_to_cells_sci_;
        std::unordered_map<unsigned, unsigned short> number_trigger_cells_in_wafers_; 
        std::unordered_map<unsigned, unsigned short> number_trigger_cells_in_wafers_sci_; 
        std::unordered_map<unsigned, unsigned> wafers_to_wafers_old_;
        std::unordered_map<unsigned, unsigned> wafers_old_to_wafers_;
        std::unordered_set<unsigned> invalid_triggercells_;

        // neighbor related maps
        // trigger cell neighbors:
        // - The key includes the module and trigger cell id
        // - The value is a set of (module_id, trigger_cell_id)
        typedef std::unordered_map<int, std::set<std::pair<short,short>>> neighbor_map;
        neighbor_map trigger_cell_neighbors_;
        neighbor_map trigger_cell_neighbors_sci_;

        // Disconnected modules and layers
        std::unordered_set<unsigned> disconnected_modules_;
        std::unordered_set<unsigned> disconnected_layers_;
        std::vector<unsigned> trigger_layers_;

        // layer offsets 
        unsigned heOffset_;
        unsigned totalLayers_;

        void fillMaps();
        void fillNeighborMap(const edm::FileInPath&,  neighbor_map&, bool);
        void fillInvalidTriggerCells();
        unsigned packTriggerCell(unsigned, unsigned) const;
        unsigned packTriggerCellWithType(unsigned, unsigned, unsigned) const;
        bool validCellId(unsigned det, unsigned cell_id) const;
        bool validTriggerCellFromCells( const unsigned ) const;

        int detIdWaferType(unsigned det, unsigned layer, short waferU, short waferV) const;
        unsigned packWaferCellId(unsigned subdet, unsigned wafer, unsigned cell) const;
        unsigned packWaferId(int waferU, int waferV) const;
        unsigned packCellId(unsigned type, unsigned cellU, unsigned cellV) const;
        unsigned packCellId(unsigned type, unsigned cell) const;
        unsigned packIetaIphi(unsigned ieta, unsigned iphi) const;
        void unpackWaferCellId(unsigned wafer_cell, unsigned& wafer, unsigned& cell) const;
        void unpackWaferId(unsigned wafer, int& waferU, int& waferV) const;
        void unpackCellId(unsigned cell, unsigned& cellU, unsigned& cellV) const;
        void unpackIetaIphi(unsigned ieta_iphi, unsigned& ieta, unsigned& iphi) const;

        unsigned layerWithOffset(unsigned) const;
};


HGCalTriggerGeometryV9Imp1::
HGCalTriggerGeometryV9Imp1(const edm::ParameterSet& conf):
    HGCalTriggerGeometryBase(conf),
    l1tCellsMapping_(conf.getParameter<edm::FileInPath>("L1TCellsMapping")),
    l1tCellsSciMapping_(conf.getParameter<edm::FileInPath>("L1TCellsSciMapping")),
    l1tWafersMapping_(conf.getParameter<edm::FileInPath>("L1TWafersMapping")),
    l1tModulesMapping_(conf.getParameter<edm::FileInPath>("L1TModulesMapping")),
    l1tCellNeighborsMapping_(conf.getParameter<edm::FileInPath>("L1TCellNeighborsMapping")),
    l1tCellNeighborsSciMapping_(conf.getParameter<edm::FileInPath>("L1TCellNeighborsSciMapping"))
{
    std::vector<unsigned> tmp_vector = conf.getParameter<std::vector<unsigned>>("DisconnectedModules");
    std::move(tmp_vector.begin(), tmp_vector.end(), std::inserter(disconnected_modules_, disconnected_modules_.end()));
    tmp_vector = conf.getParameter<std::vector<unsigned>>("DisconnectedLayers");
    std::move(tmp_vector.begin(), tmp_vector.end(), std::inserter(disconnected_layers_, disconnected_layers_.end()));
}

void
HGCalTriggerGeometryV9Imp1::
reset()
{
    cells_to_trigger_cells_.clear();
    trigger_cells_to_cells_.clear();
    cells_to_trigger_cells_sci_.clear();
    trigger_cells_to_cells_sci_.clear();
    wafers_to_wafers_old_.clear();
    wafers_old_to_wafers_.clear();
    wafer_to_module_.clear();
    module_to_wafers_.clear();
    number_trigger_cells_in_wafers_.clear();
    number_trigger_cells_in_wafers_sci_.clear();
    trigger_cell_neighbors_.clear();
    trigger_cell_neighbors_sci_.clear();
}

void
HGCalTriggerGeometryV9Imp1::
initialize(const edm::ESHandle<CaloGeometry>& calo_geometry)
{
    throw cms::Exception("BadGeometry")
        << "HGCalTriggerGeometryV9Imp1 geometry cannot be initialized with the V7/V8 HGCAL geometry";
}

void
HGCalTriggerGeometryV9Imp1::
initialize(const edm::ESHandle<HGCalGeometry>& hgc_ee_geometry,
        const edm::ESHandle<HGCalGeometry>& hgc_hsi_geometry,
        const edm::ESHandle<HGCalGeometry>& hgc_hsc_geometry
        )
{
    setEEGeometry(hgc_ee_geometry);
    setHSiGeometry(hgc_hsi_geometry);
    setHScGeometry(hgc_hsc_geometry);
    heOffset_ = eeTopology().dddConstants().layers(true);
    totalLayers_ = heOffset_ + hsiTopology().dddConstants().layers(true);
    trigger_layers_.resize(totalLayers_+1);
    unsigned trigger_layer = 0;
    for(unsigned layer=0; layer<totalLayers_; layer++)
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
    fillNeighborMap(l1tCellNeighborsMapping_, trigger_cell_neighbors_, false); // silicon 
    fillNeighborMap(l1tCellNeighborsSciMapping_, trigger_cell_neighbors_sci_, true); // scintillator
    fillInvalidTriggerCells();

}

unsigned 
HGCalTriggerGeometryV9Imp1::
getTriggerCellFromCell( const unsigned cell_id ) const
{
    unsigned det = DetId(cell_id).det();
    unsigned subdet = 0;
    int zside = 0;
    unsigned tc_type = 1;
    unsigned layer = 0;
    unsigned wafer_trigger_cell = 0;
    unsigned trigger_cell = 0;
    // Scintillator
    if(det == DetId::HGCalHSc)
    {
        HGCScintillatorDetId cell_det_id(cell_id);
        unsigned ieta = cell_det_id.ietaAbs();
        unsigned iphi = cell_det_id.iphi();
        tc_type = cell_det_id.type();
        layer = cell_det_id.layer();
        subdet = ForwardSubdetector::HGCHEB;
        zside = cell_det_id.zside();
        auto trigger_cell_itr = cells_to_trigger_cells_sci_.find(packIetaIphi(ieta, iphi));
        if(trigger_cell_itr==cells_to_trigger_cells_sci_.end())
        {
            throw cms::Exception("BadGeometry")
                << "HGCalTriggerGeometry: scintillator cell ieta=" << ieta << ", iphi="<<iphi<<" is not mapped to any trigger cell. The trigger cell mapping should be modified.\n";
        }
        trigger_cell = 0;
        wafer_trigger_cell = 0;
        unpackWaferCellId(trigger_cell_itr->second, wafer_trigger_cell, trigger_cell);
    }
    // Silicon
    else if(det == DetId::HGCalEE || det == DetId::HGCalHSi)
    {
        HGCSiliconDetId cell_det_id(cell_id);
        subdet = (det==DetId::HGCalEE ? ForwardSubdetector::HGCEE : ForwardSubdetector::HGCHEF);
        layer = cell_det_id.layer();
        zside = cell_det_id.zside();
        int type =  cell_det_id.type();
        int waferu = cell_det_id.waferU();
        int waferv = cell_det_id.waferV();
        unsigned cellu = cell_det_id.cellU();
        unsigned cellv = cell_det_id.cellV();
        auto trigger_cell_itr = cells_to_trigger_cells_.find(packCellId(type, cellu, cellv));
        if(trigger_cell_itr==cells_to_trigger_cells_.end())
        {
            throw cms::Exception("BadGeometry")
                << "HGCalTriggerGeometry: HGCal  cell " << cellu << "," << cellv << " in wafer type "<<type<<" is not mapped to any trigger cell. The trigger cell mapping should be modified.\n";
        }
        auto wafer_trigger_cell_itr = wafers_to_wafers_old_.find(packWaferId(waferu, waferv));
        if(wafer_trigger_cell_itr==wafers_to_wafers_old_.end())
        {
            throw cms::Exception("BadGeometry")
                << "HGCalTriggerGeometry: Wafer "<<waferu<<","<<waferv<<" is not mapped to any trigger wafer ID. The wafer mapping should be modified.\n";
        }
        trigger_cell = trigger_cell_itr->second;
        wafer_trigger_cell = wafer_trigger_cell_itr->second;
    }
    // Using the old HGCalDetId for trigger cells is temporary
    // For easy switch between V8 and V9 geometries
    return HGCalDetId((ForwardSubdetector)subdet, zside, layer, tc_type, wafer_trigger_cell, trigger_cell).rawId();

}

unsigned 
HGCalTriggerGeometryV9Imp1::
getModuleFromCell( const unsigned cell_id ) const
{
    return getModuleFromTriggerCell(getTriggerCellFromCell(cell_id));
}

unsigned 
HGCalTriggerGeometryV9Imp1::
getModuleFromTriggerCell( const unsigned trigger_cell_id ) const
{
    HGCalDetId trigger_cell_det_id(trigger_cell_id);
    unsigned module = 0;
    // Scintillator
    if(trigger_cell_det_id.subdetId()==ForwardSubdetector::HGCHEB)
    {
        // For scintillator, the module ID is currently encoded as the wafer in HGCalDetId
        module = trigger_cell_det_id.wafer();
    }
    // Silicon
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
    return HGCalDetId((ForwardSubdetector)trigger_cell_det_id.subdetId(), trigger_cell_det_id.zside(), trigger_cell_det_id.layer(), trigger_cell_det_id.waferType(), module, HGCalDetId::kHGCalCellMask).rawId();
}

HGCalTriggerGeometryBase::geom_set 
HGCalTriggerGeometryV9Imp1::
getCellsFromTriggerCell( const unsigned trigger_cell_id ) const
{
    HGCalDetId trigger_cell_det_id(trigger_cell_id);
    geom_set cell_det_ids;
    unsigned subdet = trigger_cell_det_id.subdetId();
    unsigned trigger_wafer = trigger_cell_det_id.wafer();
    unsigned trigger_cell = trigger_cell_det_id.cell();
    unsigned layer = trigger_cell_det_id.layer();
    // Scintillator
    if(subdet==ForwardSubdetector::HGCHEB)
    {
        int type =  hscTopology().dddConstants().getTypeTrap(layer);
        const auto& cell_range = trigger_cells_to_cells_sci_.equal_range(packWaferCellId(subdet, trigger_wafer, trigger_cell));
        for(auto tc_c_itr=cell_range.first; tc_c_itr!=cell_range.second; tc_c_itr++)
        {
            unsigned ieta = 0;
            unsigned iphi = 0;
            unpackIetaIphi(tc_c_itr->second, ieta, iphi);
            unsigned cell_det_id = HGCScintillatorDetId(type, layer, trigger_cell_det_id.zside()*ieta, iphi).rawId();
            if(validCellId(subdet, cell_det_id)) cell_det_ids.emplace(cell_det_id);
        }
    }
    // Silicon
    else
    {
        int waferu = 0;
        int waferv = 0;
        auto wafer_itr = wafers_old_to_wafers_.find(trigger_wafer);
        if(wafer_itr==wafers_old_to_wafers_.end())
        {
            throw cms::Exception("BadGeometry")
                << "HGCalTriggerGeometry: Trigger wafer ID "<<trigger_wafer<<" is not mapped to any wafer. The wafer mapping should be modified.\n";
        }
        unpackWaferId(wafer_itr->second, waferu, waferv);
        DetId::Detector det = (trigger_cell_det_id.subdetId()==ForwardSubdetector::HGCEE ? DetId::HGCalEE : DetId::HGCalHSi);
        unsigned wafer_type = detIdWaferType(det, layer, waferu, waferv);
        const auto& cell_range = trigger_cells_to_cells_.equal_range(packCellId(wafer_type, trigger_cell));
        for(auto tc_c_itr=cell_range.first; tc_c_itr!=cell_range.second; tc_c_itr++)
        {
            unsigned cellu = 0;
            unsigned cellv = 0;
            unpackCellId(tc_c_itr->second, cellu, cellv);
            unsigned cell_det_id = HGCSiliconDetId(det, trigger_cell_det_id.zside(), wafer_type, layer, waferu, waferv, cellu, cellv).rawId();
            if(validCellId(subdet, cell_det_id)) cell_det_ids.emplace(cell_det_id);
        }
    }
    return cell_det_ids;
}

HGCalTriggerGeometryBase::geom_set 
HGCalTriggerGeometryV9Imp1::
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
HGCalTriggerGeometryV9Imp1::
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
HGCalTriggerGeometryV9Imp1::
getTriggerCellsFromModule( const unsigned module_id ) const
{
    HGCalDetId module_det_id(module_id);
    geom_set trigger_cell_det_ids;
    unsigned module = module_det_id.wafer();
    // Scintillator
    if(module_det_id.subdetId()==ForwardSubdetector::HGCHEB)
    {
        // loop on the trigger cells in each module
        for(int trigger_cell=0; trigger_cell<number_trigger_cells_in_wafers_sci_.at(module); trigger_cell++)
        {
            HGCalDetId trigger_cell_id((ForwardSubdetector)module_det_id.subdetId(), module_det_id.zside(), module_det_id.layer(), module_det_id.waferType(), module, trigger_cell);
            if(validTriggerCell(trigger_cell_id)) trigger_cell_det_ids.emplace(trigger_cell_id.rawId());
        }
    }
    // Silicon
    else
    {
        auto wafer_itrs = module_to_wafers_.equal_range(module);
        // loop on the wafers included in the module
        for(auto wafer_itr=wafer_itrs.first; wafer_itr!=wafer_itrs.second; wafer_itr++)
        {
            unsigned wafer = wafer_itr->second;
            auto waferuv_itr = wafers_old_to_wafers_.find(wafer);
            if(waferuv_itr==wafers_old_to_wafers_.end()) continue;
            int waferu = 0;
            int waferv = 0;
            unpackWaferId(waferuv_itr->second, waferu, waferv);
            DetId::Detector det = (module_det_id.subdetId()==ForwardSubdetector::HGCEE ? DetId::HGCalEE : DetId::HGCalHSi);
            unsigned layer = module_det_id.layer();
            unsigned wafer_type = detIdWaferType(det, layer, waferu, waferv);
            // loop on the trigger cells in each wafer
            for(int trigger_cell=0; trigger_cell<number_trigger_cells_in_wafers_.at(wafer_type); trigger_cell++)
            {
                HGCalDetId trigger_cell_id((ForwardSubdetector)module_det_id.subdetId(), module_det_id.zside(), module_det_id.layer(), module_det_id.waferType(), wafer, trigger_cell);
                if(validTriggerCell(trigger_cell_id)) trigger_cell_det_ids.emplace(trigger_cell_id.rawId());
            }
        }
    }
    return trigger_cell_det_ids;
}

HGCalTriggerGeometryBase::geom_ordered_set 
HGCalTriggerGeometryV9Imp1::
getOrderedTriggerCellsFromModule( const unsigned module_id ) const
{
    HGCalDetId module_det_id(module_id);
    geom_ordered_set trigger_cell_det_ids;
    unsigned module = module_det_id.wafer();
    // Scintillator
    if(module_det_id.subdetId()==ForwardSubdetector::HGCHEB)
    {
        // loop on the trigger cells in each module
        for(int trigger_cell=0; trigger_cell<number_trigger_cells_in_wafers_sci_.at(module); trigger_cell++)
        {
            HGCalDetId trigger_cell_id((ForwardSubdetector)module_det_id.subdetId(), module_det_id.zside(), module_det_id.layer(), module_det_id.waferType(), module, trigger_cell);
            if(validTriggerCell(trigger_cell_id)) trigger_cell_det_ids.emplace(trigger_cell_id.rawId());
        }
    }
    // EE or FH
    else
    {
        auto wafer_itrs = module_to_wafers_.equal_range(module);
        // loop on the wafers included in the module
        for(auto wafer_itr=wafer_itrs.first; wafer_itr!=wafer_itrs.second; wafer_itr++)
        {
            unsigned wafer = wafer_itr->second;
            auto waferuv_itr = wafers_old_to_wafers_.find(wafer);
            if(waferuv_itr==wafers_old_to_wafers_.end()) continue;
            int waferu = 0;
            int waferv = 0;
            unpackWaferId(waferuv_itr->second, waferu, waferv);
            DetId::Detector det = (module_det_id.subdetId()==ForwardSubdetector::HGCEE ? DetId::HGCalEE : DetId::HGCalHSi);
            unsigned layer = module_det_id.layer();
            unsigned wafer_type = detIdWaferType(det, layer, waferu, waferv);
            // loop on the trigger cells in each wafer
            for(int trigger_cell=0; trigger_cell<number_trigger_cells_in_wafers_.at(wafer_type); trigger_cell++)
            {
                HGCalDetId trigger_cell_id((ForwardSubdetector)module_det_id.subdetId(), module_det_id.zside(), module_det_id.layer(), module_det_id.waferType(), wafer, trigger_cell);
                if(validTriggerCell(trigger_cell_id)) trigger_cell_det_ids.emplace(trigger_cell_id.rawId());
            }
        }
    }
    return trigger_cell_det_ids;
}



HGCalTriggerGeometryBase::geom_set
HGCalTriggerGeometryV9Imp1::
getNeighborsFromTriggerCell( const unsigned trigger_cell_id ) const
{
    HGCalDetId trigger_cell_det_id(trigger_cell_id);
    // Choose scintillator or silicon map
    const auto& neighbors_map = (trigger_cell_det_id.subdetId()==ForwardSubdetector::HGCHEB ? trigger_cell_neighbors_sci_ : trigger_cell_neighbors_);
    unsigned layer = trigger_cell_det_id.layer();
    unsigned type = (trigger_cell_det_id.subdetId()==ForwardSubdetector::HGCHEB ? hscTopology().dddConstants().getTypeTrap(layer):  1);
    unsigned module = trigger_cell_det_id.wafer();
    unsigned trigger_cell = trigger_cell_det_id.cell();
    // retrieve neighbors
    unsigned trigger_cell_key = (trigger_cell_det_id.subdetId()==ForwardSubdetector::HGCHEB ? packTriggerCellWithType(type, module, trigger_cell) : packTriggerCell(module, trigger_cell));
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
        HGCalDetId neighbor_det_id((ForwardSubdetector)trigger_cell_det_id.subdetId(), trigger_cell_det_id.zside(), layer, type, module_tc.first, module_tc.second);
        if(validTriggerCell(neighbor_det_id.rawId()))
        {
            neighbor_detids.emplace(neighbor_det_id.rawId());
        }
    }
    return neighbor_detids;
}


GlobalPoint 
HGCalTriggerGeometryV9Imp1::
getTriggerCellPosition(const unsigned trigger_cell_det_id) const
{
    unsigned subdet = HGCalDetId(trigger_cell_det_id).subdetId();
    // Position: barycenter of the trigger cell.
    Basic3DVector<float> triggerCellVector(0.,0.,0.);
    const auto cell_ids = getCellsFromTriggerCell(trigger_cell_det_id);
    // Scintillator
    if(subdet==ForwardSubdetector::HGCHEB)
    {
        for(const auto& cell : cell_ids)
        {
            HcalDetId cellDetId(cell);
            triggerCellVector += hscGeometry()->getPosition(cellDetId).basicVector();
        }
    }
    // Silicon
    else
    {
        for(const auto& cell : cell_ids)
        {
            HGCSiliconDetId cellDetId(cell);
            triggerCellVector += (cellDetId.det()==DetId::HGCalEE ? eeGeometry()->getPosition(cellDetId) : hsiGeometry()->getPosition(cellDetId)).basicVector();
        }
    }
    return GlobalPoint( triggerCellVector/cell_ids.size() );

}

GlobalPoint 
HGCalTriggerGeometryV9Imp1::
getModulePosition(const unsigned module_det_id) const
{
    unsigned subdet = HGCalDetId(module_det_id).subdetId();
    // Position: barycenter of the module.
    Basic3DVector<float> moduleVector(0.,0.,0.);
    const auto cell_ids = getCellsFromModule(module_det_id);
    // Scintillator
    if(subdet==ForwardSubdetector::HGCHEB)
    {
        for(const auto& cell : cell_ids)
        {
            HGCScintillatorDetId cellDetId(cell);
            moduleVector += hscGeometry()->getPosition(cellDetId).basicVector();
        }
    }
    // Silicon
    else
    {
        for(const auto& cell : cell_ids)
        {
            HGCSiliconDetId cellDetId(cell);
            moduleVector += (cellDetId.det()==DetId::HGCalEE ? eeGeometry()->getPosition(cellDetId) :  hsiGeometry()->getPosition(cellDetId)).basicVector();
        }
    }
    return GlobalPoint( moduleVector/cell_ids.size() );
}


void 
HGCalTriggerGeometryV9Imp1::
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
    }
    if(!l1tModulesMappingStream.eof())
    {
        throw cms::Exception("BadGeometryFile")
            << "Error reading L1TModulesMapping '"<<trigger_wafer<<" "<<module<<"' \n";
    }
    l1tModulesMappingStream.close();
    // read trigger cell mapping file
    std::ifstream l1tCellsMappingStream(l1tCellsMapping_.fullPath());
    if(!l1tCellsMappingStream.is_open()) 
    {
        throw cms::Exception("MissingDataFile")
            << "Cannot open HGCalTriggerGeometry L1TCellsMapping file\n";
    }
    short type = 0;
    short cellu = 0;
    short cellv = 0;
    short trigger_cell = 0;
    for(; l1tCellsMappingStream>>type>>cellu>>cellv>>trigger_cell; )
    { 
        unsigned cell_key = packCellId(type,cellu,cellv);
        unsigned trigger_cell_key = packCellId(type,trigger_cell);
        // fill cell <-> trigger cell mappings
        cells_to_trigger_cells_.emplace(cell_key, trigger_cell);
        trigger_cells_to_cells_.emplace(trigger_cell_key, cell_key);
        // fill number of trigger cells in wafers
        auto itr_insert = number_trigger_cells_in_wafers_.emplace(type, 0);
        if(trigger_cell+1 > itr_insert.first->second) itr_insert.first->second = trigger_cell+1;
    }
    if(!l1tCellsMappingStream.eof())
    {
        throw cms::Exception("BadGeometryFile")
            << "Error reading L1TCellsMapping '"<<type<<" "<<cellu<<" "<<cellv<<" "<<trigger_cell<<"' \n";
    }
    l1tCellsMappingStream.close();
    // read scintillator trigger cell mapping file
    std::ifstream l1tCellsSciMappingStream(l1tCellsSciMapping_.fullPath());
    if(!l1tCellsSciMappingStream.is_open())
    {
        throw cms::Exception("MissingDataFile")
            << "Cannot open HGCalTriggerGeometry L1TCellsSciMapping file\n";
    }
    short ieta = 0;
    short iphi = 0;
    trigger_wafer = 0;
    trigger_cell = 0;
    for(; l1tCellsSciMappingStream>>ieta>>iphi>>trigger_wafer>>trigger_cell; )
    { 
        unsigned cell_key = packIetaIphi(ieta,iphi);
        unsigned trigger_cell_key = packWaferCellId(ForwardSubdetector::HGCHEB,trigger_wafer,trigger_cell);
        // fill cell <-> trigger cell mappings
        cells_to_trigger_cells_sci_.emplace(cell_key, trigger_cell_key);
        trigger_cells_to_cells_sci_.emplace(trigger_cell_key, cell_key);
        // fill number of trigger cells in wafers
        auto itr_insert = number_trigger_cells_in_wafers_sci_.emplace(trigger_wafer, 0);
        if(trigger_cell+1 > itr_insert.first->second) itr_insert.first->second = trigger_cell+1;
    }
    if(!l1tCellsSciMappingStream.eof())
    {
        throw cms::Exception("BadGeometryFile")
            << "Error reading L1TCellsSciMapping '"<<ieta<<" "<<iphi<<" "<<trigger_wafer<<" "<<trigger_cell<<"' \n";
    }
    l1tCellsSciMappingStream.close();
    // read wafer mapping file
    std::ifstream l1tWafersMappingStream(l1tWafersMapping_.fullPath());
    if(!l1tWafersMappingStream.is_open()) 
    {
        throw cms::Exception("MissingDataFile")
            << "Cannot open HGCalTriggerGeometry L1TWafersMapping file\n";
    }
    short waferu = 0;
    short waferv = 0;
    trigger_wafer = 0;
    for(; l1tWafersMappingStream>>waferu>>waferv>>trigger_wafer; )
    { 
        unsigned wafer_key = packWaferId(waferu,waferv);
        // fill wafer u,v <-> old wafer ID mappings
        wafers_to_wafers_old_.emplace(wafer_key, trigger_wafer);
        wafers_old_to_wafers_.emplace(trigger_wafer, wafer_key);
    }
    if(!l1tWafersMappingStream.eof())
    {
        throw cms::Exception("BadGeometryFile")
            << "Error reading L1TWafersMapping '"<<waferu<<" "<<waferv<<" "<<trigger_wafer<<"' \n";
    }
    l1tWafersMappingStream.close();
}


void 
HGCalTriggerGeometryV9Imp1::
fillNeighborMap(const edm::FileInPath& file,  neighbor_map& neighbors_map, bool scintillator)
{
    // Fill trigger neighbor map
    std::ifstream l1tCellNeighborsMappingStream(file.fullPath());
    if(!l1tCellNeighborsMappingStream.is_open()) 
    {
        throw cms::Exception("MissingDataFile")
            << "Cannot open HGCalTriggerGeometry L1TCellNeighborsMapping file\n";
    }
    const unsigned line_size = 512;
    for(std::array<char,line_size> buffer; l1tCellNeighborsMappingStream.getline(&buffer[0], line_size); )
    {
        std::string line(&buffer[0]);
        // Extract keys consisting of the module id
        // and of the trigger cell id
        // Match patterns (X,Y) 
        // where X is a number with less than 4 digis
        // and Y is a number with less than 4 digits
        // For the scintillator case, match instead (X,Y,Z) patterns
        std::regex key_regex(scintillator ? "\\(\\s*\\d{1,3}\\s*,\\s*\\d{1,3}\\s*,\\s*\\d{1,3}\\s*\\)" : "\\(\\s*\\d{1,3}\\s*,\\s*\\d{1,3}\\s*\\)");
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
        unsigned map_key = 0;
        if(scintillator)
        { 
            int type = std::stoi(module_tc[0]);
            int module = std::stoi(module_tc[1]);
            int trigger_cell = std::stoi(module_tc[2]);
            map_key = packTriggerCellWithType(type, module, trigger_cell);
        }
        else
        {
            int module = std::stoi(module_tc[0]);
            int trigger_cell = std::stoi(module_tc[1]);
            map_key = packTriggerCell(module, trigger_cell);
        }
        // Extract neighbors
        // Match patterns (X,Y) 
        // where X is a number with less than 4 digits
        // and Y is a number with less than 4 digits
        std::regex neighbors_regex("\\(\\s*\\d{1,3}\\s*,\\s*\\d{1,3}\\s*\\)");
        std::vector<std::string> neighbors_tokens {
            std::sregex_token_iterator(line.begin(), line.end(), neighbors_regex), {}
        };
        if( (scintillator && neighbors_tokens.empty()) ||
            (!scintillator && neighbors_tokens.size()<2)
            )
        {
            throw cms::Exception("BadGeometry")
                << "Syntax error in the L1TCellNeighborsMapping:\n"
                << "  Cannot find any neighbor in line:\n"
                << "  '"<<&buffer[0]<<"'\n";
        }
        auto itr_insert = neighbors_map.emplace(map_key, std::set<std::pair<short,short>>());
        // The first element for silicon neighbors is the key, so start at index 1
        unsigned first_element = (scintillator ? 0 : 1);
        for(unsigned i=first_element; i<neighbors_tokens.size(); i++)
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
    if(!l1tCellNeighborsMappingStream.eof())
    {
        throw cms::Exception("BadGeometryFile")
            << "Error reading L1TCellNeighborsMapping'\n";
    }
    l1tCellNeighborsMappingStream.close();

}


void 
HGCalTriggerGeometryV9Imp1::
fillInvalidTriggerCells()
{
    unsigned n_layers_ee = eeTopology().dddConstants().layers(true);
    for(unsigned layer=1; layer<=n_layers_ee; layer++)
    {
        for(const auto& waferuv_wafer : wafers_to_wafers_old_)
        {
            int waferu = 0;
            int waferv = 0;
            unpackWaferId(waferuv_wafer.first, waferu, waferv);
            unsigned waferee_type = detIdWaferType(DetId::HGCalEE, layer, waferu, waferv);
            unsigned waferfh_type = detIdWaferType(DetId::HGCalHSi, layer, waferu, waferv);
            unsigned trigger_wafer = waferuv_wafer.second;
            // loop on the trigger cells in each wafer
            for(int trigger_cell=0; trigger_cell<number_trigger_cells_in_wafers_.at(waferee_type); trigger_cell++)
            {
                for(int zside : {-1,1})
                {
                    HGCalDetId trigger_cell_id(ForwardSubdetector::HGCEE, zside, layer, 1, trigger_wafer, trigger_cell);
                    if(!validTriggerCellFromCells(trigger_cell_id)) invalid_triggercells_.emplace(trigger_cell_id);
                    for(unsigned neighbor : getNeighborsFromTriggerCell(trigger_cell_id))
                    {
                        auto wafer_itr = wafers_old_to_wafers_.find(HGCalDetId(neighbor).wafer());
                        if(wafer_itr==wafers_old_to_wafers_.end()) invalid_triggercells_.emplace(neighbor);
                        else if(!validTriggerCellFromCells(neighbor)) invalid_triggercells_.emplace(neighbor);
                    }
                }
            }
            for(int trigger_cell=0; trigger_cell<number_trigger_cells_in_wafers_.at(waferfh_type); trigger_cell++)
            {
                for(int zside : {-1,1})
                {
                    HGCalDetId trigger_cell_id(ForwardSubdetector::HGCHEF, zside, layer, 1, trigger_wafer, trigger_cell);
                    if(!validTriggerCellFromCells(trigger_cell_id)) invalid_triggercells_.emplace(trigger_cell_id);
                    for(unsigned neighbor : getNeighborsFromTriggerCell(trigger_cell_id))
                    {
                        auto wafer_itr = wafers_old_to_wafers_.find(HGCalDetId(neighbor).wafer());
                        if(wafer_itr==wafers_old_to_wafers_.end()) invalid_triggercells_.emplace(neighbor);
                        else if(!validTriggerCellFromCells(neighbor)) invalid_triggercells_.emplace(neighbor);
                    }
                }
            }
        }
    }
    unsigned n_layers_hsc = hscTopology().dddConstants().layers(true);
    unsigned first_layer_hsc = hscTopology().dddConstants().firstLayer();
    for(unsigned layer=first_layer_hsc; layer<=first_layer_hsc+n_layers_hsc; layer++)
    {
        int type =  hscTopology().dddConstants().getTypeTrap(layer);
        for(const auto& module_ncells : number_trigger_cells_in_wafers_sci_)
        {
            unsigned trigger_wafer = module_ncells.first;
            // loop on the trigger cells in each wafer
            for(int trigger_cell=1; trigger_cell<module_ncells.second; trigger_cell++)
            {
                for(int zside : {-1,1})
                {
                    HGCalDetId trigger_cell_id(ForwardSubdetector::HGCHEB, zside, layer, type, trigger_wafer, trigger_cell);
                    if(!validTriggerCellFromCells(trigger_cell_id)) invalid_triggercells_.emplace(trigger_cell_id);
                    for(unsigned neighbor : getNeighborsFromTriggerCell(trigger_cell_id))
                    {
                        if(!validTriggerCellFromCells(neighbor)) invalid_triggercells_.emplace(neighbor);
                    }
                }
            }
        }
    }
}

unsigned 
HGCalTriggerGeometryV9Imp1::
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
HGCalTriggerGeometryV9Imp1::
packCellId(unsigned type, unsigned cell) const
{
    unsigned packed_value = 0;
    packed_value |= ((cell & HGCalDetId::kHGCalCellMask) << HGCalDetId::kHGCalCellOffset);
    packed_value |= ((type & HGCSiliconDetId::kHGCalTypeMask) << HGCSiliconDetId::kHGCalTypeOffset);
    return packed_value;
}

unsigned 
HGCalTriggerGeometryV9Imp1::
packCellId(unsigned type, unsigned cellU, unsigned cellV) const
{
    unsigned packed_value = 0;
    packed_value |= ((cellU & HGCSiliconDetId::kHGCalCellUMask) << HGCSiliconDetId::kHGCalCellUOffset);
    packed_value |= ((cellV & HGCSiliconDetId::kHGCalCellVMask) << HGCSiliconDetId::kHGCalCellVOffset);
    packed_value |= ((type & HGCSiliconDetId::kHGCalTypeMask) << HGCSiliconDetId::kHGCalTypeOffset);
    return packed_value;
}

unsigned 
HGCalTriggerGeometryV9Imp1::
packWaferId(int waferU, int waferV) const
{
    unsigned packed_value = 0;
    unsigned waferUabs = std::abs(waferU); 
    unsigned waferVabs = std::abs(waferV);
    unsigned waferUsign = (waferU >= 0) ? 0 : 1;
    unsigned waferVsign = (waferV >= 0) ? 0 : 1;
    packed_value |= ((waferUabs & HGCSiliconDetId::kHGCalWaferUMask) << HGCSiliconDetId::kHGCalWaferUOffset);
    packed_value |= ((waferUsign & HGCSiliconDetId::kHGCalWaferUSignMask) << HGCSiliconDetId::kHGCalWaferUSignOffset);
    packed_value |= ((waferVabs & HGCSiliconDetId::kHGCalWaferVMask) << HGCSiliconDetId::kHGCalWaferVOffset);
    packed_value |= ((waferVsign & HGCSiliconDetId::kHGCalWaferVSignMask) << HGCSiliconDetId::kHGCalWaferVSignOffset);
    return packed_value;
}

unsigned 
HGCalTriggerGeometryV9Imp1::
packIetaIphi(unsigned ieta, unsigned iphi) const
{
    unsigned packed_value = 0;
    packed_value |= ((iphi & HGCScintillatorDetId::kHGCalPhiMask) << HGCScintillatorDetId::kHGCalPhiOffset);
    packed_value |= ((ieta & HGCScintillatorDetId::kHGCalEtaMask) << HGCScintillatorDetId::kHGCalEtaOffset);
    return packed_value;
}

void
HGCalTriggerGeometryV9Imp1::
unpackWaferCellId(unsigned wafer_cell, unsigned& wafer, unsigned& cell) const
{
    cell =  wafer_cell & HGCalDetId::kHGCalCellMask;
    wafer = (wafer_cell>>HGCalDetId::kHGCalWaferOffset) & HGCalDetId::kHGCalWaferMask;
}

void
HGCalTriggerGeometryV9Imp1::
unpackCellId(unsigned cell, unsigned& cellU, unsigned& cellV) const
{
    cellU =  (cell >> HGCSiliconDetId::kHGCalCellUOffset) & HGCSiliconDetId::kHGCalCellUMask; 
    cellV =  (cell >> HGCSiliconDetId::kHGCalCellVOffset) & HGCSiliconDetId::kHGCalCellVMask; 
}


void
HGCalTriggerGeometryV9Imp1::
unpackWaferId(unsigned wafer, int& waferU, int& waferV) const
{
    unsigned waferUAbs = (wafer >> HGCSiliconDetId::kHGCalWaferUOffset) & HGCSiliconDetId::kHGCalWaferUMask;
    unsigned waferVAbs = (wafer >> HGCSiliconDetId::kHGCalWaferVOffset) & HGCSiliconDetId::kHGCalWaferVMask;
    waferU = ( ((wafer >> HGCSiliconDetId::kHGCalWaferUSignOffset) & HGCSiliconDetId::kHGCalWaferUSignMask) ? -waferUAbs : waferUAbs );
    waferV = ( ((wafer >> HGCSiliconDetId::kHGCalWaferVSignOffset) & HGCSiliconDetId::kHGCalWaferVSignMask) ? -waferVAbs : waferVAbs );
}


void
HGCalTriggerGeometryV9Imp1::
unpackIetaIphi(unsigned ieta_iphi, unsigned& ieta, unsigned& iphi) const
{
    iphi =  (ieta_iphi>>HGCScintillatorDetId::kHGCalPhiOffset) & HGCScintillatorDetId::kHGCalPhiMask;
    ieta = (ieta_iphi>>HGCScintillatorDetId::kHGCalEtaOffset) & HGCScintillatorDetId::kHGCalEtaMask;
}

bool 
HGCalTriggerGeometryV9Imp1::
validTriggerCell(const unsigned trigger_cell_id) const
{
    return invalid_triggercells_.find(trigger_cell_id)==invalid_triggercells_.end();
}

bool 
HGCalTriggerGeometryV9Imp1::
disconnectedModule(const unsigned module_id) const
{
    bool disconnected = false;
    if(disconnected_modules_.find(HGCalDetId(module_id).wafer())!=disconnected_modules_.end()) disconnected = true;
    if(disconnected_layers_.find(layerWithOffset(module_id))!=disconnected_layers_.end()) disconnected = true;
    return disconnected;
}

unsigned 
HGCalTriggerGeometryV9Imp1::
triggerLayer(const unsigned id) const
{
    unsigned layer = layerWithOffset(id);
    if(layer>=trigger_layers_.size()) return 0;
    return trigger_layers_[layer];
}

bool 
HGCalTriggerGeometryV9Imp1::
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
HGCalTriggerGeometryV9Imp1::
validCellId(unsigned subdet, unsigned cell_id) const
{
    bool is_valid = false;
    switch(subdet)
    {
        case ForwardSubdetector::HGCEE:
            is_valid = eeTopology().valid(cell_id);
            break;
        case ForwardSubdetector::HGCHEF:
            is_valid = hsiTopology().valid(cell_id);
            break;
        case ForwardSubdetector::HGCHEB:
            is_valid = hscTopology().valid(cell_id);
            break;
        default:
            is_valid = false;
            break;
    } 
    return is_valid;
}


unsigned 
HGCalTriggerGeometryV9Imp1::
packTriggerCell(unsigned module, unsigned trigger_cell) const
{
    unsigned packed_value = 0;
    packed_value |= ((trigger_cell & HGCalDetId::kHGCalCellMask) << HGCalDetId::kHGCalCellOffset);
    packed_value |= ((module & HGCalDetId::kHGCalWaferMask) << HGCalDetId::kHGCalWaferOffset);
    return packed_value;
}

unsigned 
HGCalTriggerGeometryV9Imp1::
packTriggerCellWithType(unsigned type, unsigned module, unsigned trigger_cell) const
{
    unsigned packed_value = 0;
    packed_value |= ((trigger_cell & HGCalDetId::kHGCalCellMask) << HGCalDetId::kHGCalCellOffset);
    packed_value |= ((module & HGCalDetId::kHGCalWaferMask) << HGCalDetId::kHGCalWaferOffset);
    packed_value |= ((type & HGCalDetId::kHGCalWaferTypeMask) << HGCalDetId::kHGCalWaferTypeOffset);
    return packed_value;
}

int 
HGCalTriggerGeometryV9Imp1::
detIdWaferType(unsigned det, unsigned layer, short waferU, short waferV) const
{
    int wafer_type = 0;
    switch(det)
    {
        case DetId::HGCalEE:
            wafer_type = eeTopology().dddConstants().getTypeHex(layer, waferU, waferV);
            break;
        case DetId::HGCalHSi:
            wafer_type = hsiTopology().dddConstants().getTypeHex(layer, waferU, waferV);
            break;
        default:
            break;
    };
    return wafer_type;
}


unsigned
HGCalTriggerGeometryV9Imp1::
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
            layer = heOffset_ + detid.layer();
            break;
        case ForwardSubdetector::HGCHEB:
            layer = heOffset_ + detid.layer();
            break;
    };
    return layer;
}



DEFINE_EDM_PLUGIN(HGCalTriggerGeometryFactory, 
        HGCalTriggerGeometryV9Imp1,
        "HGCalTriggerGeometryV9Imp1");
