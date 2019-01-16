#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalTriggerDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <regex>


class HGCalTriggerGeometryV9Imp2 : public HGCalTriggerGeometryBase
{
    public:
        HGCalTriggerGeometryV9Imp2(const edm::ParameterSet& conf);

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
        // HSc trigger cell grouping
        unsigned hSc_triggercell_size_ = 2;

        // Disconnected modules and layers
        std::unordered_set<unsigned> disconnected_layers_;
        std::vector<unsigned> trigger_layers_;

        // layer offsets 
        unsigned heOffset_;
        unsigned totalLayers_;

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


HGCalTriggerGeometryV9Imp2::
HGCalTriggerGeometryV9Imp2(const edm::ParameterSet& conf):
    HGCalTriggerGeometryBase(conf),
    hSc_triggercell_size_(conf.getParameter<unsigned>("ScintillatorTriggerCellSize"))
{
    std::vector<unsigned> tmp_vector = conf.getParameter<std::vector<unsigned>>("DisconnectedLayers");
    std::move(tmp_vector.begin(), tmp_vector.end(), std::inserter(disconnected_layers_, disconnected_layers_.end()));
}

void
HGCalTriggerGeometryV9Imp2::
reset()
{
}

void
HGCalTriggerGeometryV9Imp2::
initialize(const edm::ESHandle<CaloGeometry>& calo_geometry)
{
    throw cms::Exception("BadGeometry")
        << "HGCalTriggerGeometryV9Imp2 geometry cannot be initialized with the V7/V8 HGCAL geometry";
}

void
HGCalTriggerGeometryV9Imp2::
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
    trigger_layers_[0] = 0; // layer number 0 doesn't exist
    unsigned trigger_layer = 1;
    for(unsigned layer=1; layer<trigger_layers_.size(); layer++)
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

}

unsigned 
HGCalTriggerGeometryV9Imp2::
getTriggerCellFromCell( const unsigned cell_id ) const
{
    unsigned det = DetId(cell_id).det();
    unsigned trigger_cell_id = 0;
    // Scintillator
    if(det == DetId::HGCalHSc)
    {
        // Very rough mapping from cells to TC
        HGCScintillatorDetId cell_sc_id(cell_id);
        int ieta = ( (cell_sc_id.ietaAbs()-1)/hSc_triggercell_size_+1 )*cell_sc_id.zside();
        int iphi = (cell_sc_id.iphi()-1)/hSc_triggercell_size_+1;
        trigger_cell_id = HGCScintillatorDetId(cell_sc_id.type(), cell_sc_id.layer(), ieta, iphi);
    }
    // Silicon
    else if(det == DetId::HGCalEE || det == DetId::HGCalHSi)
    {
        HGCSiliconDetId cell_si_id(cell_id);
        unsigned subdet = (det==DetId::HGCalEE ? HGCalTriggerSubdetector::HGCalEETrigger : HGCalTriggerSubdetector::HGCalHSiTrigger);
        int layer = cell_si_id.layer();
        int zside = cell_si_id.zside();
        int type =  cell_si_id.type();
        int waferu = cell_si_id.waferU();
        int waferv = cell_si_id.waferV();
        int triggercellu = cell_si_id.triggerCellU();
        int triggercellv = cell_si_id.triggerCellV();
        trigger_cell_id = HGCalTriggerDetId(subdet, zside, type, layer, waferu, waferv, triggercellu, triggercellv);
        if(type!=HGCalTriggerDetId(trigger_cell_id).type())
        {
            std::cerr<<"Different type TC/cell in getTriggerCellFromCell()\n";
            std::cerr<<"   cell "<<type<<", TC "<<HGCalTriggerDetId(trigger_cell_id).type()<<"\n";
            std::cerr<<"   cell ID "<<cell_si_id<<"\n";
            std::cerr<<"   TC ID "<<HGCalTriggerDetId(trigger_cell_id)<<"\n";
        }
    }
    return trigger_cell_id;

}

unsigned 
HGCalTriggerGeometryV9Imp2::
getModuleFromCell( const unsigned cell_id ) const
{
    return getModuleFromTriggerCell(getTriggerCellFromCell(cell_id));
}

unsigned 
HGCalTriggerGeometryV9Imp2::
getModuleFromTriggerCell( const unsigned trigger_cell_id ) const
{
    return 0;
}

HGCalTriggerGeometryBase::geom_set 
HGCalTriggerGeometryV9Imp2::
getCellsFromTriggerCell( const unsigned trigger_cell_id ) const
{
    DetId trigger_cell_det_id(trigger_cell_id);
    unsigned det = trigger_cell_det_id.det();
    geom_set cell_det_ids;
    // Scintillator
    if(det==DetId::HGCalHSc)
    {
        HGCScintillatorDetId trigger_cell_sc_id(trigger_cell_id);
        int ieta0 = (trigger_cell_sc_id.ietaAbs()-1)*hSc_triggercell_size_+1;
        int iphi0 = (trigger_cell_sc_id.iphi()-1)*hSc_triggercell_size_+1;
        for(int ietaAbs=ieta0; ietaAbs<ieta0+(int)hSc_triggercell_size_; ietaAbs++)
        {
            int ieta = ietaAbs*trigger_cell_sc_id.zside();
            for(int iphi=iphi0; iphi<iphi0+(int)hSc_triggercell_size_; iphi++)
            {
                unsigned cell_id = HGCScintillatorDetId(trigger_cell_sc_id.type(), trigger_cell_sc_id.layer(), ieta, iphi);
                if(validCellId(DetId::HGCalHSc, cell_id)) cell_det_ids.emplace(cell_id);
            }
        }
    }
    // Silicon
    else 
    {
        HGCalTriggerDetId trigger_cell_trig_id(trigger_cell_id);
        unsigned subdet = trigger_cell_trig_id.subdet();
        if(subdet == HGCalTriggerSubdetector::HGCalEETrigger || subdet == HGCalTriggerSubdetector::HGCalHSiTrigger)
        {
            DetId::Detector cell_det = (subdet==HGCalTriggerSubdetector::HGCalEETrigger ? DetId::HGCalEE : DetId::HGCalHSi);
            int layer = trigger_cell_trig_id.layer();
            int zside = trigger_cell_trig_id.zside();
            int type =  trigger_cell_trig_id.type();
            int waferu = trigger_cell_trig_id.waferU();
            int waferv = trigger_cell_trig_id.waferV();
            std::vector<int> cellus = trigger_cell_trig_id.cellU();
            std::vector<int> cellvs = trigger_cell_trig_id.cellV();
            for(unsigned ic=0; ic<cellus.size(); ic++)
            {
                HGCSiliconDetId cell_det_id(cell_det, zside, type, layer, waferu, waferv, cellus[ic], cellvs[ic]);
                cell_det_ids.emplace(cell_det_id.rawId());
                if(type!=cell_det_id.type())
                {
                    std::cerr<<"Different type TC/cell in getCellsFromTriggerCell()\n";
                    std::cerr<<"   TC "<<type<<", cell "<<cell_det_id.type()<<"\n";
                }
            }
        }
    }
    return cell_det_ids;
}

HGCalTriggerGeometryBase::geom_set 
HGCalTriggerGeometryV9Imp2::
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
HGCalTriggerGeometryV9Imp2::
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
HGCalTriggerGeometryV9Imp2::
getTriggerCellsFromModule( const unsigned module_id ) const
{
    HGCalDetId module_det_id(module_id);
    geom_set trigger_cell_det_ids;
    unsigned module = module_det_id.wafer();
    // Scintillator
    if(module_det_id.subdetId()==ForwardSubdetector::HGCHEB)
    {
    }
    // Silicon
    else
    {
    }
    return trigger_cell_det_ids;
}

HGCalTriggerGeometryBase::geom_ordered_set 
HGCalTriggerGeometryV9Imp2::
getOrderedTriggerCellsFromModule( const unsigned module_id ) const
{
    HGCalDetId module_det_id(module_id);
    geom_ordered_set trigger_cell_det_ids;
    unsigned module = module_det_id.wafer();
    // Scintillator
    if(module_det_id.subdetId()==ForwardSubdetector::HGCHEB)
    {
    }
    // EE or FH
    else
    {
    }
    return trigger_cell_det_ids;
}



HGCalTriggerGeometryBase::geom_set
HGCalTriggerGeometryV9Imp2::
getNeighborsFromTriggerCell( const unsigned trigger_cell_id ) const
{
    HGCalDetId trigger_cell_det_id(trigger_cell_id);
    geom_set neighbor_detids;
    return neighbor_detids;
}


GlobalPoint 
HGCalTriggerGeometryV9Imp2::
getTriggerCellPosition(const unsigned trigger_cell_det_id) const
{
    unsigned det = DetId(trigger_cell_det_id).det();
    // Position: barycenter of the trigger cell.
    Basic3DVector<float> triggerCellVector(0.,0.,0.);
    const auto cell_ids = getCellsFromTriggerCell(trigger_cell_det_id);
    // Scintillator
    if(det==DetId::HGCalHSc)
    {
        for(const auto& cell : cell_ids)
        {
            HGCScintillatorDetId cellDetId(cell);
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
HGCalTriggerGeometryV9Imp2::
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



unsigned 
HGCalTriggerGeometryV9Imp2::
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
HGCalTriggerGeometryV9Imp2::
packCellId(unsigned type, unsigned cell) const
{
    unsigned packed_value = 0;
    packed_value |= ((cell & HGCalDetId::kHGCalCellMask) << HGCalDetId::kHGCalCellOffset);
    packed_value |= ((type & HGCSiliconDetId::kHGCalTypeMask) << HGCSiliconDetId::kHGCalTypeOffset);
    return packed_value;
}

unsigned 
HGCalTriggerGeometryV9Imp2::
packCellId(unsigned type, unsigned cellU, unsigned cellV) const
{
    unsigned packed_value = 0;
    packed_value |= ((cellU & HGCSiliconDetId::kHGCalCellUMask) << HGCSiliconDetId::kHGCalCellUOffset);
    packed_value |= ((cellV & HGCSiliconDetId::kHGCalCellVMask) << HGCSiliconDetId::kHGCalCellVOffset);
    packed_value |= ((type & HGCSiliconDetId::kHGCalTypeMask) << HGCSiliconDetId::kHGCalTypeOffset);
    return packed_value;
}

unsigned 
HGCalTriggerGeometryV9Imp2::
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
HGCalTriggerGeometryV9Imp2::
packIetaIphi(unsigned ieta, unsigned iphi) const
{
    unsigned packed_value = 0;
    packed_value |= ((iphi & HGCScintillatorDetId::kHGCalPhiMask) << HGCScintillatorDetId::kHGCalPhiOffset);
    packed_value |= ((ieta & HGCScintillatorDetId::kHGCalRadiusMask) << HGCScintillatorDetId::kHGCalRadiusOffset);
    return packed_value;
}

void
HGCalTriggerGeometryV9Imp2::
unpackWaferCellId(unsigned wafer_cell, unsigned& wafer, unsigned& cell) const
{
    cell =  wafer_cell & HGCalDetId::kHGCalCellMask;
    wafer = (wafer_cell>>HGCalDetId::kHGCalWaferOffset) & HGCalDetId::kHGCalWaferMask;
}

void
HGCalTriggerGeometryV9Imp2::
unpackCellId(unsigned cell, unsigned& cellU, unsigned& cellV) const
{
    cellU =  (cell >> HGCSiliconDetId::kHGCalCellUOffset) & HGCSiliconDetId::kHGCalCellUMask; 
    cellV =  (cell >> HGCSiliconDetId::kHGCalCellVOffset) & HGCSiliconDetId::kHGCalCellVMask; 
}


void
HGCalTriggerGeometryV9Imp2::
unpackWaferId(unsigned wafer, int& waferU, int& waferV) const
{
    unsigned waferUAbs = (wafer >> HGCSiliconDetId::kHGCalWaferUOffset) & HGCSiliconDetId::kHGCalWaferUMask;
    unsigned waferVAbs = (wafer >> HGCSiliconDetId::kHGCalWaferVOffset) & HGCSiliconDetId::kHGCalWaferVMask;
    waferU = ( ((wafer >> HGCSiliconDetId::kHGCalWaferUSignOffset) & HGCSiliconDetId::kHGCalWaferUSignMask) ? -waferUAbs : waferUAbs );
    waferV = ( ((wafer >> HGCSiliconDetId::kHGCalWaferVSignOffset) & HGCSiliconDetId::kHGCalWaferVSignMask) ? -waferVAbs : waferVAbs );
}


void
HGCalTriggerGeometryV9Imp2::
unpackIetaIphi(unsigned ieta_iphi, unsigned& ieta, unsigned& iphi) const
{
    iphi =  (ieta_iphi>>HGCScintillatorDetId::kHGCalPhiOffset) & HGCScintillatorDetId::kHGCalPhiMask;
    ieta = (ieta_iphi>>HGCScintillatorDetId::kHGCalRadiusOffset) & HGCScintillatorDetId::kHGCalRadiusMask;
}

bool 
HGCalTriggerGeometryV9Imp2::
validTriggerCell(const unsigned trigger_cell_id) const
{
    return true;
}

bool 
HGCalTriggerGeometryV9Imp2::
disconnectedModule(const unsigned module_id) const
{
    bool disconnected = false;
    return disconnected;
}

unsigned 
HGCalTriggerGeometryV9Imp2::
triggerLayer(const unsigned id) const
{
    unsigned layer = layerWithOffset(id);
    if(layer>=trigger_layers_.size()) return 0;
    return trigger_layers_[layer];
}

bool 
HGCalTriggerGeometryV9Imp2::
validTriggerCellFromCells(const unsigned trigger_cell_id) const
{
    // Check the validity of a trigger cell with the
    // validity of the cells. One valid cell in the 
    // trigger cell is enough to make the trigger cell
    // valid.
    const geom_set cells = getCellsFromTriggerCell(trigger_cell_id);
    bool is_valid = false;
    for(const auto cell_id : cells)
    {
        unsigned det = DetId(cell_id).det();
        is_valid |= validCellId(det, cell_id);
        if(is_valid) break;
    }
    return is_valid;
}

bool
HGCalTriggerGeometryV9Imp2::
validCellId(unsigned subdet, unsigned cell_id) const
{
    bool is_valid = false;
    switch(subdet)
    {
        case DetId::HGCalEE:
            is_valid = eeTopology().valid(cell_id);
            break;
        case DetId::HGCalHSi:
            is_valid = hsiTopology().valid(cell_id);
            break;
        case DetId::HGCalHSc:
            is_valid = hscTopology().valid(cell_id);
            break;
        default:
            is_valid = false;
            break;
    } 
    return is_valid;
}


unsigned 
HGCalTriggerGeometryV9Imp2::
packTriggerCell(unsigned module, unsigned trigger_cell) const
{
    unsigned packed_value = 0;
    packed_value |= ((trigger_cell & HGCalDetId::kHGCalCellMask) << HGCalDetId::kHGCalCellOffset);
    packed_value |= ((module & HGCalDetId::kHGCalWaferMask) << HGCalDetId::kHGCalWaferOffset);
    return packed_value;
}

unsigned 
HGCalTriggerGeometryV9Imp2::
packTriggerCellWithType(unsigned type, unsigned module, unsigned trigger_cell) const
{
    unsigned packed_value = 0;
    packed_value |= ((trigger_cell & HGCalDetId::kHGCalCellMask) << HGCalDetId::kHGCalCellOffset);
    packed_value |= ((module & HGCalDetId::kHGCalWaferMask) << HGCalDetId::kHGCalWaferOffset);
    packed_value |= ((type & HGCalDetId::kHGCalWaferTypeMask) << HGCalDetId::kHGCalWaferTypeOffset);
    return packed_value;
}

int 
HGCalTriggerGeometryV9Imp2::
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
HGCalTriggerGeometryV9Imp2::
layerWithOffset(unsigned id) const
{
    unsigned det = DetId(id).det();
    unsigned subdet = HGCalTriggerDetId(id).subdet();
    unsigned layer = 0;
    if(det==DetId::HGCalTrigger && subdet==HGCalTriggerSubdetector::HGCalEETrigger)
    {
        layer = HGCalTriggerDetId(id).layer();
    }
    else if(det==DetId::HGCalTrigger && subdet==HGCalTriggerSubdetector::HGCalHSiTrigger)
    {
        layer = heOffset_ + HGCalTriggerDetId(id).layer();
    }
    else if(det==DetId::HGCalHSc)
    {
        layer = heOffset_ + HGCScintillatorDetId(id).layer();
    }
    return layer;
}



DEFINE_EDM_PLUGIN(HGCalTriggerGeometryFactory, 
        HGCalTriggerGeometryV9Imp2,
        "HGCalTriggerGeometryV9Imp2");
