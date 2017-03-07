#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <regex>


class HGCalTriggerGeometryHexImp2 : public HGCalTriggerGeometryBase
{
    public:
        HGCalTriggerGeometryHexImp2(const edm::ParameterSet& conf);

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
        edm::FileInPath l1tCellNeighborMapping_;
        edm::FileInPath l1tWaferNeighborMapping_;
        edm::FileInPath l1tModulesMapping_;

        es_info es_info_;

        // module related maps
        std::unordered_map<short, short> wafer_to_module_ee_;
        std::unordered_map<short, short> wafer_to_module_fh_;
        std::unordered_multimap<short, short> module_to_wafers_ee_;
        std::unordered_multimap<short, short> module_to_wafers_fh_;

        // trigger cell related maps
        std::map<std::pair<short,short>, short> cells_to_trigger_cells_; // FIXME: something else than map<pair,short>?
        std::multimap<std::pair<short,short>, short> trigger_cells_to_cells_;// FIXME: something else than map<pair,short>?
        std::unordered_map<short, short> number_trigger_cells_in_wafers_; // the map key is the wafer type
        std::unordered_map<short, short> number_cells_in_wafers_; // the map key is the wafer type

        // neighbor related maps
        // trigger cell neighbors:
        // - The key includes the trigger cell id and the wafer configuration.
        // The wafer configuration is a 7 bits word encoding the type 
        // (small or large cells) of the wafer containing the trigger cell
        // (central wafer) as well as the type of the 6 surrounding wafers
        // - The value is a set of (wafer_idx, trigger_cell_id)
        // wafer_idx is a number between 0 and 7. 0=central wafer, 1..7=surrounding
        // wafers 
        std::unordered_map<int, std::set<std::pair<short,short>>> trigger_cell_neighbors_;
        // wafer neighbors:
        // List of the 6 surrounding neighbors around each wafer
        std::unordered_map<short, std::vector<short>> wafer_neighbors_ee_;
        std::unordered_map<short, std::vector<short>> wafer_neighbors_fh_;

        void fillMaps(const es_info&);
        void fillNeighborMaps(const es_info&);
        unsigned packTriggerCell(unsigned, const std::vector<int>&) const;
};


HGCalTriggerGeometryHexImp2::
HGCalTriggerGeometryHexImp2(const edm::ParameterSet& conf):
    HGCalTriggerGeometryBase(conf),
    l1tCellsMapping_(conf.getParameter<edm::FileInPath>("L1TCellsMapping")),
    l1tCellNeighborMapping_(conf.getParameter<edm::FileInPath>("L1TCellNeighborMapping")),
    l1tWaferNeighborMapping_(conf.getParameter<edm::FileInPath>("L1TWaferNeighborMapping")),
    l1tModulesMapping_(conf.getParameter<edm::FileInPath>("L1TModulesMapping"))
{
}

void
HGCalTriggerGeometryHexImp2::
reset()
{
    wafer_to_module_ee_.clear();
    wafer_to_module_fh_.clear();
    module_to_wafers_ee_.clear();
    module_to_wafers_fh_.clear();
    cells_to_trigger_cells_.clear();
    trigger_cells_to_cells_.clear();
    number_trigger_cells_in_wafers_.clear();
    number_cells_in_wafers_.clear();
}

void
HGCalTriggerGeometryHexImp2::
initialize(const es_info& esInfo)
{
    edm::LogWarning("HGCalTriggerGeometry") << "WARNING: This HGCal trigger geometry is incomplete.\n"\
                                            << "WARNING: There is no neighbor information.\n";
    es_info_ = esInfo;
    fillMaps(esInfo);
    fillNeighborMaps(esInfo);

}

unsigned 
HGCalTriggerGeometryHexImp2::
getTriggerCellFromCell( const unsigned cell_id ) const
{
    HGCalDetId cell_det_id(cell_id);
    int wafer_type = cell_det_id.waferType();
    unsigned cell = cell_det_id.cell();
    unsigned trigger_cell = 0;
    try
    {
        // FIXME: better way to do this cell->TC mapping?
        trigger_cell = cells_to_trigger_cells_.at(std::make_pair(wafer_type,cell));
    }
    catch (const std::out_of_range& e) {
        throw cms::Exception("BadGeometry")
            << "HGCalTriggerGeometry: HGCal  cell " << cell << " is not mapped to any trigger cell for the wafer type " << wafer_type
            << ". The trigger cell mapping should be modified.\n";
    }
    return HGCalDetId((ForwardSubdetector)cell_det_id.subdetId(), cell_det_id.zside(), cell_det_id.layer(), cell_det_id.waferType(), cell_det_id.wafer(), trigger_cell).rawId();
}

unsigned 
HGCalTriggerGeometryHexImp2::
getModuleFromCell( const unsigned cell_id ) const
{
    HGCalDetId cell_det_id(cell_id);
    unsigned wafer = cell_det_id.wafer();
    unsigned subdet = cell_det_id.subdetId();
    unsigned module = 0;
    try
    {
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
    }
    catch (const std::out_of_range& e) {
        throw cms::Exception("BadGeometry")
            << "HGCalTriggerGeometry: Wafer " << wafer << " is not mapped to any trigger module for subdetector " << subdet
            << ". The module mapping should be modified. See https://twiki.cern.ch/twiki/bin/viewauth/CMS/HGCALTriggerPrimitivesSimulation#Trigger_geometry for details.\n";
    }
    return HGCalDetId((ForwardSubdetector)cell_det_id.subdetId(), cell_det_id.zside(), cell_det_id.layer(), cell_det_id.waferType(), module, HGCalDetId::kHGCalCellMask).rawId();
}

unsigned 
HGCalTriggerGeometryHexImp2::
getModuleFromTriggerCell( const unsigned trigger_cell_id ) const
{
    HGCalDetId trigger_cell_det_id(trigger_cell_id);
    unsigned wafer = trigger_cell_det_id.wafer();
    unsigned subdet = trigger_cell_det_id.subdetId();
    unsigned module = 0;
    try 
    {
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
        } 
    }
    catch (const std::out_of_range& e) {
        throw cms::Exception("BadGeometry")
            << "HGCalTriggerGeometry: Wafer " << wafer << " is not mapped to any trigger module for subdetector " << subdet
            << ". The module mapping should be modified. See https://twiki.cern.ch/twiki/bin/viewauth/CMS/HGCALTriggerPrimitivesSimulation#Trigger_geometry for details.\n";
    };
    return HGCalDetId((ForwardSubdetector)trigger_cell_det_id.subdetId(), trigger_cell_det_id.zside(), trigger_cell_det_id.layer(), trigger_cell_det_id.waferType(), module, HGCalDetId::kHGCalCellMask).rawId();
}

HGCalTriggerGeometryBase::geom_set 
HGCalTriggerGeometryHexImp2::
getCellsFromTriggerCell( const unsigned trigger_cell_id ) const
{
    HGCalDetId trigger_cell_det_id(trigger_cell_id);
    int wafer_type = trigger_cell_det_id.waferType();
    unsigned trigger_cell = trigger_cell_det_id.cell();
    // FIXME: better way to do this TC->cell mapping?
    const auto& cell_range = trigger_cells_to_cells_.equal_range(std::make_pair(wafer_type,trigger_cell));
    geom_set cell_det_ids;
    for(auto tc_c_itr=cell_range.first; tc_c_itr!=cell_range.second; tc_c_itr++)
    {
        cell_det_ids.emplace(HGCalDetId((ForwardSubdetector)trigger_cell_det_id.subdetId(), trigger_cell_det_id.zside(), trigger_cell_det_id.layer(), trigger_cell_det_id.waferType(), trigger_cell_det_id.wafer(), tc_c_itr->second).rawId());
    }
    return cell_det_ids;
}

HGCalTriggerGeometryBase::geom_set 
HGCalTriggerGeometryHexImp2::
getCellsFromModule( const unsigned module_id ) const
{

    HGCalDetId module_det_id(module_id);
    unsigned module = module_det_id.wafer();
    int wafer_type = module_det_id.waferType();
    unsigned subdet = module_det_id.subdetId();
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
            cell_det_ids.emplace(HGCalDetId((ForwardSubdetector)module_det_id.subdetId(), module_det_id.zside(), module_det_id.layer(), module_det_id.waferType(), wafer_itr->second, cell).rawId());
        }
    }
    return cell_det_ids;
}

HGCalTriggerGeometryBase::geom_ordered_set 
HGCalTriggerGeometryHexImp2::
getOrderedCellsFromModule( const unsigned module_id ) const
{
    HGCalDetId module_det_id(module_id);
    unsigned module = module_det_id.wafer();
    int wafer_type = module_det_id.waferType();
    unsigned subdet = module_det_id.subdetId();
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
            return geom_ordered_set();
    };
    geom_ordered_set cell_det_ids;
    for(auto wafer_itr=wafer_itrs.first; wafer_itr!=wafer_itrs.second; wafer_itr++)
    {
        // loop on the cells in each wafer
        for(int cell=0; cell<number_cells_in_wafers_.at(wafer_type); cell++)
        {
            cell_det_ids.emplace(HGCalDetId((ForwardSubdetector)module_det_id.subdetId(), module_det_id.zside(), module_det_id.layer(), module_det_id.waferType(), wafer_itr->second, cell).rawId());
        }
    }
    return cell_det_ids;
}

HGCalTriggerGeometryBase::geom_set 
HGCalTriggerGeometryHexImp2::
getTriggerCellsFromModule( const unsigned module_id ) const
{
    HGCalDetId module_det_id(module_id);
    unsigned module = module_det_id.wafer();
    int wafer_type = module_det_id.waferType();
    unsigned subdet = module_det_id.subdetId();
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
            trigger_cell_det_ids.emplace(HGCalDetId((ForwardSubdetector)module_det_id.subdetId(), module_det_id.zside(), module_det_id.layer(), module_det_id.waferType(), wafer_itr->second, trigger_cell).rawId());
        }
    }
    return trigger_cell_det_ids;
}

HGCalTriggerGeometryBase::geom_ordered_set 
HGCalTriggerGeometryHexImp2::
getOrderedTriggerCellsFromModule( const unsigned module_id ) const
{
    HGCalDetId module_det_id(module_id);
    unsigned module = module_det_id.wafer();
    int wafer_type = module_det_id.waferType();
    unsigned subdet = module_det_id.subdetId();
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
            return geom_ordered_set();
    };
    geom_ordered_set trigger_cell_det_ids;
    // loop on the wafers included in the module
    for(auto wafer_itr=wafer_itrs.first; wafer_itr!=wafer_itrs.second; wafer_itr++)
    {
        // loop on the trigger cells in each wafer
        for(int trigger_cell=0; trigger_cell<number_trigger_cells_in_wafers_.at(wafer_type); trigger_cell++)
        {
            trigger_cell_det_ids.emplace(HGCalDetId((ForwardSubdetector)module_det_id.subdetId(), module_det_id.zside(), module_det_id.layer(), module_det_id.waferType(), wafer_itr->second, trigger_cell).rawId());
        }
    }
    return trigger_cell_det_ids;
}



HGCalTriggerGeometryBase::geom_set
HGCalTriggerGeometryHexImp2::
getNeighborsFromTriggerCell( const unsigned trigger_cell_id ) const
{
    HGCalDetId trigger_cell_det_id(trigger_cell_id);
    unsigned wafer = trigger_cell_det_id.wafer();
    int wafer_type = trigger_cell_det_id.waferType();
    unsigned subdet = trigger_cell_det_id.subdetId();
    unsigned trigger_cell = trigger_cell_det_id.cell();
    // Retrieve surrounding wafers (around the wafer containing
    // the trigger cell)
    const std::vector<short>* surrounding_wafers = nullptr;
    try
    {
        switch(subdet)
        {
            case ForwardSubdetector::HGCEE:
                surrounding_wafers = &wafer_neighbors_ee_.at(wafer);
                break;
            case ForwardSubdetector::HGCHEF:
                surrounding_wafers = &wafer_neighbors_fh_.at(wafer);
                break;
            default:
                edm::LogError("HGCalTriggerGeometry") << "Unknown wafer neighbours for subdet "<<subdet<<"\n";
                return geom_set();
        } 
    }
    catch (const std::out_of_range& e) {
        throw cms::Exception("BadGeometry")
            << "HGCalTriggerGeometry: Neighbors are not defined for wafer " << wafer << " in subdetector " << subdet
            << ". The wafer neighbor mapping should be modified. \n";
    };
    if(!surrounding_wafers)
    {
        throw cms::Exception("BadGeometry")
            << "HGCalTriggerGeometry: Neighbors are not defined for wafer " << wafer << " in subdetector " << subdet
            << ". The wafer neighbor mapping should be modified. \n";
    }
    // Find the types of the surrounding wafers
    std::vector<int> types;
    types.reserve(surrounding_wafers->size()+1); // includes the central wafer -> +1
    types.emplace_back(wafer_type);
    for(const auto w : *surrounding_wafers)
    {
        types.emplace_back(es_info_.topo_ee->dddConstants().waferTypeT(w));
    }
    // retrieve neighbors
    unsigned trigger_cell_key = packTriggerCell(trigger_cell, types);
    geom_set neighbor_detids;
    try 
    {
        const auto& neighbors = trigger_cell_neighbors_.at(trigger_cell_key);
        // create HGCalDetId of neighbors and check their validity
        neighbor_detids.reserve(neighbors.size());
        for(const auto& wafer_tc : neighbors)
        {
            unsigned neighbor_wafer = surrounding_wafers->at(wafer_tc.first);
            int type = types.at(wafer_tc.first);
            HGCalDetId neighbor_det_id((ForwardSubdetector)trigger_cell_det_id.subdetId(), trigger_cell_det_id.zside(), trigger_cell_det_id.layer(), type, neighbor_wafer, wafer_tc.second);
            if(validTriggerCell(neighbor_det_id.rawId()))
            {
                neighbor_detids.emplace(neighbor_det_id.rawId());
            }
        }
    }
    catch (const std::out_of_range& e) {
        throw cms::Exception("BadGeometry")
            << "HGCalTriggerGeometry: Neighbors are not defined for trigger cell " << trigger_cell << " with  wafer configuration "
            << "0x" << std::hex << (trigger_cell_key >> 8) << ". The trigger cell neighbor mapping should be modified. \n";
    }
    return neighbor_detids;
}


GlobalPoint 
HGCalTriggerGeometryHexImp2::
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
HGCalTriggerGeometryHexImp2::
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

void 
HGCalTriggerGeometryHexImp2::
fillNeighborMaps(const es_info& esInfo)
{
    // Fill trigger neighbor map
    std::ifstream l1tCellNeighborMappingStream(l1tCellNeighborMapping_.fullPath());
    if(!l1tCellNeighborMappingStream.is_open()) edm::LogError("HGCalTriggerGeometry") << "Cannot open L1TCellNeighborMapping file\n";
    for(std::array<char,512> buffer; l1tCellNeighborMappingStream.getline(&buffer[0], 512); )
    {
        std::string line(&buffer[0]);
        // Extract keys consisting of the wafer configuration
        // and of the trigger cell id
        // Match patterns (X,Y) 
        // where X is a set of 7 bits
        // and Y is a number with less than 4 digits
        std::regex key_regex("\\(\\s*[01]{7}\\s*,\\s*\\d{1,3}\\s*\\)");
        std::vector<std::string> key_tokens {
            std::sregex_token_iterator(line.begin(), line.end(), key_regex), {}
        };
        if(key_tokens.size()!=1)
        {
            throw cms::Exception("BadGeometry")
                << "Syntax error in the L1TCellNeighborMapping:\n"
                << "  Cannot find the trigger cell key in line:\n"
                << "  '"<<&buffer[0]<<"'\n";
        }
        std::regex digits_regex("([01]{7})|(\\d{1,3})");
        std::vector<std::string>  pair {
            std::sregex_token_iterator(key_tokens[0].begin(), key_tokens[0].end(), digits_regex), {}
        };
        // get cell id and wafer configuration
        int trigger_cell = std::stoi(pair[1]);
        std::vector<int> wafer_types;
        wafer_types.reserve(pair[0].size());
        for(const auto c : pair[0]) wafer_types.emplace_back(std::stoi(std::string(&c)));
        unsigned map_key = packTriggerCell(trigger_cell, wafer_types);
        // Extract neighbors
        // Match patterns (X,Y) 
        // where X is a number with less than 4 digits
        // and Y is one single digit (the neighbor wafer, between 0 and 6)
        std::regex neighbors_regex("\\(\\s*\\d{1,3}\\s*,\\s*\\d\\s*\\)");
        std::vector<std::string> neighbors_tokens {
            std::sregex_token_iterator(line.begin(), line.end(), neighbors_regex), {}
        };
        if(neighbors_tokens.size()==0)
        {
            throw cms::Exception("BadGeometry")
                << "Syntax error in the L1TCellNeighborMapping:\n"
                << "  Cannot find any neighbor in line:\n"
                << "  '"<<&buffer[0]<<"'\n";
        }
        auto itr_insert = trigger_cell_neighbors_.emplace(map_key, std::set<std::pair<short,short>>());
        for(const auto& neighbor : neighbors_tokens)
        {
            std::vector<std::string>  pair_neighbor {
                std::sregex_token_iterator(neighbor.begin(), neighbor.end(), digits_regex), {}
            };
            short neighbor_wafer(std::stoi(pair_neighbor[1]));
            short neighbor_cell(std::stoi(pair_neighbor[0]));
            itr_insert.first->second.emplace(neighbor_wafer, neighbor_cell);
        }
    }
    if(!l1tCellNeighborMappingStream.eof()) edm::LogWarning("HGCalTriggerGeometry") << "Error reading L1TCellNeighborMapping'\n";
    l1tCellNeighborMappingStream.close();

    // Fill wafer neighbor map
    std::ifstream l1tWaferNeighborMappingStream(l1tWaferNeighborMapping_.fullPath());
    if(!l1tWaferNeighborMappingStream.is_open()) edm::LogError("HGCalTriggerGeometry") << "Cannot open L1TWaferNeighborMapping file\n";
    for(std::array<char,512> buffer; l1tWaferNeighborMappingStream.getline(&buffer[0], 512); )
    {
        std::string line(&buffer[0]);
        // split line using spaces as delimiter
        std::regex delimiter("\\s+");
        std::vector<std::string>  tokens {
            std::sregex_token_iterator(line.begin(), line.end(), delimiter, -1), {}
        };
        if(tokens.size()!=8)
        {
            throw cms::Exception("BadGeometry")
                << "Syntax error in the L1TWaferNeighborMapping in line:\n"
                << "  '"<<&buffer[0]<<"'\n"
                << "  A line should be composed of 8 integers separated by spaces:\n"
                << "  subdet waferid neighbor1 neighbor2 neighbor3 neighbor4 neighbor5 neighbor6\n";
        }
        short subdet(std::stoi(tokens[0]));
        short wafer(std::stoi(tokens[1]));

        std::unordered_map<short, std::vector<short>>* wafer_neighbors;
        switch(subdet)
        {
            case ForwardSubdetector::HGCEE:
                wafer_neighbors = &wafer_neighbors_ee_;
                break;
            case ForwardSubdetector::HGCHEF:
                wafer_neighbors = &wafer_neighbors_fh_;
                break;
            default:
                throw cms::Exception("BadGeometry")
                    << "Unknown subdet " << subdet << " in L1TWaferNeighborMapping:\n"
                    << "  '"<<&buffer[0]<<"'\n";
        };
        auto wafer_itr = wafer_neighbors->emplace(wafer, std::vector<short>());
        for(auto neighbor_itr=tokens.cbegin()+2; neighbor_itr!=tokens.cend(); ++neighbor_itr)
        {
            wafer_itr.first->second.emplace_back(std::stoi(*neighbor_itr));
        }
    }
    std::cout<<"EE\n";
    for(const auto& wafer_neighbors : wafer_neighbors_ee_)
    {
        std::cout<<wafer_neighbors.first<<"\n";
        for(const auto& neighbor : wafer_neighbors.second)
        {
            std::cout<<neighbor<<" ";
        }
        std::cout<<"\n";
    }
    std::cout<<"FH\n";
    for(const auto& wafer_neighbors : wafer_neighbors_fh_)
    {
        std::cout<<wafer_neighbors.first<<"\n";
        for(const auto& neighbor : wafer_neighbors.second)
        {
            std::cout<<neighbor<<" ";
        }
        std::cout<<"\n";
    }
}

unsigned 
HGCalTriggerGeometryHexImp2::
packTriggerCell(unsigned trigger_cell, const std::vector<int>& wafer_types) const
{
    unsigned packed_value = trigger_cell;
    for(unsigned i=0; i<wafer_types.size(); i++)
    {
        // trigger cell id on 8 bits
        if(wafer_types.at(i)==1) packed_value += (0x1<<(8+i));
    }
    return packed_value;
}

bool 
HGCalTriggerGeometryHexImp2::
validTriggerCell(const unsigned trigger_cell_id) const
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
        switch(subdet)
        {
            case ForwardSubdetector::HGCEE:
                is_valid |= es_info_.topo_ee->valid(cell_id);
                break;
            case ForwardSubdetector::HGCHEF:
                is_valid |= es_info_.topo_fh->valid(cell_id);
                break;
            default:
                is_valid = false;
                break;
        } 
        if(is_valid) break;
    }
    return is_valid;
}




DEFINE_EDM_PLUGIN(HGCalTriggerGeometryFactory, 
        HGCalTriggerGeometryHexImp2,
        "HGCalTriggerGeometryHexImp2");
