#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "DataFormats/ForwardDetId/interface/HGCTriggerDetId.h"

#include <vector>
#include <iostream>
#include <fstream>


class HGCalTriggerGeometryHexImp1 : public HGCalTriggerGeometryBase
{
    public:
        HGCalTriggerGeometryHexImp1(const edm::ParameterSet& conf);

        virtual void initialize(const es_info& ) override final;

    private:
        edm::FileInPath l1tCellsMapping_;
        edm::FileInPath l1tModulesMapping_;
};


/*****************************************************************/
HGCalTriggerGeometryHexImp1::HGCalTriggerGeometryHexImp1(const edm::ParameterSet& conf):
    HGCalTriggerGeometryBase(conf),
    l1tCellsMapping_(conf.getParameter<edm::FileInPath>("L1TCellsMapping")),
    l1tModulesMapping_(conf.getParameter<edm::FileInPath>("L1TModulesMapping"))
/*****************************************************************/
{
}


/*****************************************************************/
void HGCalTriggerGeometryHexImp1::initialize(const es_info& esInfo)
/*****************************************************************/
{
    // FIXME: !!!Only for HGCEE for the moment!!!
    edm::LogWarning("HGCalTriggerGeometry") << "WARNING: This HGCal trigger geometry is incomplete.\n"\
                                            << "WARNING: Only the EE part is covered.\n"\
                                            << "WARNING: There is no neighbor information.\n";
    //
    // read module mapping file
    std::map<short, short> wafer_to_module;
    std::ifstream l1tModulesMappingStream(l1tModulesMapping_.fullPath());
    if(!l1tModulesMappingStream.is_open()) edm::LogError("HGCalTriggerGeometry") << "Cannot open L1TModulesMapping file\n";
    short wafer   = 0;
    short module  = 0;
    for(; l1tModulesMappingStream>>wafer>>module; )
    {
        wafer_to_module[wafer] = module;
    }
    l1tModulesMappingStream.close();
    //
    // read trigger cell mapping file
    std::map<std::pair<short,short>, short> cells_to_trigger_cells;
    std::ifstream l1tCellsMappingStream(l1tCellsMapping_.fullPath());
    if(!l1tCellsMappingStream.is_open()) edm::LogError("HGCalTriggerGeometry") << "Cannot open L1TCellsMapping file\n";
    short wafertype   = 0;
    short cell        = 0;
    short triggercell = 0;
    for(; l1tCellsMappingStream>>wafertype>>cell>>triggercell; )
    {
        cells_to_trigger_cells[std::make_pair((wafertype?1:-1),cell)] = triggercell;
    }
    //if(!l1tCellsMappingStream.eof()) edm::LogWarning("HGCalTriggerGeometry") << "Error reading L1TCellsMapping'"<<layer<<" "<<cell<<" "<<triggercell<<" "<<subsector<<"' \n";
    l1tCellsMappingStream.close();
    for(const auto& id : esInfo.geom_ee->getValidGeomDetIds())
    {
        if(id.rawId()==0) continue;
        HGCalDetId waferid(id); 
        short module      = wafer_to_module[waferid.wafer()];
        int nCells = esInfo.topo_ee->dddConstants().numberCellsHexagon(waferid.wafer());
        for(int i=0;i<nCells;i++)
        {
            short triggercell = cells_to_trigger_cells[std::make_pair(waferid.waferType(),i)];
            // Fill cell -> trigger cell mapping
            HGCalDetId cellid(ForwardSubdetector(waferid.subdetId()), waferid.zside(), waferid.layer(), waferid.waferType(), waferid.wafer(), i); 
            HGCalDetId triggerDetid(ForwardSubdetector(waferid.subdetId()), waferid.zside(), waferid.layer(), waferid.waferType(), waferid.wafer(), triggercell); 
            cells_to_trigger_cells_.insert( std::make_pair(cellid, triggerDetid) );
            // Fill trigger cell -> module mapping
            HGCalDetId moduleDetid(ForwardSubdetector(waferid.subdetId()), waferid.zside(), waferid.layer(), waferid.waferType(), module, HGCalDetId::kHGCalCellMask);
            trigger_cells_to_modules_.insert( std::make_pair(triggerDetid, moduleDetid) ); // do nothing if trigger cell has already been inserted
        }
    }
    //
    // Build trigger cells and fill map
    typedef HGCalTriggerGeometry::TriggerCell::list_type list_cells;
    // make list of cells in trigger cells
    std::map<unsigned, list_cells> trigger_cells_to_cells;
    for(const auto& cell_triggercell : cells_to_trigger_cells_)
    {
        unsigned cell = cell_triggercell.first;
        unsigned triggercell = cell_triggercell.second;
        trigger_cells_to_cells.insert( std::make_pair(triggercell, list_cells()) );
        trigger_cells_to_cells.at(triggercell).insert(cell);
    }
    for(const auto& triggercell_cells : trigger_cells_to_cells)
    {
        unsigned triggercellId = triggercell_cells.first;
        list_cells cellIds = triggercell_cells.second;
        // Position: for the moment, barycenter of the trigger cell.
        Basic3DVector<float> triggercellVector(0.,0.,0.);
        for(const auto& cell : cellIds)
        {
            HGCalDetId cellId(cell);
            triggercellVector += esInfo.geom_ee->getPosition(cellId).basicVector();
        }
        GlobalPoint triggercellPoint( triggercellVector/cellIds.size() );
        const auto& tc2mItr = trigger_cells_to_modules_.find(triggercellId);
        unsigned moduleId = (tc2mItr!=trigger_cells_to_modules_.end() ? tc2mItr->second : 0); // 0 if the trigger cell doesn't belong to a module
        //unsigned moduleId = trigger_cells_to_modules_.at(triggercellId);
        // FIXME: empty neighbours
        std::unique_ptr<const HGCalTriggerGeometry::TriggerCell> triggercellPtr(new HGCalTriggerGeometry::TriggerCell(triggercellId, moduleId, triggercellPoint, list_cells(), cellIds));
        trigger_cells_.insert( std::make_pair(triggercellId, std::move(triggercellPtr)) );
    }
    //
    // Build modules and fill map
    typedef HGCalTriggerGeometry::Module::list_type list_triggercells;
    typedef HGCalTriggerGeometry::Module::tc_map_type tc_map_to_cells;
    // make list of trigger cells in modules
    std::map<unsigned, list_triggercells> modules_to_trigger_cells;
    for(const auto& triggercell_module : trigger_cells_to_modules_)
    {
        unsigned triggercell = triggercell_module.first;
        unsigned module      = triggercell_module.second;
        modules_to_trigger_cells.insert( std::make_pair(module, list_triggercells()) );
        modules_to_trigger_cells.at(module).insert(triggercell);
    }
    for(const auto& module_triggercell : modules_to_trigger_cells)
    {
        unsigned moduleId = module_triggercell.first;
        list_triggercells triggercellIds = module_triggercell.second;
        tc_map_to_cells cellsInTriggerCells;
        // Position: for the moment, barycenter of the module, from trigger cell positions
        Basic3DVector<float> moduleVector(0.,0.,0.);
        for(const auto& triggercell : triggercellIds)
        {
            const auto& cells_in_tc = trigger_cells_to_cells.at(triggercell);
            for( const unsigned cell : cells_in_tc ) 
            {
              cellsInTriggerCells.emplace(triggercell,cell);
            }
            moduleVector += trigger_cells_.at(triggercell)->position().basicVector();
        }
        GlobalPoint modulePoint( moduleVector/triggercellIds.size() );
        // FIXME: empty neighbours
        std::unique_ptr<const HGCalTriggerGeometry::Module> modulePtr(new HGCalTriggerGeometry::Module(moduleId, modulePoint, list_triggercells(), triggercellIds, cellsInTriggerCells));
        modules_.insert( std::make_pair(moduleId, std::move(modulePtr)) );
    }
}


DEFINE_EDM_PLUGIN(HGCalTriggerGeometryFactory, 
        HGCalTriggerGeometryHexImp1,
        "HGCalTriggerGeometryHexImp1");
