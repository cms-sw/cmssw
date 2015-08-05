#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

#include <vector>
#include <iostream>
#include <fstream>

class HGCalTriggerGeometryImp1 : public HGCalTriggerGeometryBase
{
    public:
        HGCalTriggerGeometryImp1(const edm::ParameterSet& conf);

        virtual void initialize(const es_info& ) override final;

    private:
        edm::FileInPath l1tMapping_;
};


/*****************************************************************/
HGCalTriggerGeometryImp1::HGCalTriggerGeometryImp1(const edm::ParameterSet& conf):
    HGCalTriggerGeometryBase(conf),
    l1tMapping_(conf.getParameter<edm::FileInPath>("L1TMapping"))
/*****************************************************************/
{
}


/*****************************************************************/
void HGCalTriggerGeometryImp1::initialize(const es_info& esInfo)
/*****************************************************************/
{
    // FIXME: !!!Only for HGCEE for the moment!!!
    edm::LogWarning("HGCalTriggerGeometry") << "WARNING: This HGCal trigger geometry is incomplete.\n"\
                                            << "WARNING: Only the EE part is covered.\n"\
                                            << "WARNING: Dummy hardcoded modules are used.\n"\
                                            << "WARNING: There is no neighbor information.\n";

    std::ifstream l1tMappingStream(l1tMapping_.fullPath());
    if(!l1tMappingStream.is_open()) edm::LogError("HGCalTriggerGeometry") << "Cannot open L1TMapping file\n";
    //
    // read mapping file
    short layer       = 0;
    short cell        = 0;
    short triggercell = 0;
    short subsector   = 0;
    short module      = 0;
    for(; l1tMappingStream>>layer>>cell>>triggercell>>subsector; )
    {
        layer++; //FIXME: currently the first layer in the mapping file has index=0, should be changed to 1
        if(layer>30 || layer<=0) 
        {
            edm::LogWarning("HGCalTriggerGeometry") << "Bad layer index in L1TMapping\n"; 
            continue; 
        }
        // FIXME: dummy module ID (mapping not yet available)
        module = triggercell/64;
        // Loop on all sectors
        // FIXME:  Number of sectors in each zside should not be hardcoded
        for(unsigned z=0; z<=1; z++)
        {
            int zside = (z==0 ? -1 : 1);
            for(unsigned sector=1; sector<=18; sector++)
            {
                HGCEEDetId detid(HGCEE, zside, layer, sector, subsector, cell); 
                GlobalPoint point = esInfo.geom_ee->getPosition(detid);
                if(point.x()==0 && point.y()==0) std::cout<<" Cell "<<detid()<<" layer="<<layer<<" sector="<<sector<<" subsec="<<subsector<<" cell="<<cell<<"\n";
                // FIXME: Use temporarily HGCEEDetId to compute trigger cell id
                HGCEEDetId triggerDetid(HGCEE, zside, layer, sector, 1, triggercell); // Dummy subsector
                const auto& ret = cells_to_trigger_cells_.insert( std::make_pair(detid(), triggerDetid()) );
                if(!ret.second) edm::LogWarning("HGCalTriggerGeometry") << "Duplicate cell in L1TMapping\n";
                // FIXME: Use temporarily HGCEEDetId to compute module id
                HGCEEDetId moduleDetid(HGCEE, zside, layer, sector, 1, module); // Dummy subsector
                // Only the first occurence of the trigger cell is filled (one trigger cell is not
                // shared between different modules)
                trigger_cells_to_modules_.insert( std::make_pair(triggerDetid(), moduleDetid()) );
            }
        }
    }
    l1tMappingStream.close();
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
            HGCEEDetId cellId(cell);
            triggercellVector += esInfo.geom_ee->getPosition(cellId).basicVector();
        }
        GlobalPoint triggercellPoint( triggercellVector/cellIds.size() );
        unsigned moduleId = trigger_cells_to_modules_.at(triggercellId);
        // FIXME: empty neighbours
        std::unique_ptr<const HGCalTriggerGeometry::TriggerCell> triggercellPtr(new HGCalTriggerGeometry::TriggerCell(triggercellId, moduleId, triggercellPoint, list_cells(), cellIds));
        trigger_cells_.insert( std::make_pair(triggercellId, std::move(triggercellPtr)) );
    }
    //
    // Build modules and fill map
    typedef HGCalTriggerGeometry::Module::list_type list_triggercells;
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
        // Position: for the moment, barycenter of the module, from trigger cell positions
        Basic3DVector<float> moduleVector(0.,0.,0.);
        for(const auto& triggercell : triggercellIds)
        {
            moduleVector += trigger_cells_.at(triggercell)->position().basicVector();
        }
        GlobalPoint modulePoint( moduleVector/triggercellIds.size() );
        // FIXME: empty neighbours
        std::unique_ptr<const HGCalTriggerGeometry::Module> modulePtr(new HGCalTriggerGeometry::Module(moduleId, modulePoint, list_triggercells(), triggercellIds));
        modules_.insert( std::make_pair(moduleId, std::move(modulePtr)) );
    }
}


DEFINE_EDM_PLUGIN(HGCalTriggerGeometryFactory, 
        HGCalTriggerGeometryImp1,
        "HGCalTriggerGeometryImp1");
