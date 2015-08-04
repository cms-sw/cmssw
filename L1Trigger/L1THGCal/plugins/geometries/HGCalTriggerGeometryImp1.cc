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
    // FIXME: Only for HGCEE for the moment
    std::ifstream l1tMappingStream(l1tMapping_.fullPath());
    if(!l1tMappingStream.is_open()) edm::LogError("HGCalTriggerGeometry") << "Cannot open L1TMapping file\n";

    // read mapping file
    short layer       = 0;
    short cell        = 0;
    short triggercell = 0;
    short subsector   = 0;
    for(; l1tMappingStream>>layer>>cell>>triggercell>>subsector; )
    {
        if(layer>=30 || layer<0) 
        {
            edm::LogWarning("HGCalTriggerGeometry") << "Bad layer index in L1TMapping\n"; 
            continue; 
        }
        // Loop on all sectors
        // FIXME:  Number of sectors in each zside should not be hardcoded
        for(unsigned z=0; z<=1; z++)
        {
            int zside = (z==0 ? -1 : 1);
            for(unsigned sector=1; sector<=18; sector++)
            {
                HGCEEDetId detid(HGCEE, zside, layer, sector, subsector, cell); 
                // FIXME: Use temporarily HGCEEDetId to compute trigger cell id
                HGCEEDetId triggerDetid(HGCEE, zside, layer, sector, 1, triggercell); // Dummy subsector
                const auto& ret = cells_to_trigger_cells_.insert( std::make_pair(detid(), triggerDetid()) );
                if(!ret.second) edm::LogWarning("HGCalTriggerGeometry") << "Duplicate cell in L1TMapping\n";
            }
        }
    }
    l1tMappingStream.close();

    // Build trigger cells and fill map
    typedef HGCalTriggerGeometry::TriggerCell::list_type list_cells;
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
        // FIXME: dummy module ID, empty neighbours
        std::unique_ptr<const HGCalTriggerGeometry::TriggerCell> triggercellPtr(new HGCalTriggerGeometry::TriggerCell(triggercellId, 1, triggercellPoint, list_cells(), cellIds));
        trigger_cells_.insert( std::make_pair(triggercellId, std::move(triggercellPtr)) );
    }
}


DEFINE_EDM_PLUGIN(HGCalTriggerGeometryFactory, 
        HGCalTriggerGeometryImp1,
        "HGCalTriggerGeometryImp1");
