#include <iostream>
#include <string>
#include <vector>

#include "TTree.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

#include <stdlib.h> 

namespace 
{  
    template<typename T>
    struct array_deleter
    {
        void operator () (T* arr) { delete [] arr; }
    };
}


class HGCalTriggerGeomTester : public edm::EDAnalyzer 
{
    public:
        explicit HGCalTriggerGeomTester(const edm::ParameterSet& );
        ~HGCalTriggerGeomTester();

        virtual void beginRun(const edm::Run&, const edm::EventSetup&);
        virtual void analyze(const edm::Event&, const edm::EventSetup&);


    private:
        void fillTriggerGeometry(const HGCalTriggerGeometryBase::es_info& );
        void checkConsistency(const HGCalTriggerGeometryBase::es_info& );
        void setTreeModuleSize(const size_t n);
        void setTreeTriggerCellSize(const size_t n);

        edm::ESHandle<HGCalTriggerGeometryBase> triggerGeometry_;
        edm::Service<TFileService> fs_;
        bool no_trigger_;
        TTree* treeModules_;
        TTree* treeTriggerCells_;
        TTree* treeCells_;
        // tree variables
        int   moduleId_     ;
        int   moduleSide_   ;
        int   moduleSubdet_ ;
        int   moduleLayer_  ;
        int   module_       ;
        float moduleX_      ;
        float moduleY_      ;
        float moduleZ_      ;
        int   moduleTC_N_   ;
        std::shared_ptr<int>   moduleTC_id_    ;
        std::shared_ptr<int>   moduleTC_zside_ ;
        std::shared_ptr<int>   moduleTC_subdet_;
        std::shared_ptr<int>   moduleTC_layer_ ;
        std::shared_ptr<int>   moduleTC_wafer_;
        std::shared_ptr<int>   moduleTC_cell_  ;
        std::shared_ptr<float> moduleTC_x_     ;
        std::shared_ptr<float> moduleTC_y_     ;
        std::shared_ptr<float> moduleTC_z_     ;
        int   triggerCellId_     ;
        int   triggerCellSide_   ;
        int   triggerCellSubdet_ ;
        int   triggerCellLayer_  ;
        int   triggerCellWafer_ ;
        int   triggerCell_       ;
        float triggerCellX_      ;
        float triggerCellY_      ;
        float triggerCellZ_      ;
        int   triggerCellCell_N_ ;
        std::shared_ptr<int>   triggerCellCell_id_    ;
        std::shared_ptr<int>   triggerCellCell_zside_ ;
        std::shared_ptr<int>   triggerCellCell_subdet_;
        std::shared_ptr<int>   triggerCellCell_layer_ ;
        std::shared_ptr<int>   triggerCellCell_wafer_;
        std::shared_ptr<int>   triggerCellCell_cell_  ;
        std::shared_ptr<float> triggerCellCell_x_     ;
        std::shared_ptr<float> triggerCellCell_y_     ;
        std::shared_ptr<float> triggerCellCell_z_     ;
        int   cellId_     ;
        int   cellSide_   ;
        int   cellSubdet_ ;
        int   cellLayer_  ;
        int   cellWafer_ ;
        int   cellWaferType_ ;
        int cellWaferRow_;
        int cellWaferColumn_;
        int   cell_       ;
        float cellX_      ;
        float cellY_      ;
        float cellZ_      ;
        float cellX1_     ;
        float cellY1_     ;
        float cellX2_     ;
        float cellY2_     ;
        float cellX3_     ;
        float cellY3_     ;
        float cellX4_     ;
        float cellY4_     ;
        
};


/*****************************************************************/
HGCalTriggerGeomTester::HGCalTriggerGeomTester(const edm::ParameterSet& conf):
    no_trigger_(false)
/*****************************************************************/
{

    // initialize output trees
    treeModules_ = fs_->make<TTree>("TreeModules","Tree of all HGC modules");
    treeModules_->Branch("id"             , &moduleId_            , "id/I");
    treeModules_->Branch("zside"          , &moduleSide_          , "zside/I");
    treeModules_->Branch("subdet"         , &moduleSubdet_        , "subdet/I");
    treeModules_->Branch("layer"          , &moduleLayer_         , "layer/I");
    treeModules_->Branch("module"         , &module_              , "module/I");
    treeModules_->Branch("x"              , &moduleX_             , "x/F");
    treeModules_->Branch("y"              , &moduleY_             , "y/F");
    treeModules_->Branch("z"              , &moduleZ_             , "z/F");
    treeModules_->Branch("tc_n"           , &moduleTC_N_          , "tc_n/I");
    moduleTC_id_    .reset(new int[1],   array_deleter<int>());
    moduleTC_zside_ .reset(new int[1],   array_deleter<int>());
    moduleTC_subdet_.reset(new int[1],   array_deleter<int>());
    moduleTC_layer_ .reset(new int[1],   array_deleter<int>());
    moduleTC_wafer_ .reset(new int[1],   array_deleter<int>());
    moduleTC_cell_  .reset(new int[1],   array_deleter<int>());
    moduleTC_x_     .reset(new float[1], array_deleter<float>());
    moduleTC_y_     .reset(new float[1], array_deleter<float>());
    moduleTC_z_     .reset(new float[1], array_deleter<float>());
    treeModules_->Branch("tc_id"          , moduleTC_id_.get()     , "tc_id[tc_n]/I");
    treeModules_->Branch("tc_zside"       , moduleTC_zside_.get()  , "tc_zside[tc_n]/I");
    treeModules_->Branch("tc_subdet"      , moduleTC_subdet_.get() , "tc_subdet[tc_n]/I");
    treeModules_->Branch("tc_layer"       , moduleTC_layer_.get()  , "tc_layer[tc_n]/I");
    treeModules_->Branch("tc_wafer"       , moduleTC_wafer_.get()  , "tc_wafer[tc_n]/I");
    treeModules_->Branch("tc_cell"        , moduleTC_cell_.get()   , "tc_cell[tc_n]/I");
    treeModules_->Branch("tc_x"           , moduleTC_x_.get()      , "tc_x[tc_n]/F");
    treeModules_->Branch("tc_y"           , moduleTC_y_.get()      , "tc_y[tc_n]/F");
    treeModules_->Branch("tc_z"           , moduleTC_z_.get()      , "tc_z[tc_n]/F");
    //
    treeTriggerCells_ = fs_->make<TTree>("TreeTriggerCells","Tree of all HGC trigger cells");
    treeTriggerCells_->Branch("id"             , &triggerCellId_            , "id/I");
    treeTriggerCells_->Branch("zside"          , &triggerCellSide_          , "zside/I");
    treeTriggerCells_->Branch("subdet"         , &triggerCellSubdet_        , "subdet/I");
    treeTriggerCells_->Branch("layer"          , &triggerCellLayer_         , "layer/I");
    treeTriggerCells_->Branch("wafer"          , &triggerCellWafer_          , "wafer/I");
    treeTriggerCells_->Branch("triggercell"    , &triggerCell_              , "triggercell/I");
    treeTriggerCells_->Branch("x"              , &triggerCellX_             , "x/F");
    treeTriggerCells_->Branch("y"              , &triggerCellY_             , "y/F");
    treeTriggerCells_->Branch("z"              , &triggerCellZ_             , "z/F");
    treeTriggerCells_->Branch("c_n"            , &triggerCellCell_N_        , "c_n/I");
    triggerCellCell_id_    .reset(new int[1],   array_deleter<int>());
    triggerCellCell_zside_ .reset(new int[1],   array_deleter<int>());
    triggerCellCell_subdet_ .reset(new int[0],   array_deleter<int>());
    triggerCellCell_layer_ .reset(new int[1],   array_deleter<int>());
    triggerCellCell_wafer_ .reset(new int[1],   array_deleter<int>());
    triggerCellCell_cell_  .reset(new int[1],   array_deleter<int>());
    triggerCellCell_x_     .reset(new float[1], array_deleter<float>());
    triggerCellCell_y_     .reset(new float[1], array_deleter<float>());
    triggerCellCell_z_     .reset(new float[1], array_deleter<float>());
    treeTriggerCells_->Branch("c_id"           , triggerCellCell_id_.get()     , "c_id[c_n]/I");
    treeTriggerCells_->Branch("c_zside"        , triggerCellCell_zside_.get()  , "c_zside[c_n]/I");
    treeTriggerCells_->Branch("c_subdet"       , triggerCellCell_subdet_.get() , "c_subdet[c_n]/I");
    treeTriggerCells_->Branch("c_layer"        , triggerCellCell_layer_.get()  , "c_layer[c_n]/I");
    treeTriggerCells_->Branch("c_wafer"        , triggerCellCell_wafer_.get()  , "c_wafer[c_n]/I");
    treeTriggerCells_->Branch("c_cell"         , triggerCellCell_cell_.get()   , "c_cell[c_n]/I");
    treeTriggerCells_->Branch("c_x"            , triggerCellCell_x_.get()      , "c_x[c_n]/F");
    treeTriggerCells_->Branch("c_y"            , triggerCellCell_y_.get()      , "c_y[c_n]/F");
    treeTriggerCells_->Branch("c_z"            , triggerCellCell_z_.get()      , "c_z[c_n]/F");
    //
    treeCells_ = fs_->make<TTree>("TreeCells","Tree of all HGC cells");
    treeCells_->Branch("id"             , &cellId_            , "id/I");
    treeCells_->Branch("zside"          , &cellSide_          , "zside/I");
    treeCells_->Branch("subdet"         , &cellSubdet_        , "subdet/I");
    treeCells_->Branch("layer"          , &cellLayer_         , "layer/I");
    treeCells_->Branch("wafer"          , &cellWafer_         , "wafer/I");
    treeCells_->Branch("wafertype"      , &cellWaferType_     , "wafertype/I");
    treeCells_->Branch("waferrow"          , &cellWaferRow_         , "waferrow/I");
    treeCells_->Branch("wafercolumn"          , &cellWaferColumn_         , "wafercolumn/I");
    treeCells_->Branch("cell"           , &cell_              , "cell/I");
    treeCells_->Branch("x"              , &cellX_             , "x/F");
    treeCells_->Branch("y"              , &cellY_             , "y/F");
    treeCells_->Branch("z"              , &cellZ_             , "z/F");
    treeCells_->Branch("x1"             , &cellX1_            , "x1/F");
    treeCells_->Branch("y1"             , &cellY1_            , "y1/F");
    treeCells_->Branch("x2"             , &cellX2_            , "x2/F");
    treeCells_->Branch("y2"             , &cellY2_            , "y2/F");
    treeCells_->Branch("x3"             , &cellX3_            , "x3/F");
    treeCells_->Branch("y3"             , &cellY3_            , "y3/F");
    treeCells_->Branch("x4"             , &cellX4_            , "x4/F");
    treeCells_->Branch("y4"             , &cellY4_            , "y4/F");
}



/*****************************************************************/
HGCalTriggerGeomTester::~HGCalTriggerGeomTester() 
/*****************************************************************/
{
}

/*****************************************************************/
void HGCalTriggerGeomTester::beginRun(const edm::Run& /*run*/, 
                                          const edm::EventSetup& es)
/*****************************************************************/
{
    es.get<IdealGeometryRecord>().get("", triggerGeometry_);


    HGCalTriggerGeometryBase::es_info info;
    const std::string& ee_sd_name = triggerGeometry_->eeSDName();
    const std::string& fh_sd_name = triggerGeometry_->fhSDName();
    const std::string& bh_sd_name = triggerGeometry_->bhSDName();
    es.get<IdealGeometryRecord>().get(ee_sd_name,info.geom_ee);
    es.get<IdealGeometryRecord>().get(fh_sd_name,info.geom_fh);
    es.get<IdealGeometryRecord>().get(bh_sd_name,info.geom_bh);
    es.get<IdealGeometryRecord>().get(ee_sd_name,info.topo_ee);
    es.get<IdealGeometryRecord>().get(fh_sd_name,info.topo_fh);
    es.get<IdealGeometryRecord>().get(bh_sd_name,info.topo_bh);

    try
    {
        checkConsistency(info);
    }
    catch(const cms::Exception& e) {
        edm::LogWarning("HGCalTriggerGeometryTester") << "Problem with the trigger geometry detected. Only the basic cells tree will be filled\n";
        edm::LogWarning("HGCalTriggerGeometryTester") << e.message() << "\n";
        no_trigger_ = true;
    }
    fillTriggerGeometry(info);
}


void HGCalTriggerGeomTester::checkConsistency(const HGCalTriggerGeometryBase::es_info& info)
{
    



    std::unordered_map<uint32_t, std::unordered_set<uint32_t>> modules_to_triggercells;
    std::unordered_map<uint32_t, std::unordered_set<uint32_t>> modules_to_cells;
    std::unordered_map<uint32_t, std::unordered_set<uint32_t>> triggercells_to_cells;

    // EE
    for(const auto& id : info.geom_ee->getValidGeomDetIds())
    {
        if(id.rawId()==0) continue;
        HGCalDetId waferid(id); 
        int nCells = info.topo_ee->dddConstants().numberCellsHexagon(waferid.wafer());
        for(int i=0;i<nCells;i++)
        {
            HGCalDetId id(ForwardSubdetector(waferid.subdetId()), waferid.zside(), waferid.layer(), waferid.waferType(), waferid.wafer(), i);
            if(!info.topo_ee->valid(id)) continue;
            // fill trigger cells
            uint32_t trigger_cell = triggerGeometry_->getTriggerCellFromCell(id);
            auto itr_insert = triggercells_to_cells.emplace(trigger_cell, std::unordered_set<uint32_t>());
            itr_insert.first->second.emplace(id);
            // fill modules
            uint32_t module = triggerGeometry_->getModuleFromCell(id);
            itr_insert = modules_to_cells.emplace(module, std::unordered_set<uint32_t>());
            itr_insert.first->second.emplace(id);
        }
    }
    // FH
    for(const auto& id : info.geom_fh->getValidGeomDetIds())
    {
        if(id.rawId()==0) continue;
        HGCalDetId waferid(id); 
        int nCells = info.topo_fh->dddConstants().numberCellsHexagon(waferid.wafer());
        for(int i=0;i<nCells;i++)
        {
            HGCalDetId id(ForwardSubdetector(waferid.subdetId()), waferid.zside(), waferid.layer(), waferid.waferType(), waferid.wafer(), i);
            if(!info.topo_fh->valid(id)) continue;
            // fill trigger cells
            uint32_t trigger_cell = triggerGeometry_->getTriggerCellFromCell(id);
            auto itr_insert = triggercells_to_cells.emplace(trigger_cell, std::unordered_set<uint32_t>());
            itr_insert.first->second.emplace(id);
            // fill modules
            uint32_t module = triggerGeometry_->getModuleFromCell(id);
            itr_insert = modules_to_cells.emplace(module, std::unordered_set<uint32_t>());
            itr_insert.first->second.emplace(id);
        }
    }

    std::cout<<"Checking cell -> trigger cell -> cell consistency\n";
    // Loop over trigger cells
    for( const auto& triggercell_cells : triggercells_to_cells )
    {
        HGCalDetId id(triggercell_cells.first);
        // fill modules
        uint32_t module = triggerGeometry_->getModuleFromTriggerCell(id);
        auto itr_insert = modules_to_triggercells.emplace(module, std::unordered_set<uint32_t>());
        itr_insert.first->second.emplace(id);
        // Check consistency of cells included in trigger cell
        HGCalTriggerGeometryBase::geom_set cells_geom = triggerGeometry_->getCellsFromTriggerCell(id);
        const auto& cells = triggercell_cells.second;
        for(auto cell : cells)
        {
            if(cells_geom.find(cell)==cells_geom.end())
            {
                HGCalDetId cellid(cell);
                std::cout<<"Error: \n Cell "<<cell<<"("<<cellid<<")\n has not been found in \n trigger cell "<<id<<"\n";
                std::cout<<" Available cells are:\n";
                for(auto cell_geom : cells_geom)
                {
                    std::cout<<cell_geom<<" ";
                }
                std::cout<<"\n";
            }
        }
    }
    std::cout<<"Checking trigger cell -> module -> trigger cell consistency\n";
    // Loop over modules
    for( const auto& module_triggercells : modules_to_triggercells )
    {
        HGCalDetId id(module_triggercells.first);
        // Check consistency of trigger cells included in module
        HGCalTriggerGeometryBase::geom_set triggercells_geom = triggerGeometry_->getTriggerCellsFromModule(id);
        const auto& triggercells = module_triggercells.second;
        for(auto cell : triggercells)
        {
            if(triggercells_geom.find(cell)==triggercells_geom.end())
            {
                HGCalDetId cellid(cell);
                std::cout<<"Error: \n Trigger cell "<<cell<<"("<<cellid<<")\n has not been found in \n module "<<id<<"\n";
                std::cout<<" Available trigger cells are:\n";
                for(auto cell_geom : triggercells_geom)
                {
                    std::cout<<cell_geom<<" ";
                }
                std::cout<<"\n";
            }
        }
    }
    std::cout<<"Checking cell -> module -> cell consistency\n";
    for( const auto& module_cells : modules_to_cells )
    {
        HGCalDetId id(module_cells.first);
        // Check consistency of cells included in module
        HGCalTriggerGeometryBase::geom_set cells_geom = triggerGeometry_->getCellsFromModule(id);
        const auto& cells = module_cells.second;
        for(auto cell : cells)
        {
            if(cells_geom.find(cell)==cells_geom.end())
            {
                HGCalDetId cellid(cell);
                std::cout<<"Error: \n Cell "<<cell<<"("<<cellid<<")\n has not been found in \n module "<<id<<"\n";
                std::cout<<" Available cells are:\n";
                for(auto cell_geom : cells_geom)
                {
                    std::cout<<cell_geom<<" ";
                }
                std::cout<<"\n";
            }
        }
    }
}


/*****************************************************************/
void HGCalTriggerGeomTester::fillTriggerGeometry(const HGCalTriggerGeometryBase::es_info& info)
/*****************************************************************/
{
   std::unordered_map<uint32_t, std::unordered_set<uint32_t>> modules;
   std::unordered_map<uint32_t, std::unordered_set<uint32_t>> trigger_cells;

    // Loop over cells
    std::cout<<"Filling cells tree\n";
    // EE
    for(const auto& id : info.geom_ee->getValidGeomDetIds())
    {
        if(id.rawId()==0) continue;
        HGCalDetId waferid(id); 
        int nCells = info.topo_ee->dddConstants().numberCellsHexagon(waferid.wafer());
        for(int i=0;i<nCells;i++)
        {
            HGCalDetId id(ForwardSubdetector(waferid.subdetId()), waferid.zside(), waferid.layer(), waferid.waferType(), waferid.wafer(), i);
            if(!info.topo_ee->valid(id)) continue;
            cellId_         = id.rawId();
            cellSide_       = id.zside();
            cellSubdet_     = id.subdetId();
            cellLayer_      = id.layer();
            cellWafer_      = id.wafer();
            cellWaferType_  = id.waferType();
            auto row_column = info.topo_ee->dddConstants().rowColumnWafer(id.wafer());
            cellWaferRow_  = row_column.first;
            cellWaferColumn_ = row_column.second;
            cell_           = id.cell();
            //
            GlobalPoint center = info.geom_ee->getPosition(id);
            cellX_      = center.x();
            cellY_      = center.y();
            cellZ_      = center.z();
            std::vector<GlobalPoint> corners = info.geom_ee->getCorners(id);
            if(corners.size()<4) std::cout<<"#corners < 4\n";
            else
            {
                cellX1_      = corners.at(0).x();
                cellY1_      = corners.at(0).y();
                cellX2_      = corners.at(1).x();
                cellY2_      = corners.at(1).y();
                cellX3_      = corners.at(2).x();
                cellY3_      = corners.at(2).y();
                cellX4_      = corners.at(3).x();
                cellY4_      = corners.at(3).y();
            }
            treeCells_->Fill();
            // fill trigger cells
            if(!no_trigger_)
            {
                uint32_t trigger_cell = triggerGeometry_->getTriggerCellFromCell(id);
                auto itr_insert = trigger_cells.emplace(trigger_cell, std::unordered_set<uint32_t>());
                itr_insert.first->second.emplace(id);
            }
        }
    }
    // FH
    for(const auto& id : info.geom_fh->getValidGeomDetIds())
    {
        if(id.rawId()==0) continue;
        HGCalDetId waferid(id); 
        int nCells = info.topo_fh->dddConstants().numberCellsHexagon(waferid.wafer());
        for(int i=0;i<nCells;i++)
        {
            HGCalDetId id(ForwardSubdetector(waferid.subdetId()), waferid.zside(), waferid.layer(), waferid.waferType(), waferid.wafer(), i);
            if(!info.topo_fh->valid(id)) continue;
            cellId_         = id.rawId();
            cellSide_       = id.zside();
            cellSubdet_     = id.subdetId();
            cellLayer_      = id.layer();
            cellWafer_      = id.wafer();
            cellWaferType_  = id.waferType();
            auto row_column = info.topo_fh->dddConstants().rowColumnWafer(id.wafer());
            cellWaferRow_  = row_column.first;
            cellWaferColumn_ = row_column.second;
            cell_           = id.cell();
            //
            GlobalPoint center = info.geom_fh->getPosition(id);
            cellX_      = center.x();
            cellY_      = center.y();
            cellZ_      = center.z();
            std::vector<GlobalPoint> corners = info.geom_fh->getCorners(id);
            if(corners.size()<4) std::cout<<"#corners < 4\n";
            else
            {
                cellX1_      = corners.at(0).x();
                cellY1_      = corners.at(0).y();
                cellX2_      = corners.at(1).x();
                cellY2_      = corners.at(1).y();
                cellX3_      = corners.at(2).x();
                cellY3_      = corners.at(2).y();
                cellX4_      = corners.at(3).x();
                cellY4_      = corners.at(3).y();
            }
            treeCells_->Fill();
            // fill trigger cells
            if(!no_trigger_)
            {
                uint32_t trigger_cell = triggerGeometry_->getTriggerCellFromCell(id);
                auto itr_insert = trigger_cells.emplace(trigger_cell, std::unordered_set<uint32_t>());
                itr_insert.first->second.emplace(id);
            }
        }
    }
    // if problem detected in the trigger geometry, don't produce trigger trees
    if(no_trigger_) return;

    // Loop over trigger cells
    std::cout<<"Filling trigger cells tree\n";
    for( const auto& triggercell_cells : trigger_cells )
    {
        HGCalDetId id(triggercell_cells.first);
        GlobalPoint position = triggerGeometry_->getTriggerCellPosition(id);
        triggerCellId_     = id.rawId();
        triggerCellSide_   = id.zside();
        triggerCellSubdet_ = id.subdetId();
        triggerCellLayer_  = id.layer();
        triggerCellWafer_  = id.wafer();
        triggerCell_       = id.cell();
        triggerCellX_      = position.x();
        triggerCellY_      = position.y();
        triggerCellZ_      = position.z();
        triggerCellCell_N_ = triggercell_cells.second.size();
        //
        setTreeTriggerCellSize(triggerCellCell_N_);
        size_t ic = 0;
        for(const auto& c : triggercell_cells.second)
        {
            HGCalDetId cId(c);
            GlobalPoint position = (cId.subdetId()==ForwardSubdetector::HGCEE ? info.geom_ee->getPosition(cId) :  info.geom_fh->getPosition(cId));
            triggerCellCell_id_    .get()[ic] = c;
            triggerCellCell_zside_ .get()[ic] = cId.zside();
            triggerCellCell_subdet_.get()[ic] = cId.subdetId();
            triggerCellCell_layer_ .get()[ic] = cId.layer();
            triggerCellCell_wafer_ .get()[ic] = cId.wafer();
            triggerCellCell_cell_  .get()[ic] = cId.cell();
            triggerCellCell_x_     .get()[ic] = position.x();
            triggerCellCell_y_     .get()[ic] = position.y();
            triggerCellCell_z_     .get()[ic] = position.z();
            ic++;
        }
        //
        treeTriggerCells_->Fill();
        // fill modules
        uint32_t module = triggerGeometry_->getModuleFromTriggerCell(id);
        auto itr_insert = modules.emplace(module, std::unordered_set<uint32_t>());
        itr_insert.first->second.emplace(id);
    }
    // Loop over modules
    std::cout<<"Filling modules tree\n";
    for( const auto& module_triggercells : modules )
    {
        HGCalDetId id(module_triggercells.first);
        GlobalPoint position = triggerGeometry_->getModulePosition(id);
        moduleId_     = id.rawId();
        moduleSide_   = id.zside();
        moduleSubdet_ = id.subdetId();
        moduleLayer_  = id.layer();
        module_       = id.wafer();
        moduleX_      = position.x();
        moduleY_      = position.y();
        moduleZ_      = position.z();
        moduleTC_N_   = module_triggercells.second.size();
        //
        setTreeModuleSize(moduleTC_N_);
        size_t itc = 0;
        for(const auto& tc : module_triggercells.second)
        {
            HGCalDetId tcId(tc);
            GlobalPoint position = triggerGeometry_->getTriggerCellPosition(tcId);
            moduleTC_id_    .get()[itc] = tc;
            moduleTC_zside_ .get()[itc] = tcId.zside();
            moduleTC_subdet_.get()[itc] = tcId.subdetId();
            moduleTC_layer_ .get()[itc] = tcId.layer();
            moduleTC_wafer_ .get()[itc] = tcId.wafer();
            moduleTC_cell_  .get()[itc] = tcId.cell();
            moduleTC_x_     .get()[itc] = position.x();
            moduleTC_y_     .get()[itc] = position.y();
            moduleTC_z_     .get()[itc] = position.z();
            itc++;
        }
        //
        treeModules_->Fill();
    }

}


/*****************************************************************/
void HGCalTriggerGeomTester::analyze(const edm::Event& e, 
        const edm::EventSetup& es) 
/*****************************************************************/
{

}


/*****************************************************************/
void HGCalTriggerGeomTester::setTreeModuleSize(const size_t n) 
/*****************************************************************/
{
    moduleTC_id_    .reset(new int[n],   array_deleter<int>());
    moduleTC_zside_ .reset(new int[n],   array_deleter<int>());
    moduleTC_subdet_.reset(new int[n],   array_deleter<int>());
    moduleTC_layer_ .reset(new int[n],   array_deleter<int>());
    moduleTC_wafer_ .reset(new int[n],   array_deleter<int>());
    moduleTC_cell_  .reset(new int[n],   array_deleter<int>());
    moduleTC_x_     .reset(new float[n], array_deleter<float>());
    moduleTC_y_     .reset(new float[n], array_deleter<float>());
    moduleTC_z_     .reset(new float[n], array_deleter<float>());

    treeModules_->GetBranch("tc_id")     ->SetAddress(moduleTC_id_    .get());
    treeModules_->GetBranch("tc_zside")  ->SetAddress(moduleTC_zside_ .get());
    treeModules_->GetBranch("tc_subdet") ->SetAddress(moduleTC_subdet_.get());
    treeModules_->GetBranch("tc_layer")  ->SetAddress(moduleTC_layer_ .get());
    treeModules_->GetBranch("tc_wafer")  ->SetAddress(moduleTC_wafer_ .get());
    treeModules_->GetBranch("tc_cell")   ->SetAddress(moduleTC_cell_  .get());
    treeModules_->GetBranch("tc_x")      ->SetAddress(moduleTC_x_     .get());
    treeModules_->GetBranch("tc_y")      ->SetAddress(moduleTC_y_     .get());
    treeModules_->GetBranch("tc_z")      ->SetAddress(moduleTC_z_     .get());
}

/*****************************************************************/
void HGCalTriggerGeomTester::setTreeTriggerCellSize(const size_t n) 
/*****************************************************************/
{
    triggerCellCell_id_    .reset(new int[n],   array_deleter<int>());
    triggerCellCell_zside_ .reset(new int[n],   array_deleter<int>());
    triggerCellCell_subdet_.reset(new int[n],   array_deleter<int>());
    triggerCellCell_layer_ .reset(new int[n],   array_deleter<int>());
    triggerCellCell_wafer_ .reset(new int[n],   array_deleter<int>());
    triggerCellCell_cell_  .reset(new int[n],   array_deleter<int>());
    triggerCellCell_x_     .reset(new float[n], array_deleter<float>());
    triggerCellCell_y_     .reset(new float[n], array_deleter<float>());
    triggerCellCell_z_     .reset(new float[n], array_deleter<float>());

    treeTriggerCells_->GetBranch("c_id")     ->SetAddress(triggerCellCell_id_    .get());
    treeTriggerCells_->GetBranch("c_zside")  ->SetAddress(triggerCellCell_zside_ .get());
    treeTriggerCells_->GetBranch("c_subdet") ->SetAddress(triggerCellCell_subdet_.get());
    treeTriggerCells_->GetBranch("c_layer")  ->SetAddress(triggerCellCell_layer_ .get());
    treeTriggerCells_->GetBranch("c_wafer")  ->SetAddress(triggerCellCell_wafer_ .get());
    treeTriggerCells_->GetBranch("c_cell")   ->SetAddress(triggerCellCell_cell_  .get());
    treeTriggerCells_->GetBranch("c_x")      ->SetAddress(triggerCellCell_x_     .get());
    treeTriggerCells_->GetBranch("c_y")      ->SetAddress(triggerCellCell_y_     .get());
    treeTriggerCells_->GetBranch("c_z")      ->SetAddress(triggerCellCell_z_     .get());
}


//define this as a plug-in
DEFINE_FWK_MODULE(HGCalTriggerGeomTester);
