#include <iostream>
#include <string>
#include <vector>

#include "TTree.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
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
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

#include <cstdlib> 

namespace 
{  
    template<typename T>
    struct array_deleter
    {
        void operator () (T* arr) { delete [] arr; }
    };
}


class HGCalTriggerGeomTesterV9 : public edm::stream::EDAnalyzer<>
{
    public:
        explicit HGCalTriggerGeomTesterV9(const edm::ParameterSet& );
        ~HGCalTriggerGeomTesterV9();

        virtual void beginRun(const edm::Run&, const edm::EventSetup&);
        virtual void analyze(const edm::Event&, const edm::EventSetup&);


    private:
        void fillTriggerGeometry();
        bool checkMappingConsistency();
        bool checkNeighborConsistency();
        void setTreeModuleSize(const size_t n);
        void setTreeModuleCellSize(const size_t n);
        void setTreeTriggerCellSize(const size_t n);
        void setTreeCellCornerSize(const size_t n);
        void setTreeTriggerCellNeighborSize(const size_t n);

        edm::ESHandle<HGCalTriggerGeometryBase> triggerGeometry_;
        edm::ESHandle<HGCalGeometry> eeGeometry_;
        edm::ESHandle<HGCalGeometry> hsiGeometry_;
        edm::ESHandle<HGCalGeometry> hscGeometry_;
        edm::Service<TFileService> fs_;
        bool no_trigger_;
        bool no_neighbors_;
        TTree* treeModules_;
        TTree* treeTriggerCells_;
        TTree* treeCells_;
        TTree* treeCellsBH_;
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
        int   moduleCell_N_   ;
        std::shared_ptr<int>   moduleCell_id_    ;
        std::shared_ptr<int>   moduleCell_zside_ ;
        std::shared_ptr<int>   moduleCell_subdet_;
        std::shared_ptr<int>   moduleCell_layer_ ;
        std::shared_ptr<int>   moduleCell_waferU_;
        std::shared_ptr<int>   moduleCell_waferV_;
        std::shared_ptr<int>   moduleCell_cellU_  ;
        std::shared_ptr<int>   moduleCell_cellV_  ;
        std::shared_ptr<float> moduleCell_x_     ;
        std::shared_ptr<float> moduleCell_y_     ;
        std::shared_ptr<float> moduleCell_z_     ;
        int   triggerCellId_     ;
        int   triggerCellSide_   ;
        int   triggerCellSubdet_ ;
        int   triggerCellLayer_  ;
        int   triggerCellWafer_ ;
        int   triggerCell_       ;
        float triggerCellX_      ;
        float triggerCellY_      ;
        float triggerCellZ_      ;
        int triggerCellNeighbor_N_;
        std::shared_ptr<int>   triggerCellNeighbor_id_    ;
        std::shared_ptr<int>   triggerCellNeighbor_zside_ ;
        std::shared_ptr<int>   triggerCellNeighbor_subdet_;
        std::shared_ptr<int>   triggerCellNeighbor_layer_ ;
        std::shared_ptr<int>   triggerCellNeighbor_wafer_;
        std::shared_ptr<int>   triggerCellNeighbor_cell_  ;
        std::shared_ptr<float>   triggerCellNeighbor_distance_  ;
        int   triggerCellCell_N_ ;
        std::shared_ptr<int>   triggerCellCell_id_    ;
        std::shared_ptr<int>   triggerCellCell_zside_ ;
        std::shared_ptr<int>   triggerCellCell_subdet_;
        std::shared_ptr<int>   triggerCellCell_layer_ ;
        std::shared_ptr<int>   triggerCellCell_waferU_;
        std::shared_ptr<int>   triggerCellCell_waferV_;
        std::shared_ptr<int>   triggerCellCell_cellU_  ;
        std::shared_ptr<int>   triggerCellCell_cellV_  ;
        std::shared_ptr<int>   triggerCellCell_ieta_  ;
        std::shared_ptr<int>   triggerCellCell_iphi_  ;
        std::shared_ptr<float> triggerCellCell_x_     ;
        std::shared_ptr<float> triggerCellCell_y_     ;
        std::shared_ptr<float> triggerCellCell_z_     ;
        int   cellId_     ;
        int   cellSide_   ;
        int   cellSubdet_ ;
        int   cellLayer_  ;
        int   cellWaferU_ ;
        int   cellWaferV_ ;
        int   cellWaferType_ ;
        int cellWaferRow_;
        int cellWaferColumn_;
        int   cellU_       ;
        int   cellV_       ;
        float cellX_      ;
        float cellY_      ;
        float cellZ_      ;
        int cellCornersN_;
        std::shared_ptr<float> cellCornersX_      ;
        std::shared_ptr<float> cellCornersY_      ;
        std::shared_ptr<float> cellCornersZ_      ;
        int   cellBHId_     ;
        int   cellBHType_     ;
        int   cellBHSide_   ;
        int   cellBHSubdet_ ;
        int   cellBHLayer_  ;
        int   cellBHIEta_ ;
        int   cellBHIPhi_ ;
        float cellBHEta_      ;
        float cellBHPhi_      ;
        float cellBHX_      ;
        float cellBHY_      ;
        float cellBHZ_      ;
        float cellBHX1_     ;
        float cellBHY1_     ;
        float cellBHX2_     ;
        float cellBHY2_     ;
        float cellBHX3_     ;
        float cellBHY3_     ;
        float cellBHX4_     ;
        float cellBHY4_     ;

    private:
        typedef std::unordered_map<uint32_t, std::unordered_set<uint32_t>>  trigger_map_set;
        
};


/*****************************************************************/
HGCalTriggerGeomTesterV9::HGCalTriggerGeomTesterV9(const edm::ParameterSet& conf):
    no_trigger_(false),
    no_neighbors_(false)
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
    treeModules_->Branch("c_n"           , &moduleCell_N_          , "c_n/I");
    moduleCell_id_    .reset(new int[1],   array_deleter<int>());
    moduleCell_zside_ .reset(new int[1],   array_deleter<int>());
    moduleCell_subdet_.reset(new int[1],   array_deleter<int>());
    moduleCell_layer_ .reset(new int[1],   array_deleter<int>());
    moduleCell_waferU_ .reset(new int[1],   array_deleter<int>());
    moduleCell_waferV_ .reset(new int[1],   array_deleter<int>());
    moduleCell_cellU_  .reset(new int[1],   array_deleter<int>());
    moduleCell_cellV_  .reset(new int[1],   array_deleter<int>());
    moduleCell_x_     .reset(new float[1], array_deleter<float>());
    moduleCell_y_     .reset(new float[1], array_deleter<float>());
    moduleCell_z_     .reset(new float[1], array_deleter<float>());
    treeModules_->Branch("c_id"          , moduleCell_id_.get()     , "c_id[c_n]/I");
    treeModules_->Branch("c_zside"       , moduleCell_zside_.get()  , "c_zside[c_n]/I");
    treeModules_->Branch("c_subdet"      , moduleCell_subdet_.get() , "c_subdet[c_n]/I");
    treeModules_->Branch("c_layer"       , moduleCell_layer_.get()  , "c_layer[c_n]/I");
    treeModules_->Branch("c_waferu"       , moduleCell_waferU_.get()  , "c_waferu[c_n]/I");
    treeModules_->Branch("c_waferv"       , moduleCell_waferV_.get()  , "c_waferv[c_n]/I");
    treeModules_->Branch("c_cellu"        , moduleCell_cellU_.get()   , "c_cellu[c_n]/I");
    treeModules_->Branch("c_cellv"        , moduleCell_cellV_.get()   , "c_cellv[c_n]/I");
    treeModules_->Branch("c_x"           , moduleCell_x_.get()      , "c_x[c_n]/F");
    treeModules_->Branch("c_y"           , moduleCell_y_.get()      , "c_y[c_n]/F");
    treeModules_->Branch("c_z"           , moduleCell_z_.get()      , "c_z[c_n]/F");
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
    treeTriggerCells_->Branch("neighbor_n"     , &triggerCellNeighbor_N_    , "neighbor_n/I");
    triggerCellNeighbor_id_ .reset(new int[1],   array_deleter<int>());
    triggerCellNeighbor_zside_ .reset(new int[1],   array_deleter<int>());
    triggerCellNeighbor_subdet_ .reset(new int[1],   array_deleter<int>());
    triggerCellNeighbor_layer_ .reset(new int[1],   array_deleter<int>());
    triggerCellNeighbor_wafer_ .reset(new int[1],   array_deleter<int>());
    triggerCellNeighbor_cell_  .reset(new int[1],   array_deleter<int>());
    triggerCellNeighbor_distance_  .reset(new float[1],   array_deleter<float>());
    treeTriggerCells_->Branch("neighbor_id", triggerCellNeighbor_id_.get(), "neighbor_id[neighbor_n]/I");
    treeTriggerCells_->Branch("neighbor_zside", triggerCellNeighbor_zside_.get()  , "neighbor_zside[neighbor_n]/I");
    treeTriggerCells_->Branch("neighbor_subdet", triggerCellNeighbor_subdet_.get() , "neighbor_subdet[neighbor_n]/I");
    treeTriggerCells_->Branch("neighbor_layer", triggerCellNeighbor_layer_.get()  , "neighbor_layer[neighbor_n]/I");
    treeTriggerCells_->Branch("neighbor_wafer", triggerCellNeighbor_wafer_.get()  , "neighbor_wafer[neighbor_n]/I");
    treeTriggerCells_->Branch("neighbor_cell", triggerCellNeighbor_cell_.get()   , "neighbor_cell[neighbor_n]/I");
    treeTriggerCells_->Branch("neighbor_distance", triggerCellNeighbor_distance_.get()   , "neighbor_distance[neighbor_n]/F");
    treeTriggerCells_->Branch("c_n"            , &triggerCellCell_N_        , "c_n/I");
    triggerCellCell_id_    .reset(new int[1],   array_deleter<int>());
    triggerCellCell_zside_ .reset(new int[1],   array_deleter<int>());
    triggerCellCell_subdet_ .reset(new int[1],   array_deleter<int>());
    triggerCellCell_layer_ .reset(new int[1],   array_deleter<int>());
    triggerCellCell_waferU_ .reset(new int[1],   array_deleter<int>());
    triggerCellCell_waferV_ .reset(new int[1],   array_deleter<int>());
    triggerCellCell_cellU_  .reset(new int[1],   array_deleter<int>());
    triggerCellCell_cellV_  .reset(new int[1],   array_deleter<int>());
    triggerCellCell_ieta_  .reset(new int[1],   array_deleter<int>());
    triggerCellCell_iphi_  .reset(new int[1],   array_deleter<int>());
    triggerCellCell_x_     .reset(new float[1], array_deleter<float>());
    triggerCellCell_y_     .reset(new float[1], array_deleter<float>());
    triggerCellCell_z_     .reset(new float[1], array_deleter<float>());
    treeTriggerCells_->Branch("c_id"           , triggerCellCell_id_.get()     , "c_id[c_n]/I");
    treeTriggerCells_->Branch("c_zside"        , triggerCellCell_zside_.get()  , "c_zside[c_n]/I");
    treeTriggerCells_->Branch("c_subdet"       , triggerCellCell_subdet_.get() , "c_subdet[c_n]/I");
    treeTriggerCells_->Branch("c_layer"        , triggerCellCell_layer_.get()  , "c_layer[c_n]/I");
    treeTriggerCells_->Branch("c_waferu"        , triggerCellCell_waferU_.get()  , "c_waferu[c_n]/I");
    treeTriggerCells_->Branch("c_waferv"        , triggerCellCell_waferV_.get()  , "c_waferv[c_n]/I");
    treeTriggerCells_->Branch("c_cellu"         , triggerCellCell_cellU_.get()   , "c_cellu[c_n]/I");
    treeTriggerCells_->Branch("c_cellv"         , triggerCellCell_cellV_.get()   , "c_cellv[c_n]/I");
    treeTriggerCells_->Branch("c_ieta"         , triggerCellCell_ieta_.get()   , "c_cell[c_n]/I");
    treeTriggerCells_->Branch("c_iphi"         , triggerCellCell_iphi_.get()   , "c_cell[c_n]/I");
    treeTriggerCells_->Branch("c_x"            , triggerCellCell_x_.get()      , "c_x[c_n]/F");
    treeTriggerCells_->Branch("c_y"            , triggerCellCell_y_.get()      , "c_y[c_n]/F");
    treeTriggerCells_->Branch("c_z"            , triggerCellCell_z_.get()      , "c_z[c_n]/F");
    //
    treeCells_ = fs_->make<TTree>("TreeCells","Tree of all HGC cells");
    treeCells_->Branch("id"             , &cellId_            , "id/I");
    treeCells_->Branch("zside"          , &cellSide_          , "zside/I");
    treeCells_->Branch("subdet"         , &cellSubdet_        , "subdet/I");
    treeCells_->Branch("layer"          , &cellLayer_         , "layer/I");
    treeCells_->Branch("waferu"          , &cellWaferU_         , "waferu/I");
    treeCells_->Branch("waferv"          , &cellWaferV_         , "waferv/I");
    treeCells_->Branch("wafertype"      , &cellWaferType_     , "wafertype/I");
    treeCells_->Branch("waferrow"          , &cellWaferRow_         , "waferrow/I");
    treeCells_->Branch("wafercolumn"          , &cellWaferColumn_         , "wafercolumn/I");
    treeCells_->Branch("cellu"           , &cellU_              , "cellu/I");
    treeCells_->Branch("cellv"           , &cellV_              , "cellv/I");
    treeCells_->Branch("x"              , &cellX_             , "x/F");
    treeCells_->Branch("y"              , &cellY_             , "y/F");
    treeCells_->Branch("z"              , &cellZ_             , "z/F");
    treeCells_->Branch("corner_n"       , &cellCornersN_     , "corner_n/I");
    treeCells_->Branch("corner_x"       , cellCornersX_.get()      , "corner_x[corner_n]/F");
    treeCells_->Branch("corner_y"       , cellCornersY_.get()      , "corner_y[corner_n]/F");
    treeCells_->Branch("corner_z"       , cellCornersZ_.get()      , "corner_z[corner_n]/F");
    //
    treeCellsBH_ = fs_->make<TTree>("TreeCellsBH","Tree of all BH cells");
    treeCellsBH_->Branch("id", &cellBHId_, "id/I");
    treeCellsBH_->Branch("type", &cellBHType_, "type/I");
    treeCellsBH_->Branch("zside", &cellBHSide_, "zside/I");
    treeCellsBH_->Branch("subdet", &cellBHSubdet_, "subdet/I");
    treeCellsBH_->Branch("layer", &cellBHLayer_, "layer/I");
    treeCellsBH_->Branch("ieta", &cellBHIEta_, "ieta/I");
    treeCellsBH_->Branch("iphi", &cellBHIPhi_, "iphi/I");
    treeCellsBH_->Branch("eta", &cellBHEta_, "eta/F");
    treeCellsBH_->Branch("phi", &cellBHPhi_, "phi/F");
    treeCellsBH_->Branch("x", &cellBHX_, "x/F");
    treeCellsBH_->Branch("y", &cellBHY_, "y/F");
    treeCellsBH_->Branch("z", &cellBHZ_, "z/F");
    treeCellsBH_->Branch("x1", &cellBHX1_, "x1/F");
    treeCellsBH_->Branch("y1", &cellBHY1_, "y1/F");
    treeCellsBH_->Branch("x2", &cellBHX2_, "x2/F");
    treeCellsBH_->Branch("y2", &cellBHY2_, "y2/F");
    treeCellsBH_->Branch("x3", &cellBHX3_, "x3/F");
    treeCellsBH_->Branch("y3", &cellBHY3_, "y3/F");
    treeCellsBH_->Branch("x4", &cellBHX4_, "x4/F");
    treeCellsBH_->Branch("y4", &cellBHY4_, "y4/F");
}



/*****************************************************************/
HGCalTriggerGeomTesterV9::~HGCalTriggerGeomTesterV9() 
/*****************************************************************/
{
}

/*****************************************************************/
void HGCalTriggerGeomTesterV9::beginRun(const edm::Run& /*run*/, 
                                          const edm::EventSetup& es)
/*****************************************************************/
{
    es.get<CaloGeometryRecord>().get("", triggerGeometry_);

    no_trigger_ = !checkMappingConsistency();
    no_neighbors_ = !checkNeighborConsistency();
    fillTriggerGeometry();
}


bool HGCalTriggerGeomTesterV9::checkMappingConsistency()
{
    try
    {
        trigger_map_set modules_to_triggercells;
        trigger_map_set modules_to_cells;
        trigger_map_set triggercells_to_cells;

        // EE
        for(const auto& id : triggerGeometry_->eeGeometry()->getValidDetIds())
        {
            HGCSiliconDetId detid(id); 
            if(!triggerGeometry_->eeTopology().valid(id)) continue;
            // fill trigger cells
            uint32_t trigger_cell = triggerGeometry_->getTriggerCellFromCell(id);
            auto itr_insert = triggercells_to_cells.emplace(trigger_cell, std::unordered_set<uint32_t>());
            itr_insert.first->second.emplace(id);
            // fill modules
            uint32_t module = triggerGeometry_->getModuleFromCell(id);
            itr_insert = modules_to_cells.emplace(module, std::unordered_set<uint32_t>());
            itr_insert.first->second.emplace(id);
        }
        // HSi
        for(const auto& id : triggerGeometry_->hsiGeometry()->getValidDetIds())
        {
            HGCSiliconDetId detid(id); 
            if(!triggerGeometry_->hsiTopology().valid(id)) continue;
            // fill trigger cells
            uint32_t trigger_cell = triggerGeometry_->getTriggerCellFromCell(id);
            auto itr_insert = triggercells_to_cells.emplace(trigger_cell, std::unordered_set<uint32_t>());
            itr_insert.first->second.emplace(id);
            // fill modules
            uint32_t module = triggerGeometry_->getModuleFromCell(id);
            itr_insert = modules_to_cells.emplace(module, std::unordered_set<uint32_t>());
            itr_insert.first->second.emplace(id);
        }
        // HSc
        for(const auto& id : triggerGeometry_->hscGeometry()->getValidDetIds())
        {
            // fill trigger cells
            unsigned layer = HGCScintillatorDetId(id).layer();
            if(HGCScintillatorDetId(id).type()!=triggerGeometry_->hscTopology().dddConstants().getTypeTrap(layer))
            {
                std::cout<<"Sci cell type = "<<HGCScintillatorDetId(id).type()<<" != "<<triggerGeometry_->hscTopology().dddConstants().getTypeTrap(layer)<<"\n";
            }
            uint32_t trigger_cell = triggerGeometry_->getTriggerCellFromCell(id);
            auto itr_insert = triggercells_to_cells.emplace(trigger_cell, std::unordered_set<uint32_t>());
            itr_insert.first->second.emplace(id);
            // fill modules
            uint32_t module = triggerGeometry_->getModuleFromCell(id);
            itr_insert = modules_to_cells.emplace(module, std::unordered_set<uint32_t>());
            itr_insert.first->second.emplace(id);
        }



        edm::LogPrint("TriggerCellCheck")<<"Checking cell -> trigger cell -> cell consistency";
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
                    if(id.subdetId()==ForwardSubdetector::HGCHEB)
                    {
                        edm::LogProblem("BadTriggerCell")<<"Error: \n Cell "<<cell<<"("<<HGCScintillatorDetId(cell)<<")\n has not been found in \n trigger cell "<<id;
                        std::stringstream output;
                        output<<" Available cells are:\n";
                        for(auto cell_geom : cells_geom) output<<"     "<<HGCScintillatorDetId(cell_geom)<<"\n";
                        edm::LogProblem("BadTriggerCell")<<output.str();
                    }
                    else
                    {
                        edm::LogProblem("BadTriggerCell")<<"Error: \n Cell "<<cell<<"("<<HGCSiliconDetId(cell)<<")\n has not been found in \n trigger cell "<<id;
                        std::stringstream output;
                        output<<" Available cells are:\n";
                        for(auto cell_geom : cells_geom) output<<"     "<<HGCSiliconDetId(cell_geom)<<"\n";
                        edm::LogProblem("BadTriggerCell")<<output.str();
                    }
                    throw cms::Exception("BadGeometry")
                        << "HGCalTriggerGeometry: Found inconsistency in cell <-> trigger cell mapping";
                }
            }
        }
        edm::LogPrint("ModuleCheck")<<"Checking trigger cell -> module -> trigger cell consistency";
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
                    edm::LogProblem("BadModule")<<"Error: \n Trigger cell "<<cell<<"("<<cellid<<")\n has not been found in \n module "<<id;
                    std::stringstream output;
                    output<<" Available trigger cells are:\n";
                    for(auto cell_geom : triggercells_geom)
                    {
                        output<<"     "<<HGCalDetId(cell_geom)<<"\n";
                    }
                    edm::LogProblem("BadModule")<<output.str();
                    throw cms::Exception("BadGeometry")
                        << "HGCalTriggerGeometry: Found inconsistency in trigger cell <->  module mapping";
                }
            }
        }
        edm::LogPrint("ModuleCheck")<<"Checking cell -> module -> cell consistency";
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
                    if(id.subdetId()==ForwardSubdetector::HGCHEB)
                    {
                        edm::LogProblem("BadModule")<<"Error: \n Cell "<<cell<<"("<<HGCScintillatorDetId(cell)<<")\n has not been found in \n module "<<id;
                    }
                    else
                    {
                        edm::LogProblem("BadModule")<<"Error: \n Cell "<<cell<<"("<<HGCSiliconDetId(cell)<<")\n has not been found in \n module "<<id;
                    }
                    std::stringstream output;
                    output<<" Available cells are:\n";
                    for(auto cell_geom : cells_geom)
                    {
                        output<<cell_geom<<" ";
                    }
                    edm::LogProblem("BadModule")<<output.str();
                    throw cms::Exception("BadGeometry")
                        << "HGCalTriggerGeometry: Found inconsistency in cell <->  module mapping";
                }
            }
        }
    }
    catch(const cms::Exception& e) {
        edm::LogWarning("HGCalTriggerGeometryTester") << "Problem with the trigger geometry detected. Only the basic cells tree will be filled\n";
        edm::LogWarning("HGCalTriggerGeometryTester") << e.message() << "\n";
        return false;
    }
    return true;
}


bool HGCalTriggerGeomTesterV9::checkNeighborConsistency()
{
    try
    {
        trigger_map_set triggercells_to_cells;

        // EE
        for(const auto& id : triggerGeometry_->eeGeometry()->getValidDetIds())
        {
            if(!triggerGeometry_->eeTopology().valid(id)) continue;
            // fill trigger cells
            // Skip trigger cells in module 0
            uint32_t trigger_cell = triggerGeometry_->getTriggerCellFromCell(id);
            uint32_t module = triggerGeometry_->getModuleFromTriggerCell(trigger_cell);
            if(HGCalDetId(module).wafer()==0) continue;
            auto itr_insert = triggercells_to_cells.emplace(trigger_cell, std::unordered_set<uint32_t>());
            itr_insert.first->second.emplace(id);
        }
        // HSi
        for(const auto& id : triggerGeometry_->hsiGeometry()->getValidDetIds())
        {
            if(!triggerGeometry_->hsiTopology().valid(id)) continue;
            // fill trigger cells
            // Skip trigger cells in module 0
            uint32_t trigger_cell = triggerGeometry_->getTriggerCellFromCell(id);
            uint32_t module = triggerGeometry_->getModuleFromTriggerCell(trigger_cell);
            if(HGCalDetId(module).wafer()==0) continue;
            auto itr_insert = triggercells_to_cells.emplace(trigger_cell, std::unordered_set<uint32_t>());
            itr_insert.first->second.emplace(id);
        }

        // HSc
        for(const auto& id : triggerGeometry_->hscGeometry()->getValidDetIds())
        {
            if(!triggerGeometry_->hscTopology().valid(id)) continue;
            // fill trigger cells
            // Skip trigger cells in module 0
            uint32_t trigger_cell = triggerGeometry_->getTriggerCellFromCell(id);
            uint32_t module = triggerGeometry_->getModuleFromTriggerCell(trigger_cell);
            if(HGCalDetId(module).wafer()==0) continue;
            auto itr_insert = triggercells_to_cells.emplace(trigger_cell, std::unordered_set<uint32_t>());
            itr_insert.first->second.emplace(id);
        }

        edm::LogPrint("NeighborCheck")<<"Checking trigger cell neighbor consistency";
        // Loop over trigger cells
        for( const auto& triggercell_cells : triggercells_to_cells )
        {
            unsigned triggercell_id(triggercell_cells.first);
            const auto neighbors = triggerGeometry_->getNeighborsFromTriggerCell(triggercell_id);
            for(const auto neighbor : neighbors)
            {
                const auto neighbors_of_neighbor = triggerGeometry_->getNeighborsFromTriggerCell(neighbor);
                // check if the original cell is included in the neigbors of neighbor
                if(neighbors_of_neighbor.find(triggercell_id)==neighbors_of_neighbor.end())
                {
                    edm::LogProblem("BadNeighbor")<<"Error: \n Trigger cell "<< HGCalDetId(neighbor) << "\n is a neighbor of \n" << HGCalDetId(triggercell_id);
                    edm::LogProblem("BadNeighbor")<<" But the opposite is not true";
                    std::stringstream output;
                    output<<" List of neighbors of neighbor = \n";
                    for(const auto neighbor_of_neighbor : neighbors_of_neighbor)
                    {
                        output<<"  "<< HGCalDetId(neighbor_of_neighbor)<<"\n";
                    }
                    edm::LogProblem("BadNeighbor")<<output.str();
                }
            }
        }
    }
    catch(const cms::Exception& e) {
        edm::LogWarning("HGCalTriggerGeometryTester") << "Problem with the trigger neighbors detected. No neighbor information will be filled\n";
        edm::LogWarning("HGCalTriggerGeometryTester") << e.message() << "\n";
        return false;
    }
    return true;
}


/*****************************************************************/
void HGCalTriggerGeomTesterV9::fillTriggerGeometry()
/*****************************************************************/
{
    trigger_map_set modules;
    trigger_map_set trigger_cells;

    // Loop over cells
    edm::LogPrint("TreeFilling")<<"Filling cells tree";
    // EE
    std::cout<<"Filling EE geometry\n";
    for(const auto& id : triggerGeometry_->eeGeometry()->getValidDetIds())
    {
        HGCSiliconDetId detid(id); 
        cellId_         = detid.rawId();
        cellSide_       = detid.zside();
        cellSubdet_     = detid.subdet();
        cellLayer_      = detid.layer();
        cellWaferU_     = detid.waferU();
        cellWaferV_     = detid.waferV();
        cellU_          = detid.cellU();
        cellV_          = detid.cellV();
        int type1 = detid.type();
        int type2 = triggerGeometry_->eeTopology().dddConstants().getTypeHex(cellLayer_, cellWaferU_, cellWaferV_);
        if(type1!=type2)
        {
            std::cout<<"Found incompatible wafer types:\n  "<<detid<<"\n";
        }
        //
        GlobalPoint center = triggerGeometry_->eeGeometry()->getPosition(id);
        cellX_      = center.x();
        cellY_      = center.y();
        cellZ_      = center.z();
        std::vector<GlobalPoint> corners = triggerGeometry_->eeGeometry()->getCorners(id);
        cellCornersN_ = corners.size();
        setTreeCellCornerSize(cellCornersN_);
        for(unsigned i=0; i<corners.size(); i++)
        {
            cellCornersX_.get()[i] = corners[i].x();
            cellCornersY_.get()[i] = corners[i].y();
            cellCornersZ_.get()[i] = corners[i].z();
        }
        treeCells_->Fill();
        // fill trigger cells
        if(!no_trigger_)
        {
            uint32_t trigger_cell = triggerGeometry_->getTriggerCellFromCell(id);
            // Skip trigger cells in module 0
            uint32_t module = triggerGeometry_->getModuleFromTriggerCell(trigger_cell);
            if(HGCalDetId(module).wafer()==0) continue;
            auto itr_insert = trigger_cells.emplace(trigger_cell, std::unordered_set<uint32_t>());
            itr_insert.first->second.emplace(id);
        }
    }
    std::cout<<"Filling HSi geometry\n";
    for(const auto& id : triggerGeometry_->hsiGeometry()->getValidDetIds())
    {
        HGCSiliconDetId detid(id); 
        cellId_         = detid.rawId();
        cellSide_       = detid.zside();
        cellSubdet_     = detid.subdet();
        cellLayer_      = detid.layer();
        cellWaferU_     = detid.waferU();
        cellWaferV_     = detid.waferV();
        cellU_          = detid.cellU();
        cellV_          = detid.cellV();
        int type1 = detid.type();
        int type2 = triggerGeometry_->hsiTopology().dddConstants().getTypeHex(cellLayer_, cellWaferU_, cellWaferV_);
        if(type1!=type2)
        {
            std::cout<<"Found incompatible wafer types:\n  "<<detid<<"\n";
        }
        //
        GlobalPoint center = triggerGeometry_->hsiGeometry()->getPosition(id);
        cellX_      = center.x();
        cellY_      = center.y();
        cellZ_      = center.z();
        std::vector<GlobalPoint> corners = triggerGeometry_->hsiGeometry()->getCorners(id);
        cellCornersN_ = corners.size();
        setTreeCellCornerSize(cellCornersN_);
        for(unsigned i=0; i<corners.size(); i++)
        {
            cellCornersX_.get()[i] = corners[i].x();
            cellCornersY_.get()[i] = corners[i].y();
            cellCornersZ_.get()[i] = corners[i].z();
        }
        treeCells_->Fill();
        // fill trigger cells
        if(!no_trigger_)
        {
            uint32_t trigger_cell = triggerGeometry_->getTriggerCellFromCell(id);
            // Skip trigger cells in module 0
            uint32_t module = triggerGeometry_->getModuleFromTriggerCell(trigger_cell);
            if(HGCalDetId(module).wafer()==0) continue;
            auto itr_insert = trigger_cells.emplace(trigger_cell, std::unordered_set<uint32_t>());
            itr_insert.first->second.emplace(id);
        }
    }
    std::cout<<"Filling HSc geometry\n";
    for(const auto& id : triggerGeometry_->hscGeometry()->getValidDetIds())
    {
        HGCScintillatorDetId cellid(id); 
        cellBHId_ = cellid.rawId();
        cellBHType_ = cellid.type();
        cellBHSide_ = cellid.zside();
        cellBHSubdet_ = cellid.subdetId();
        cellBHLayer_ = cellid.layer();
        cellBHIEta_ = cellid.ieta();
        cellBHIPhi_ = cellid.iphi();
        //
        GlobalPoint center = triggerGeometry_->hscGeometry()->getPosition(id);
        cellBHEta_      = center.eta();
        cellBHPhi_      = center.phi();
        cellBHX_      = center.x();
        cellBHY_      = center.y();
        cellBHZ_      = center.z();
        auto corners = triggerGeometry_->hscGeometry()->getCorners(id);
        if(corners.size()>=4)
        {
            cellBHX1_      = corners[0].x();
            cellBHY1_      = corners[0].y();
            cellBHX2_      = corners[1].x();
            cellBHY2_      = corners[1].y();
            cellBHX3_      = corners[2].x();
            cellBHY3_      = corners[2].y();
            cellBHX4_      = corners[3].x();
            cellBHY4_      = corners[3].y();
        }
        treeCellsBH_->Fill();
        // fill trigger cells
        if(!no_trigger_)
        {
            uint32_t trigger_cell = triggerGeometry_->getTriggerCellFromCell(id);
            // Skip trigger cells in module 0
            uint32_t module = triggerGeometry_->getModuleFromTriggerCell(trigger_cell);
            if(HGCalDetId(module).wafer()==0) continue;
            auto itr_insert = trigger_cells.emplace(trigger_cell, std::unordered_set<uint32_t>());
            itr_insert.first->second.emplace(id);
        }
    }

    // if problem detected in the trigger geometry, don't produce trigger trees
    if(no_trigger_) return;

    // Loop over trigger cells
    edm::LogPrint("TreeFilling")<<"Filling trigger cells tree";
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
            if(id.subdetId()==ForwardSubdetector::HGCHEB)
            {
                HGCScintillatorDetId cId(c);
                GlobalPoint cell_position = triggerGeometry_->hscGeometry()->getPosition(cId);
                triggerCellCell_id_    .get()[ic] = c;
                triggerCellCell_zside_ .get()[ic] = cId.zside();
                triggerCellCell_subdet_.get()[ic] = cId.subdetId();
                triggerCellCell_layer_ .get()[ic] = cId.layer();
                triggerCellCell_waferU_ .get()[ic] = 0;
                triggerCellCell_waferV_ .get()[ic] = 0;
                triggerCellCell_cellU_  .get()[ic] = 0;
                triggerCellCell_cellV_  .get()[ic] = 0;
                triggerCellCell_ieta_  .get()[ic] = cId.ietaAbs();
                triggerCellCell_iphi_  .get()[ic] = cId.iphi();
                triggerCellCell_x_     .get()[ic] = cell_position.x();
                triggerCellCell_y_     .get()[ic] = cell_position.y();
                triggerCellCell_z_     .get()[ic] = cell_position.z();
            }
            else
            {
                HGCSiliconDetId cId(c);
                GlobalPoint cell_position = (cId.det()==DetId::HGCalEE ? triggerGeometry_->eeGeometry()->getPosition(cId) :  triggerGeometry_->hsiGeometry()->getPosition(cId));
                triggerCellCell_id_    .get()[ic] = c;
                triggerCellCell_zside_ .get()[ic] = cId.zside();
                triggerCellCell_subdet_.get()[ic] = cId.subdetId();
                triggerCellCell_layer_ .get()[ic] = cId.layer();
                triggerCellCell_waferU_ .get()[ic] = cId.waferU();
                triggerCellCell_waferV_ .get()[ic] = cId.waferV();
                triggerCellCell_cellU_  .get()[ic] = cId.cellU();
                triggerCellCell_cellV_  .get()[ic] = cId.cellV();
                triggerCellCell_ieta_  .get()[ic] = 0;
                triggerCellCell_iphi_  .get()[ic] = 0;
                triggerCellCell_x_     .get()[ic] = cell_position.x();
                triggerCellCell_y_     .get()[ic] = cell_position.y();
                triggerCellCell_z_     .get()[ic] = cell_position.z();
            }
            ic++;
        }
        // Get neighbors
        if(!no_neighbors_)
        {
            const auto neighbors = triggerGeometry_->getNeighborsFromTriggerCell(id.rawId());
            triggerCellNeighbor_N_ = neighbors.size();
            setTreeTriggerCellNeighborSize(triggerCellNeighbor_N_);
            size_t in = 0;
            for(const auto neighbor : neighbors)
            {
                HGCalDetId nId(neighbor);
                // std::cout<<"Neighbor ID "<<nId<<"\n";
                GlobalPoint neighbor_position = triggerGeometry_->getTriggerCellPosition(neighbor);
                triggerCellNeighbor_id_.get()[in] = neighbor;
                triggerCellNeighbor_zside_ .get()[in] = nId.zside();
                triggerCellNeighbor_subdet_.get()[in] = nId.subdetId();
                triggerCellNeighbor_layer_ .get()[in] = nId.layer();
                triggerCellNeighbor_wafer_ .get()[in] = nId.wafer();
                triggerCellNeighbor_cell_  .get()[in] = nId.cell();
                triggerCellNeighbor_distance_.get()[in] = (neighbor_position - position).mag();
                in++;
            }
        }
        
        treeTriggerCells_->Fill();
        // fill modules
        uint32_t module = triggerGeometry_->getModuleFromTriggerCell(id);
        auto itr_insert = modules.emplace(module, std::unordered_set<uint32_t>());
        itr_insert.first->second.emplace(id);
    }
    // Loop over modules
    edm::LogPrint("TreeFilling")<<"Filling modules tree";
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
        auto cells_in_module = triggerGeometry_->getCellsFromModule(id);
        moduleCell_N_   = cells_in_module.size();
        //
        setTreeModuleCellSize(moduleCell_N_);
        size_t ic = 0;
        for(const auto& c : cells_in_module)
        {
            if(id.subdetId()==ForwardSubdetector::HGCHEB)
            {
                HGCScintillatorDetId cId(c);
                GlobalPoint cell_position = triggerGeometry_->hscGeometry()->getPosition(cId);
                triggerCellCell_id_    .get()[ic] = c;
                triggerCellCell_zside_ .get()[ic] = cId.zside();
                triggerCellCell_subdet_.get()[ic] = cId.subdetId();
                triggerCellCell_layer_ .get()[ic] = cId.layer();
                triggerCellCell_waferU_ .get()[ic] = 0;
                triggerCellCell_waferV_ .get()[ic] = 0;
                triggerCellCell_cellU_  .get()[ic] = 0;
                triggerCellCell_cellV_  .get()[ic] = 0;
                triggerCellCell_ieta_  .get()[ic] = cId.ietaAbs();
                triggerCellCell_iphi_  .get()[ic] = cId.iphi();
                triggerCellCell_x_     .get()[ic] = cell_position.x();
                triggerCellCell_y_     .get()[ic] = cell_position.y();
                triggerCellCell_z_     .get()[ic] = cell_position.z();
            }
            else
            {
                HGCSiliconDetId cId(c);
                const GlobalPoint position = (cId.det()==DetId::HGCalEE ? triggerGeometry_->eeGeometry()->getPosition(cId) :  triggerGeometry_->hsiGeometry()->getPosition(cId));
                moduleCell_id_    .get()[ic] = c;
                moduleCell_zside_ .get()[ic] = cId.zside();
                moduleCell_subdet_.get()[ic] = cId.subdetId();
                moduleCell_layer_ .get()[ic] = cId.layer();
                moduleCell_waferU_ .get()[ic] = cId.waferU();
                moduleCell_waferV_ .get()[ic] = cId.waferV();
                moduleCell_cellU_  .get()[ic] = cId.cellU();
                moduleCell_cellV_  .get()[ic] = cId.cellV();
                moduleCell_x_     .get()[ic] = position.x();
                moduleCell_y_     .get()[ic] = position.y();
                moduleCell_z_     .get()[ic] = position.z();
                ic++;
            }
        }
        //
        treeModules_->Fill();
    }

}


/*****************************************************************/
void HGCalTriggerGeomTesterV9::analyze(const edm::Event& e, 
        const edm::EventSetup& es) 
/*****************************************************************/
{

}


/*****************************************************************/
void HGCalTriggerGeomTesterV9::setTreeModuleSize(const size_t n) 
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
void HGCalTriggerGeomTesterV9::setTreeModuleCellSize(const size_t n) 
/*****************************************************************/
{
    moduleCell_id_    .reset(new int[n],   array_deleter<int>());
    moduleCell_zside_ .reset(new int[n],   array_deleter<int>());
    moduleCell_subdet_.reset(new int[n],   array_deleter<int>());
    moduleCell_layer_ .reset(new int[n],   array_deleter<int>());
    moduleCell_waferU_ .reset(new int[n],   array_deleter<int>());
    moduleCell_waferV_ .reset(new int[n],   array_deleter<int>());
    moduleCell_cellU_  .reset(new int[n],   array_deleter<int>());
    moduleCell_cellV_  .reset(new int[n],   array_deleter<int>());
    moduleCell_x_     .reset(new float[n], array_deleter<float>());
    moduleCell_y_     .reset(new float[n], array_deleter<float>());
    moduleCell_z_     .reset(new float[n], array_deleter<float>());

    treeModules_->GetBranch("c_id")     ->SetAddress(moduleCell_id_    .get());
    treeModules_->GetBranch("c_zside")  ->SetAddress(moduleCell_zside_ .get());
    treeModules_->GetBranch("c_subdet") ->SetAddress(moduleCell_subdet_.get());
    treeModules_->GetBranch("c_layer")  ->SetAddress(moduleCell_layer_ .get());
    treeModules_->GetBranch("c_waferu")  ->SetAddress(moduleCell_waferU_ .get());
    treeModules_->GetBranch("c_waferv")  ->SetAddress(moduleCell_waferV_ .get());
    treeModules_->GetBranch("c_cellu")   ->SetAddress(moduleCell_cellU_  .get());
    treeModules_->GetBranch("c_cellv")   ->SetAddress(moduleCell_cellV_  .get());
    treeModules_->GetBranch("c_x")      ->SetAddress(moduleCell_x_     .get());
    treeModules_->GetBranch("c_y")      ->SetAddress(moduleCell_y_     .get());
    treeModules_->GetBranch("c_z")      ->SetAddress(moduleCell_z_     .get());
}

/*****************************************************************/
void HGCalTriggerGeomTesterV9::setTreeTriggerCellSize(const size_t n) 
/*****************************************************************/
{
    triggerCellCell_id_    .reset(new int[n],   array_deleter<int>());
    triggerCellCell_zside_ .reset(new int[n],   array_deleter<int>());
    triggerCellCell_subdet_.reset(new int[n],   array_deleter<int>());
    triggerCellCell_layer_ .reset(new int[n],   array_deleter<int>());
    triggerCellCell_waferU_ .reset(new int[n],   array_deleter<int>());
    triggerCellCell_waferV_ .reset(new int[n],   array_deleter<int>());
    triggerCellCell_cellU_  .reset(new int[n],   array_deleter<int>());
    triggerCellCell_cellV_  .reset(new int[n],   array_deleter<int>());
    triggerCellCell_ieta_  .reset(new int[n],   array_deleter<int>());
    triggerCellCell_iphi_  .reset(new int[n],   array_deleter<int>());
    triggerCellCell_x_     .reset(new float[n], array_deleter<float>());
    triggerCellCell_y_     .reset(new float[n], array_deleter<float>());
    triggerCellCell_z_     .reset(new float[n], array_deleter<float>());

    treeTriggerCells_->GetBranch("c_id")     ->SetAddress(triggerCellCell_id_    .get());
    treeTriggerCells_->GetBranch("c_zside")  ->SetAddress(triggerCellCell_zside_ .get());
    treeTriggerCells_->GetBranch("c_subdet") ->SetAddress(triggerCellCell_subdet_.get());
    treeTriggerCells_->GetBranch("c_layer")  ->SetAddress(triggerCellCell_layer_ .get());
    treeTriggerCells_->GetBranch("c_waferu")  ->SetAddress(triggerCellCell_waferU_ .get());
    treeTriggerCells_->GetBranch("c_waferv")  ->SetAddress(triggerCellCell_waferV_ .get());
    treeTriggerCells_->GetBranch("c_cellu")   ->SetAddress(triggerCellCell_cellU_  .get());
    treeTriggerCells_->GetBranch("c_cellv")   ->SetAddress(triggerCellCell_cellV_  .get());
    treeTriggerCells_->GetBranch("c_ieta")   ->SetAddress(triggerCellCell_ieta_  .get());
    treeTriggerCells_->GetBranch("c_iphi")   ->SetAddress(triggerCellCell_iphi_  .get());
    treeTriggerCells_->GetBranch("c_x")      ->SetAddress(triggerCellCell_x_     .get());
    treeTriggerCells_->GetBranch("c_y")      ->SetAddress(triggerCellCell_y_     .get());
    treeTriggerCells_->GetBranch("c_z")      ->SetAddress(triggerCellCell_z_     .get());
}


/*****************************************************************/
void HGCalTriggerGeomTesterV9::setTreeCellCornerSize(const size_t n) 
/*****************************************************************/
{
    cellCornersX_.reset(new float[n],   array_deleter<float>());
    cellCornersY_.reset(new float[n],   array_deleter<float>());
    cellCornersZ_.reset(new float[n],   array_deleter<float>());

    treeCells_->GetBranch("corner_x")->SetAddress(cellCornersX_.get());
    treeCells_->GetBranch("corner_y")->SetAddress(cellCornersY_.get());
    treeCells_->GetBranch("corner_z")->SetAddress(cellCornersZ_.get());
}



/*****************************************************************/
void HGCalTriggerGeomTesterV9::setTreeTriggerCellNeighborSize(const size_t n) 
/*****************************************************************/
{
    triggerCellNeighbor_id_.reset(new int[n],array_deleter<int>());
    triggerCellNeighbor_zside_ .reset(new int[n],   array_deleter<int>());
    triggerCellNeighbor_subdet_.reset(new int[n],   array_deleter<int>());
    triggerCellNeighbor_layer_ .reset(new int[n],   array_deleter<int>());
    triggerCellNeighbor_wafer_ .reset(new int[n],   array_deleter<int>());
    triggerCellNeighbor_cell_  .reset(new int[n],   array_deleter<int>());
    triggerCellNeighbor_distance_  .reset(new float[n],   array_deleter<float>());
    treeTriggerCells_->GetBranch("neighbor_id")->SetAddress(triggerCellNeighbor_id_.get());
    treeTriggerCells_->GetBranch("neighbor_zside")  ->SetAddress(triggerCellNeighbor_zside_ .get());
    treeTriggerCells_->GetBranch("neighbor_subdet") ->SetAddress(triggerCellNeighbor_subdet_.get());
    treeTriggerCells_->GetBranch("neighbor_layer")  ->SetAddress(triggerCellNeighbor_layer_ .get());
    treeTriggerCells_->GetBranch("neighbor_wafer")  ->SetAddress(triggerCellNeighbor_wafer_ .get());
    treeTriggerCells_->GetBranch("neighbor_cell")   ->SetAddress(triggerCellNeighbor_cell_  .get());
    treeTriggerCells_->GetBranch("neighbor_distance")   ->SetAddress(triggerCellNeighbor_distance_  .get());
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalTriggerGeomTesterV9);
