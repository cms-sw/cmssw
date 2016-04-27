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

#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/HGCTriggerDetId.h"

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
        void setTreeModuleSize(const size_t n);
        void setTreeTriggerCellSize(const size_t n);

        std::unique_ptr<HGCalTriggerGeometryBase> triggerGeometry_; 
        edm::Service<TFileService> fs_;
        TTree* treeModules_;
        TTree* treeTriggerCells_;
        TTree* treeCells_;
        // tree variables
        int   moduleId_     ;
        int   moduleSide_   ;
        int   moduleLayer_  ;
        int   moduleSector_ ;
        int   module_       ;
        float moduleX_      ;
        float moduleY_      ;
        float moduleZ_      ;
        int   moduleTC_N_   ;
        std::shared_ptr<int>   moduleTC_id_    ;
        std::shared_ptr<int>   moduleTC_zside_ ;
        std::shared_ptr<int>   moduleTC_layer_ ;
        std::shared_ptr<int>   moduleTC_sector_;
        std::shared_ptr<int>   moduleTC_cell_  ;
        std::shared_ptr<float> moduleTC_x_     ;
        std::shared_ptr<float> moduleTC_y_     ;
        std::shared_ptr<float> moduleTC_z_     ;
        int   triggerCellId_     ;
        int   triggerCellSide_   ;
        int   triggerCellLayer_  ;
        int   triggerCellSector_ ;
        int   triggerCellModule_ ;
        int   triggerCell_       ;
        float triggerCellX_      ;
        float triggerCellY_      ;
        float triggerCellZ_      ;
        int   triggerCellCell_N_ ;
        std::shared_ptr<int>   triggerCellCell_id_    ;
        std::shared_ptr<int>   triggerCellCell_zside_ ;
        std::shared_ptr<int>   triggerCellCell_layer_ ;
        std::shared_ptr<int>   triggerCellCell_sector_;
        std::shared_ptr<int>   triggerCellCell_cell_  ;
        std::shared_ptr<float> triggerCellCell_x_     ;
        std::shared_ptr<float> triggerCellCell_y_     ;
        std::shared_ptr<float> triggerCellCell_z_     ;
        int   cellId_     ;
        int   cellSide_   ;
        int   cellLayer_  ;
        int   cellSector_ ;
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
        //
};


/*****************************************************************/
HGCalTriggerGeomTester::HGCalTriggerGeomTester(const edm::ParameterSet& conf) 
/*****************************************************************/
{
    //setup geometry 
    const edm::ParameterSet& geometryConfig = conf.getParameterSet("TriggerGeometry");
    const std::string& trigGeomName = geometryConfig.getParameter<std::string>("TriggerGeometryName");
    HGCalTriggerGeometryBase* geometry = HGCalTriggerGeometryFactory::get()->create(trigGeomName,geometryConfig);
    triggerGeometry_.reset(geometry);

    // initialize output trees
    treeModules_ = fs_->make<TTree>("TreeModules","Tree of all HGC modules");
    treeModules_->Branch("id"             , &moduleId_            , "id/I");
    treeModules_->Branch("zside"          , &moduleSide_          , "zside/I");
    treeModules_->Branch("layer"          , &moduleLayer_         , "layer/I");
    treeModules_->Branch("sector"         , &moduleSector_        , "sector/I");
    treeModules_->Branch("module"         , &module_              , "module/I");
    treeModules_->Branch("x"              , &moduleX_             , "x/F");
    treeModules_->Branch("y"              , &moduleY_             , "y/F");
    treeModules_->Branch("z"              , &moduleZ_             , "z/F");
    treeModules_->Branch("tc_n"           , &moduleTC_N_          , "tc_n/I");
    moduleTC_id_    .reset(new int[1],   array_deleter<int>());
    moduleTC_zside_ .reset(new int[1],   array_deleter<int>());
    moduleTC_layer_ .reset(new int[1],   array_deleter<int>());
    moduleTC_sector_.reset(new int[1],   array_deleter<int>());
    moduleTC_cell_  .reset(new int[1],   array_deleter<int>());
    moduleTC_x_     .reset(new float[1], array_deleter<float>());
    moduleTC_y_     .reset(new float[1], array_deleter<float>());
    moduleTC_z_     .reset(new float[1], array_deleter<float>());
    treeModules_->Branch("tc_id"          , moduleTC_id_.get()     , "tc_id[tc_n]/I");
    treeModules_->Branch("tc_zside"       , moduleTC_zside_.get()  , "tc_zside[tc_n]/I");
    treeModules_->Branch("tc_layer"       , moduleTC_layer_.get()  , "tc_layer[tc_n]/I");
    treeModules_->Branch("tc_sector"      , moduleTC_sector_.get() , "tc_sector[tc_n]/I");
    treeModules_->Branch("tc_cell"        , moduleTC_cell_.get()   , "tc_cell[tc_n]/I");
    treeModules_->Branch("tc_x"           , moduleTC_x_.get()      , "tc_x[tc_n]/F");
    treeModules_->Branch("tc_y"           , moduleTC_y_.get()      , "tc_y[tc_n]/F");
    treeModules_->Branch("tc_z"           , moduleTC_z_.get()      , "tc_z[tc_n]/F");
    //
    treeTriggerCells_ = fs_->make<TTree>("TreeTriggerCells","Tree of all HGC trigger cells");
    treeTriggerCells_->Branch("id"             , &triggerCellId_            , "id/I");
    treeTriggerCells_->Branch("zside"          , &triggerCellSide_          , "zside/I");
    treeTriggerCells_->Branch("layer"          , &triggerCellLayer_         , "layer/I");
    treeTriggerCells_->Branch("sector"         , &triggerCellSector_        , "sector/I");
    treeTriggerCells_->Branch("module"         , &triggerCellModule_        , "module/I");
    treeTriggerCells_->Branch("triggercell"    , &triggerCell_              , "triggercell/I");
    treeTriggerCells_->Branch("x"              , &triggerCellX_             , "x/F");
    treeTriggerCells_->Branch("y"              , &triggerCellY_             , "y/F");
    treeTriggerCells_->Branch("z"              , &triggerCellZ_             , "z/F");
    treeTriggerCells_->Branch("c_n"            , &triggerCellCell_N_        , "c_n/I");
    triggerCellCell_id_    .reset(new int[1],   array_deleter<int>());
    triggerCellCell_zside_ .reset(new int[1],   array_deleter<int>());
    triggerCellCell_layer_ .reset(new int[1],   array_deleter<int>());
    triggerCellCell_sector_.reset(new int[1],   array_deleter<int>());
    triggerCellCell_cell_  .reset(new int[1],   array_deleter<int>());
    triggerCellCell_x_     .reset(new float[1], array_deleter<float>());
    triggerCellCell_y_     .reset(new float[1], array_deleter<float>());
    triggerCellCell_z_     .reset(new float[1], array_deleter<float>());
    treeTriggerCells_->Branch("c_id"           , triggerCellCell_id_.get()     , "c_id[c_n]/I");
    treeTriggerCells_->Branch("c_zside"        , triggerCellCell_zside_.get()  , "c_zside[c_n]/I");
    treeTriggerCells_->Branch("c_layer"        , triggerCellCell_layer_.get()  , "c_layer[c_n]/I");
    treeTriggerCells_->Branch("c_sector"       , triggerCellCell_sector_.get() , "c_sector[c_n]/I");
    treeTriggerCells_->Branch("c_cell"         , triggerCellCell_cell_.get()   , "c_cell[c_n]/I");
    treeTriggerCells_->Branch("c_x"            , triggerCellCell_x_.get()      , "c_x[c_n]/F");
    treeTriggerCells_->Branch("c_y"            , triggerCellCell_y_.get()      , "c_y[c_n]/F");
    treeTriggerCells_->Branch("c_z"            , triggerCellCell_z_.get()      , "c_z[c_n]/F");
    //
    treeCells_ = fs_->make<TTree>("TreeCells","Tree of all HGC cells");
    treeCells_->Branch("id"             , &cellId_            , "id/I");
    treeCells_->Branch("zside"          , &cellSide_          , "zside/I");
    treeCells_->Branch("layer"          , &cellLayer_         , "layer/I");
    treeCells_->Branch("sector"         , &cellSector_        , "sector/I");
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
    triggerGeometry_->reset();
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
    triggerGeometry_->initialize(info);

    fillTriggerGeometry(info);
}


/*****************************************************************/
void HGCalTriggerGeomTester::fillTriggerGeometry(const HGCalTriggerGeometryBase::es_info& info)
/*****************************************************************/
{
    // Loop over modules
    std::cout<<"Filling modules tree\n";
    for( const auto& id_module : triggerGeometry_->modules() )
    {
        HGCTriggerDetId id(id_module.first);
        const auto& modulePtr = id_module.second;
        moduleId_     = id.rawId();
        moduleSide_   = id.zside();
        moduleLayer_  = id.layer();
        moduleSector_ = id.sector();
        module_       = id.module();
        moduleX_      = modulePtr->position().x();
        moduleY_      = modulePtr->position().y();
        moduleZ_      = modulePtr->position().z();
        moduleTC_N_   = modulePtr->components().size();
        //
        setTreeModuleSize(moduleTC_N_);
        size_t itc = 0;
        for(const auto& tc : modulePtr->components())
        {
            HGCTriggerDetId tcId(tc);
            const auto& triggerCell = triggerGeometry_->triggerCells().at(tc);
            moduleTC_id_    .get()[itc] = tc;
            moduleTC_zside_ .get()[itc] = tcId.zside();
            moduleTC_layer_ .get()[itc] = tcId.layer();
            moduleTC_sector_.get()[itc] = tcId.sector();
            moduleTC_cell_  .get()[itc] = tcId.cell();
            moduleTC_x_     .get()[itc] = triggerCell->position().x();
            moduleTC_y_     .get()[itc] = triggerCell->position().y();
            moduleTC_z_     .get()[itc] = triggerCell->position().z();
            itc++;
        }
        //
        treeModules_->Fill();
    }
    // Loop over trigger cells
    std::cout<<"Filling trigger cells tree\n";
    for( const auto& id_triggercell : triggerGeometry_->triggerCells() )
    {
        HGCTriggerDetId id(id_triggercell.first);
        const auto& triggerCellPtr = id_triggercell.second;
        triggerCellId_     = id.rawId();
        triggerCellSide_   = id.zside();
        triggerCellLayer_  = id.layer();
        triggerCellSector_ = id.sector();
        triggerCellModule_ = id.module();
        triggerCell_       = id.cell();
        triggerCellX_      = triggerCellPtr->position().x();
        triggerCellY_      = triggerCellPtr->position().y();
        triggerCellZ_      = triggerCellPtr->position().z();
        triggerCellCell_N_ = triggerCellPtr->components().size();
        //
        setTreeTriggerCellSize(triggerCellCell_N_);
        size_t ic = 0;
        for(const auto& c : triggerCellPtr->components())
        {
            HGCEEDetId cId(c);
            GlobalPoint position = info.geom_ee->getPosition(cId);
            triggerCellCell_id_    .get()[ic] = c;
            triggerCellCell_zside_ .get()[ic] = cId.zside();
            triggerCellCell_layer_ .get()[ic] = cId.layer();
            triggerCellCell_sector_.get()[ic] = cId.sector();
            triggerCellCell_cell_  .get()[ic] = cId.cell();
            triggerCellCell_x_     .get()[ic] = position.x();
            triggerCellCell_y_     .get()[ic] = position.y();
            triggerCellCell_z_     .get()[ic] = position.z();
            ic++;
        }
        //
        treeTriggerCells_->Fill();
    }
    // Loop over cells
    std::cout<<"Filling cells tree\n";
    for (int izz=0; izz<=1; izz++) 
    {
        int iz = (2*izz-1);
        for (int subsec=0; subsec<=1; ++subsec) 
        {
            for (int sec=1; sec<=18; ++sec) 
            {
                for (int lay=1; lay<=30; ++lay) 
                {
                    for (int cell=0; cell<8000; ++cell) 
                    {
                        const HGCEEDetId id(HGCEE,iz,lay,sec,subsec,cell);
                        if(!info.topo_ee->valid(id)) continue;
                        cellId_     = id.rawId();
                        cellSide_   = id.zside();
                        cellLayer_  = id.layer();
                        cellSector_ = id.sector();
                        cell_       = id.cell();
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
                    }
                }
            }
        }
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
    moduleTC_layer_ .reset(new int[n],   array_deleter<int>());
    moduleTC_sector_.reset(new int[n],   array_deleter<int>());
    moduleTC_cell_  .reset(new int[n],   array_deleter<int>());
    moduleTC_x_     .reset(new float[n], array_deleter<float>());
    moduleTC_y_     .reset(new float[n], array_deleter<float>());
    moduleTC_z_     .reset(new float[n], array_deleter<float>());

    treeModules_->GetBranch("tc_id")     ->SetAddress(moduleTC_id_    .get());
    treeModules_->GetBranch("tc_zside")  ->SetAddress(moduleTC_zside_ .get());
    treeModules_->GetBranch("tc_layer")  ->SetAddress(moduleTC_layer_ .get());
    treeModules_->GetBranch("tc_sector") ->SetAddress(moduleTC_sector_.get());
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
    triggerCellCell_layer_ .reset(new int[n],   array_deleter<int>());
    triggerCellCell_sector_.reset(new int[n],   array_deleter<int>());
    triggerCellCell_cell_  .reset(new int[n],   array_deleter<int>());
    triggerCellCell_x_     .reset(new float[n], array_deleter<float>());
    triggerCellCell_y_     .reset(new float[n], array_deleter<float>());
    triggerCellCell_z_     .reset(new float[n], array_deleter<float>());

    treeTriggerCells_->GetBranch("c_id")     ->SetAddress(triggerCellCell_id_    .get());
    treeTriggerCells_->GetBranch("c_zside")  ->SetAddress(triggerCellCell_zside_ .get());
    treeTriggerCells_->GetBranch("c_layer")  ->SetAddress(triggerCellCell_layer_ .get());
    treeTriggerCells_->GetBranch("c_sector") ->SetAddress(triggerCellCell_sector_.get());
    treeTriggerCells_->GetBranch("c_cell")   ->SetAddress(triggerCellCell_cell_  .get());
    treeTriggerCells_->GetBranch("c_x")      ->SetAddress(triggerCellCell_x_     .get());
    treeTriggerCells_->GetBranch("c_y")      ->SetAddress(triggerCellCell_y_     .get());
    treeTriggerCells_->GetBranch("c_z")      ->SetAddress(triggerCellCell_z_     .get());
}


//define this as a plug-in
DEFINE_FWK_MODULE(HGCalTriggerGeomTester);
