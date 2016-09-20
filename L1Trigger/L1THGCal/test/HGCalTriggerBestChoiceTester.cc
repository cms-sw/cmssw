#include <iostream>
#include <string>
#include <vector>


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

#include "DataFormats/L1THGCal/interface/HGCFETriggerDigi.h"
#include "DataFormats/L1THGCal/interface/HGCFETriggerDigiFwd.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "DataFormats/ForwardDetId/interface/HGCTriggerDetId.h"

#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/HGCTriggerDetId.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerFECodecBase.h"
#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalBestChoiceCodecImpl.h"

#include <stdlib.h> 
#include "TH2.h"

using namespace std;

class HGCalTriggerBestChoiceTester : public edm::EDAnalyzer 
{
    public:
        explicit HGCalTriggerBestChoiceTester(const edm::ParameterSet& );
        ~HGCalTriggerBestChoiceTester();

        virtual void beginRun(const edm::Run&, const edm::EventSetup&);
        virtual void analyze(const edm::Event&, const edm::EventSetup&);


    private:
        void checkSelectedCells(const edm::Event&, const edm::EventSetup&);
        void rerunBestChoiceFragments(const edm::Event&, const edm::EventSetup&);
        void fillModule(const std::vector<HGCDataFrame<HGCalDetId,HGCSample>>&,
                const HGCalBestChoiceDataPayload&,
                const vector<pair<HGCalDetId, uint32_t > >& );
        // inputs
        edm::EDGetToken inputee_, inputfh_, inputbh_, inputbeall_, inputbeselect_;
        //
        std::unique_ptr<HGCalTriggerGeometryBase> triggerGeometry_; 
        std::unique_ptr<HGCalBestChoiceCodecImpl> codec_;
        edm::Service<TFileService> fs_;
        // histos
        TH1F* hgcCellsPerModule_;
        TH1F* hgcCellData_;
        TH1F* hgcCellData_linampl_;
        TH1F* hgcCellModuleSum_;
        TH1F* triggerCellsPerModule_;
        TH1F* triggerCellData_;
        TH1F* triggerCellModuleSum_;
        TH2F* selectedCellsVsAllCells_ee_; 
        TH2F* energyLossVsNCells_ee_;
        TH2F* selectedCellsVsAllCells_fh_; 
        TH2F* energyLossVsNCells_fh_;

};


/*****************************************************************/
HGCalTriggerBestChoiceTester::HGCalTriggerBestChoiceTester(const edm::ParameterSet& conf):
  inputee_(consumes<HGCEEDigiCollection>(conf.getParameter<edm::InputTag>("eeDigis"))),
  inputfh_(consumes<HGCHEDigiCollection>(conf.getParameter<edm::InputTag>("fhDigis"))), 
  //inputbh_(consumes<HGCHEDigiCollection>(conf.getParameter<edm::InputTag>("bhDigis"))),
  inputbeall_(consumes<l1t::HGCalClusterBxCollection>(conf.getParameter<edm::InputTag>("beClustersAll"))),
  inputbeselect_(consumes<l1t::HGCalClusterBxCollection>(conf.getParameter<edm::InputTag>("beClustersSelect")))
/*****************************************************************/
{
    //setup geometry 
    const edm::ParameterSet& geometryConfig = conf.getParameterSet("TriggerGeometry");
    const std::string& trigGeomName = geometryConfig.getParameter<std::string>("TriggerGeometryName");
    HGCalTriggerGeometryBase* geometry = HGCalTriggerGeometryFactory::get()->create(trigGeomName,geometryConfig);
    triggerGeometry_.reset(geometry);

    //setup FE codec
    const edm::ParameterSet& feCodecConfig = conf.getParameterSet("FECodec");
    codec_.reset( new HGCalBestChoiceCodecImpl(feCodecConfig) );

    // initialize output trees
    hgcCellsPerModule_       = fs_->make<TH1F>("hgcCellsPerModule","Number of cells per module", 64, 0., 64.);
    hgcCellData_             = fs_->make<TH1F>("hgcCellData","Cell values", 500, 0., 500.);
    //
    hgcCellData_linampl_     = fs_->make<TH1F>("hgcCellData_linampl_","Cell linearized amplitudes values All", 1250, 0, 25000);
    //
    hgcCellModuleSum_        = fs_->make<TH1F>("hgcCellModuleSum","Cell sum in modules", 1000, 0., 1000.);
    triggerCellsPerModule_   = fs_->make<TH1F>("TriggerCellsPerModule","Number of trigger cells per module", 64, 0., 64.);
    triggerCellData_         = fs_->make<TH1F>("TriggerCellData","Trigger cell values", 500, 0., 500.);
    triggerCellModuleSum_    = fs_->make<TH1F>("TriggerCellModuleSum","Trigger cell sum in modules", 1000, 0., 1000.);
    //
    selectedCellsVsAllCells_ee_ = fs_->make<TH2F>("selectedCellsVsAllCells_ee","Number of selected cells vs number of cell", 128, 0, 128, 128, 0., 128.);
    energyLossVsNCells_ee_      = fs_->make<TH2F>("energyLossVsNCells_ee","Relative energy loss after selection vs number of cell", 128, 0., 128., 101, 0, 1.01);
    selectedCellsVsAllCells_fh_ = fs_->make<TH2F>("selectedCellsVsAllCells_fh","Number of selected cells vs number of cell", 128, 0, 128, 128, 0., 128.);
    energyLossVsNCells_fh_      = fs_->make<TH2F>("energyLossVsNCells_fh","Relative energy loss after selection vs number of cell", 128, 0., 128., 101, 0, 1.01);
}



/*****************************************************************/
HGCalTriggerBestChoiceTester::~HGCalTriggerBestChoiceTester() 
/*****************************************************************/
{
}

/*****************************************************************/
void HGCalTriggerBestChoiceTester::beginRun(const edm::Run& /*run*/, 
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
}

/*****************************************************************/
void HGCalTriggerBestChoiceTester::analyze(const edm::Event& e, 
                                        const edm::EventSetup& es) 
/*****************************************************************/
{
    checkSelectedCells(e, es);
    rerunBestChoiceFragments(e, es);

}

/*****************************************************************/
void HGCalTriggerBestChoiceTester::checkSelectedCells(const edm::Event& e, 
                                        const edm::EventSetup& es) 
/*****************************************************************/
{
    edm::Handle<l1t::HGCalClusterBxCollection> be_clusters_all_h;
    edm::Handle<l1t::HGCalClusterBxCollection> be_clusters_select_h;
    e.getByToken(inputbeall_,be_clusters_all_h);
    e.getByToken(inputbeselect_,be_clusters_select_h);

    const l1t::HGCalClusterBxCollection& be_clusters_all = *be_clusters_all_h;
    const l1t::HGCalClusterBxCollection& be_clusters_select = *be_clusters_select_h;

    // store trigger cells module by module. tuple = zside,subdet,layer,module
    std::map<std::tuple<uint32_t, uint32_t,uint32_t,uint32_t>, std::vector<std::pair<uint32_t,uint32_t>>> module_triggercells_all;
    for(auto cl_itr=be_clusters_all.begin(0); cl_itr!=be_clusters_all.end(0); cl_itr++)   
    {
        const l1t::HGCalCluster& cluster = *cl_itr;
        uint32_t zside = cluster.eta()<0. ? 0 : 1;
        auto itr_insert = module_triggercells_all.emplace( std::make_tuple(zside, cluster.subDet(), cluster.layer(), cluster.module()),  std::vector<std::pair<uint32_t,uint32_t>>());
        itr_insert.first->second.emplace_back(cluster.hwEta(), cluster.hwPt()); // FIXME: the index within the module has been stored in hwEta
    }
    std::map<std::tuple<uint32_t, uint32_t,uint32_t,uint32_t>, std::vector<std::pair<uint32_t,uint32_t>>> module_triggercells_select;
    for(auto cl_itr=be_clusters_select.begin(0); cl_itr!=be_clusters_select.end(0); cl_itr++)   
    {
        const l1t::HGCalCluster& cluster = *cl_itr;
        uint32_t zside = cluster.eta()<0. ? 0 : 1;
        auto itr_insert = module_triggercells_select.emplace( std::make_tuple(zside, cluster.subDet(), cluster.layer(), cluster.module()),  std::vector<std::pair<uint32_t,uint32_t>>());
        itr_insert.first->second.emplace_back(cluster.hwEta(), cluster.hwPt()); // FIXME: the index within the module has been stored in hwEta
    }

    // Compare 'all' and 'selected' trigger cells, module by module
    for(const auto& module_cells : module_triggercells_all)
    {
        const auto& module_cells_select_itr = module_triggercells_select.find(module_cells.first);
        if(module_cells_select_itr==module_triggercells_select.end())
        {
            std::cout<<"ERROR: Cannot find module for selected cells\n"; 
        }
        size_t ncells_all = module_cells.second.size();
        size_t ncells_select = module_cells_select_itr->second.size();
        uint32_t energy_all = 0;
        uint32_t energy_select = 0;
        for(const auto& id_energy : module_cells.second) energy_all += id_energy.second;
        for(const auto& id_energy : module_cells_select_itr->second) energy_select += id_energy.second;
        if(std::get<1>(module_cells.first)==ForwardSubdetector::HGCEE)
        {
            selectedCellsVsAllCells_ee_->Fill(ncells_all, ncells_select);
            if(energy_all>0) energyLossVsNCells_ee_->Fill(ncells_all, (double)energy_select/(double)energy_all);
        }
        else if(std::get<1>(module_cells.first)==ForwardSubdetector::HGCHEF) 
        {
            selectedCellsVsAllCells_fh_->Fill(ncells_all, ncells_select);
            if(energy_all>0) energyLossVsNCells_fh_->Fill(ncells_all, (double)energy_select/(double)energy_all);
        }
    }

    //std::cout<<"All trigger cells = "<<be_clusters_all.size(0)<<"\n";
    //std::cout<<"Selected trigger cells = "<<be_clusters_select.size(0)<<"\n";
}

/*****************************************************************/
void HGCalTriggerBestChoiceTester::rerunBestChoiceFragments(const edm::Event& e, 
                                        const edm::EventSetup& es) 
/*****************************************************************/
{
    // retrieve digi collections
    edm::Handle<HGCEEDigiCollection> ee_digis_h;
    edm::Handle<HGCHEDigiCollection> fh_digis_h;
    e.getByToken(inputee_,ee_digis_h);
    e.getByToken(inputfh_,fh_digis_h);

    const HGCEEDigiCollection& ee_digis = *ee_digis_h;
    const HGCHEDigiCollection& fh_digis = *fh_digis_h;

    HGCalBestChoiceDataPayload data;

    //loop on modules
    for( const auto& module : triggerGeometry_->modules() ) {        
        HGCalDetId moduleId(module.first);
        // prepare input data
        std::vector<HGCDataFrame<HGCalDetId,HGCSample>> dataframes;
        vector<pair<HGCalDetId, uint32_t > > linearized_dataframes;

        // loop over EE or FH digis and fill digis belonging to that module
        if(moduleId.subdetId()==ForwardSubdetector::HGCEE) {
            for(const auto& eedata : ee_digis) {
                if(module.second->containsCell(eedata.id())) {
                    dataframes.emplace_back(eedata.id());
                    for(int i=0; i<eedata.size(); i++) {
                        dataframes.back().setSample(i, eedata.sample(i));
                    }
                }
            }  
        }
        else if(moduleId.subdetId()==ForwardSubdetector::HGCHEF) {
            for(const auto& fhdata : fh_digis) {
                if(module.second->containsCell(fhdata.id())) {
                    dataframes.emplace_back(fhdata.id());
                    for(int i=0; i<fhdata.size(); i++) {
                        dataframes.back().setSample(i, fhdata.sample(i));
                    }
                }
            }  
        }

        //  Best choice encoding
        data.reset();
        codec_->linearize(*(module.second), dataframes, linearized_dataframes);
        codec_->triggerCellSums(*(module.second), linearized_dataframes, data);
        codec_->bestChoiceSelect(data);
        std::vector<bool> dataword = codec_->encode(data);
        HGCalBestChoiceDataPayload datadecoded = codec_->decode(dataword);
        fillModule(dataframes, datadecoded, linearized_dataframes);
    } //end loop on modules

}


/*****************************************************************/
void HGCalTriggerBestChoiceTester::fillModule( const std::vector<HGCDataFrame<HGCalDetId,HGCSample>>& dataframes,
        const HGCalBestChoiceDataPayload& fe_payload,
        const vector<pair<HGCalDetId, uint32_t > >& linearized_dataframes)
/*****************************************************************/
{
    // HGC cells part
    size_t nHGCDigi = 0;
    unsigned hgcCellModuleSum = 0;
    for(const auto& frame : dataframes)
    {
        uint32_t value = frame[2].data();
        if(value>0)
        {
            nHGCDigi++;
            hgcCellModuleSum += value;
            hgcCellData_->Fill(value);
        }
    }
    hgcCellsPerModule_->Fill(nHGCDigi);
    hgcCellModuleSum_->Fill(hgcCellModuleSum);

    for(const auto& frame : linearized_dataframes){
        hgcCellData_linampl_-> Fill(frame.second);
    }

    // trigger cells part

    size_t nFEDigi = 0;
    unsigned triggerCellModuleSum = 0;
    for(const auto& tc : fe_payload.payload)
    {
        uint32_t tcShifted = (tc<<2);
        if(tcShifted>0)
        {
            nFEDigi++;
            triggerCellModuleSum += tcShifted;
            triggerCellData_->Fill(tcShifted);
        }
    }
    triggerCellsPerModule_->Fill(nFEDigi);
    triggerCellModuleSum_->Fill(triggerCellModuleSum);
}
      
//define this as a plug-in
DEFINE_FWK_MODULE(HGCalTriggerBestChoiceTester);
