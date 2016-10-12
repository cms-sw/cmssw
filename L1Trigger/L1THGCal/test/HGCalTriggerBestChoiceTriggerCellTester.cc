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

#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerFECodecBase.h"
#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalTriggerCellBestChoiceCodecImpl.h"
#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalTriggerCellBestChoiceCodec.h"

#include <stdlib.h> 
#include <map> 
#include "TH2.h"


class HGCalTriggerBestChoiceTriggerCellTester : public edm::EDAnalyzer 
{
    public:
        explicit HGCalTriggerBestChoiceTriggerCellTester(const edm::ParameterSet& );
        ~HGCalTriggerBestChoiceTriggerCellTester();

        virtual void beginRun(const edm::Run&, const edm::EventSetup&);
        virtual void analyze(const edm::Event&, const edm::EventSetup&);


    private:
        void checkSelectedCells(const edm::Event&, const edm::EventSetup&);
        void rerunBestChoiceFragments(const edm::Event&, const edm::EventSetup&);
        void fillModule(const std::vector<HGCDataFrame<HGCalDetId,HGCSample>>&, const std::vector<std::pair<HGCalDetId, uint32_t > >&, const HGCalTriggerCellBestChoiceDataPayload&,  const HGCalTriggerCellBestChoiceDataPayload&,const HGCalTriggerCellBestChoiceDataPayload&,   const std::map <HGCalDetId,double>&, const std::unordered_map<uint32_t, double>& );

        // inputs
        edm::EDGetToken inputee_, inputfh_, inputbh_, inputbeall_, inputbeselect_;
        bool is_Simhit_comp_;
        edm::EDGetToken SimHits_inputee_, SimHits_inputfh_, SimHits_inputbh_;
        //
        edm::ESHandle<HGCalTriggerGeometryBase> triggerGeometry_;
        std::unique_ptr<HGCalTriggerCellBestChoiceCodecImpl> codec_;
        edm::Service<TFileService> fs_;
        HGCalTriggerGeometryBase::es_info info_;

        // histos
        TH1F* hgcCellData_;
        TH1F* hgcCellData_SimHitasso_;
        TH1F* hgcCellSimHits_;
        TH2F* hgcCellData_vsSimHits_;
        TH1F* hgcCellsPerModule_;
        TH1F* hgcCellModuleSum_;
        //
        TH1F* hgcCellData_linampl_;
        TH2F* hgcCellData_linampl_vsSimHits_;
        TH2F* hgcCellData_linampl_vsSimHits_zoom_;
        TH1F* triggerCellData_noBestChoice_;
        TH2F* triggerCellData_noBestChoice_vsSimHits_;
        TH1F* triggerCellSimHits_noBestChoice_;
        TH1F* triggerCellData_BestChoice_;
        TH2F* triggerCellData_BestChoice_vsSimHits_;
        TH1F* triggerCellData_;
        TH2F* triggerCellData_vsSimHits_;
        TH1F* triggerCellsPerModule_;
        TH1F* triggerCellModuleSum_;
        //
        TH2F* selectedCellsVsAllCells_ee_; 
        TH2F* energyLossVsNCells_ee_;
        TH2F* selectedCellsVsAllCells_fh_; 
        TH2F* energyLossVsNCells_fh_;
        //

};


HGCalTriggerBestChoiceTriggerCellTester::HGCalTriggerBestChoiceTriggerCellTester(const edm::ParameterSet& conf):
    inputee_(consumes<HGCEEDigiCollection>(conf.getParameter<edm::InputTag>("eeDigis"))),
    inputfh_(consumes<HGCHEDigiCollection>(conf.getParameter<edm::InputTag>("fhDigis"))), 
    //inputbh_(consumes<HGCHEDigiCollection>(conf.getParameter<edm::InputTag>("bhDigis"))),
    inputbeall_(consumes<l1t::HGCalClusterBxCollection>(conf.getParameter<edm::InputTag>("beClustersAll"))),
    inputbeselect_(consumes<l1t::HGCalClusterBxCollection>(conf.getParameter<edm::InputTag>("beClustersSelect"))),
    is_Simhit_comp_(conf.getParameter<bool>("isSimhitComp")),
    SimHits_inputee_(consumes<edm::PCaloHitContainer>(conf.getParameter<edm::InputTag>("eeSimHits"))),
    SimHits_inputfh_(consumes<edm::PCaloHitContainer>(conf.getParameter<edm::InputTag>("fhSimHits")))
    // SimHits_inputbh_(consumes<edm::PCaloHitContainer>(conf.getParameter<edm::InputTag>("bhSimHits")))
{
    //setup FE codec
    const edm::ParameterSet& feCodecConfig = conf.getParameterSet("FECodec");
    codec_.reset( new HGCalTriggerCellBestChoiceCodecImpl(feCodecConfig) );

    // initialize output trees
    // HGC Cells
    hgcCellData_ = fs_->make<TH1F>("hgcCellData","Cell values", 1000, 0., 2000.);
    if (is_Simhit_comp_) 
    {
        hgcCellData_SimHitasso_ = fs_->make<TH1F>("hgcCellData_SimHitasso_","Cell values with an associated SimHit", 1000, 0, 2000.);
        hgcCellSimHits_ = fs_->make<TH1F>("hgcCellSimHits_","Cell simhit energies", 500, 0, 0.16);
        hgcCellData_vsSimHits_ = fs_->make<TH2F>("hgcCellData_vsSimHits_","Cell values vs simhit energies", 500,0,0.16,1000, 0., 2000.); 
    }
    hgcCellsPerModule_ = fs_->make<TH1F>("hgcCellsPerModule","Number of cells per module", 128, 0., 128.);
    hgcCellModuleSum_  = fs_->make<TH1F>("hgcCellModuleSum","Cell sum in modules", 1000, 0., 3000.);
    //
    hgcCellData_linampl_ = fs_->make<TH1F>("hgcCellData_linampl_","Cell linearized amplitudes values All", 1000, 0, 70000);
    if (is_Simhit_comp_) 
    {
        hgcCellData_linampl_vsSimHits_ = fs_->make<TH2F>("hgcCellData_linampl_vsSimHits_","Cell linearized amplitudes vs  simhit energies",500,0,0.16,1000,0,70000); 
        hgcCellData_linampl_vsSimHits_zoom_ = fs_->make<TH2F>("hgcCellData_linampl_vsSimHits_zoom_","Cell linearized amplitudes vssimhit energies, zoomed",1000,0,0.002,1000,0,1000); 
    }

    // HGC Trigger cells
    triggerCellData_noBestChoice_ = fs_->make<TH1F>("triggerCellData_noBestChoice_","Trigger cell values, no best choice", 1000, 0., 70000.);
    if (is_Simhit_comp_)
    {
        triggerCellData_noBestChoice_vsSimHits_ = fs_->make<TH2F>("triggerCellData_noBestChoice_vsSimHits_","Trigger cell values vs simhit energies, no best choice", 500,0,0.16,1000, 0., 70000.);
        triggerCellSimHits_noBestChoice_  = fs_->make<TH1F>("triggerCellSimHits_noBestChoice","Trigger cell simhit energies, no best choice", 500, 0, 0.16);
    }
    triggerCellData_BestChoice_ = fs_->make<TH1F>("triggerCellData_BestChoice_","Trigger cell values, best choice", 1000, 0., 70000.);
    if (is_Simhit_comp_) triggerCellData_BestChoice_vsSimHits_ = fs_->make<TH2F>("triggerCellData_BestChoice_vsSimHits_","Trigger cell values vs simhit energies, best choice", 500,0,0.16,1000, 0., 70000.);
    triggerCellData_  = fs_->make<TH1F>("triggerCellData","Trigger cell values", 1100, 0., 1100.);
    if (is_Simhit_comp_)
    {
        triggerCellData_vsSimHits_  = fs_->make<TH2F>("triggerCellData_vsSimHits_","Trigger cell values vs simhit energies", 500,0,0.16,1100, 0., 1100.);
    }
    triggerCellsPerModule_ = fs_->make<TH1F>("triggerCellsPerModule","Number of trigger cells per module", 64, 0., 64.);
    triggerCellModuleSum_ = fs_->make<TH1F>("TriggerCellModuleSum","Trigger cell sum in modules", 1000, 0., 10000.);
    //
    selectedCellsVsAllCells_ee_ = fs_->make<TH2F>("selectedCellsVsAllCells_ee","Number of selected cells vs number of cell", 128, 0, 128, 128, 0., 128.);
    energyLossVsNCells_ee_ = fs_->make<TH2F>("energyLossVsNCells_ee","Relative energy loss after selection vs number of cell", 128, 0., 128., 101, 0, 1.01);
    selectedCellsVsAllCells_fh_ = fs_->make<TH2F>("selectedCellsVsAllCells_fh","Number of selected cells vs number of cell", 128, 0, 128, 128, 0., 128.);
    energyLossVsNCells_fh_ = fs_->make<TH2F>("energyLossVsNCells_fh","Relative energy loss after selection vs number of cell", 128, 0., 128., 101, 0, 1.01);

}



HGCalTriggerBestChoiceTriggerCellTester::~HGCalTriggerBestChoiceTriggerCellTester() 
{
}

void HGCalTriggerBestChoiceTriggerCellTester::beginRun(const edm::Run& /*run*/, 
        const edm::EventSetup& es)
{
    es.get<IdealGeometryRecord>().get(triggerGeometry_);

    const std::string& ee_sd_name = triggerGeometry_->eeSDName();
    const std::string& fh_sd_name = triggerGeometry_->fhSDName();
    const std::string& bh_sd_name = triggerGeometry_->bhSDName();
    es.get<IdealGeometryRecord>().get(ee_sd_name,info_.geom_ee);
    es.get<IdealGeometryRecord>().get(fh_sd_name,info_.geom_fh);
    es.get<IdealGeometryRecord>().get(bh_sd_name,info_.geom_bh);
    es.get<IdealGeometryRecord>().get(ee_sd_name,info_.topo_ee);
    es.get<IdealGeometryRecord>().get(fh_sd_name,info_.topo_fh);
    es.get<IdealGeometryRecord>().get(bh_sd_name,info_.topo_bh);

}

void HGCalTriggerBestChoiceTriggerCellTester::analyze(const edm::Event& e, 
        const edm::EventSetup& es) 
{
    checkSelectedCells(e, es);
    rerunBestChoiceFragments(e, es);

}

void HGCalTriggerBestChoiceTriggerCellTester::checkSelectedCells(const edm::Event& e, 
        const edm::EventSetup& es) 
{
    edm::Handle<l1t::HGCalClusterBxCollection> be_clusters_all_h;
    edm::Handle<l1t::HGCalClusterBxCollection> be_clusters_select_h;
    e.getByToken(inputbeall_,be_clusters_all_h);
    e.getByToken(inputbeselect_,be_clusters_select_h);

    const l1t::HGCalClusterBxCollection& be_clusters_all = *be_clusters_all_h;
    const l1t::HGCalClusterBxCollection& be_clusters_select = *be_clusters_select_h;
    std::cout<<"Size all TC = "<<be_clusters_all.size(0)<<"\n";
    std::cout<<"Size selected TC = "<<be_clusters_select.size(0)<<"\n";

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
            std::cout<<" Trigger cells contained in module:\n";
            for(const auto& id_value : module_cells.second)
            {
                std::cout<<"  "<<id_value.first<<" "<<id_value.second<<"\n";
            }
            continue;
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
            if(energy_all>0 && ncells_all<12 && energy_select<energy_all)
            {
                std::cout<<" All trigger cells contained in module:\n";
                for(const auto& id_value : module_cells.second)
                {
                    std::cout<<id_value.second<<" ";
                }
                std::cout<<"\n";
                std::cout<<" Selected trigger cells contained in module:\n";
                for(const auto& id_value : module_cells_select_itr->second)
                {
                    std::cout<<id_value.second<<" ";
                }
                std::cout<<"\n";
            }
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

void HGCalTriggerBestChoiceTriggerCellTester::rerunBestChoiceFragments(const edm::Event& e, 
        const edm::EventSetup& es) 
{
    // retrieve digi collections
    edm::Handle<HGCEEDigiCollection> ee_digis_h;
    edm::Handle<HGCHEDigiCollection> fh_digis_h;
    e.getByToken(inputee_,ee_digis_h);
    e.getByToken(inputfh_,fh_digis_h);

    const HGCEEDigiCollection& ee_digis = *ee_digis_h;
    const HGCHEDigiCollection& fh_digis = *fh_digis_h;

    HGCalTriggerCellBestChoiceDataPayload data;

    // retrieve simhit collections
    std::map<HGCalDetId, double> simhit_energies;
    if (is_Simhit_comp_) {
        edm::Handle<edm::PCaloHitContainer> ee_simhits_h;
        e.getByToken(SimHits_inputee_,ee_simhits_h);
        const edm::PCaloHitContainer& ee_simhits = *ee_simhits_h;
        edm::Handle<edm::PCaloHitContainer> fh_simhits_h;
        e.getByToken(SimHits_inputfh_,fh_simhits_h);
        const edm::PCaloHitContainer& fh_simhits = *fh_simhits_h;

        // simhit/digi association EE
        HGCalDetId digiid, simid;
        int layer=0,cell=0, sec=0, subsec=0, zp=0,subdet=0;
        ForwardSubdetector mysubdet;
        HGCalDetId recoDetId ;
        int n_hits_asso=0;

        // create a map containing all simhit energies
        std::unordered_map<uint32_t, double> simhits;
        for( const auto& simhit : ee_simhits ) { 
            simid = (HGCalDetId)simhit.id();
            HGCalTestNumbering::unpackHexagonIndex(simid, subdet, zp, layer, sec, subsec, cell); 
            mysubdet = (ForwardSubdetector)(subdet);
            std::pair<int,int> recoLayerCell = info_.topo_ee->dddConstants().simToReco(cell,layer,sec,info_.topo_ee->detectorType());
            cell  = recoLayerCell.first;
            layer = recoLayerCell.second;
            if (layer<0 || cell<0) {
                continue;
            }
            recoDetId = HGCalDetId(mysubdet,zp,layer,subsec,sec,cell);
            auto itr_insert = simhits.emplace(recoDetId, 0.);
            itr_insert.first->second += simhit.energy();
        }
        // find simhit energies associated to digis
        for(const auto& data : ee_digis) {
            digiid= (HGCalDetId) data.id();
            double hit_energy=0;
            auto itr = simhits.find(digiid);
            if(itr!=simhits.end()){
                n_hits_asso++;
                hit_energy = itr->second;
            }
            simhit_energies[digiid] =  hit_energy; 
        }

        // simhit/digi association FH
        layer=0;
        cell=0;
        sec=0;
        subsec=0;
        zp=0;
        subdet=0;
        int n_hits_asso_fh=0;

        // create a map containing all simhit energies
        simhits.clear();
        for( const auto& simhit : fh_simhits ) { 
            simid = (HGCalDetId) simhit.id();
            HGCalTestNumbering::unpackHexagonIndex(simid, subdet, zp, layer, sec, subsec, cell); 
            mysubdet = (ForwardSubdetector)(subdet);
            std::pair<int,int> recoLayerCell = info_.topo_fh->dddConstants().simToReco(cell,layer,sec,info_.topo_fh->detectorType());
            cell  = recoLayerCell.first;
            layer = recoLayerCell.second;
            if (layer<0 || cell<0) {
                continue;
            }
            recoDetId = HGCalDetId(mysubdet,zp,layer,subsec,sec,cell);
            auto itr_insert = simhits.emplace(recoDetId, 0.);
            itr_insert.first->second += simhit.energy();
        }
        // find simhit energies associated to digis
        for(const auto& data : fh_digis) {
            digiid= (HGCalDetId) data.id();
            double hit_energy=0;
            auto itr = simhits.find(digiid);
            if(itr!=simhits.end()){
                n_hits_asso_fh++;
                hit_energy = itr->second;
            }
            simhit_energies[digiid] =  hit_energy; 
        }

    }
    // Find modules containing hits and prepare list of hits for each module
    std::unordered_map<uint32_t, std::vector<HGCEEDataFrame>> hit_modules_ee;
    for(const auto& eedata : ee_digis)
    {
        uint32_t module = triggerGeometry_->getModuleFromCell(eedata.id());
        auto itr_insert = hit_modules_ee.emplace(module,std::vector<HGCEEDataFrame>());
        itr_insert.first->second.push_back(eedata);
    }
    std::unordered_map<uint32_t,std::vector<HGCHEDataFrame>> hit_modules_fh;
    for(const auto& fhdata : fh_digis)
    {
        uint32_t module = triggerGeometry_->getModuleFromCell(fhdata.id());
        auto itr_insert = hit_modules_fh.emplace(module, std::vector<HGCHEDataFrame>());
        itr_insert.first->second.push_back(fhdata);
    }
    // loop on modules containing hits and call front-end processing
    for( const auto& module_hits : hit_modules_ee ) 
    {        
        // prepare input data
        std::vector<HGCDataFrame<HGCalDetId,HGCSample>> dataframes;
        std::vector<std::pair<HGCalDetId, uint32_t > > linearized_dataframes;
        // loop over EE and fill digis belonging to that module
        for(const auto& eedata : module_hits.second)
        {
            dataframes.emplace_back(eedata.id());
            for(int i=0; i<eedata.size(); i++) 
            {
                dataframes.back().setSample(i, eedata.sample(i));
            }
        }  
        // Association simhit energies with trigger cells
        std::unordered_map<uint32_t, double> TC_simhit_energies;
        if (is_Simhit_comp_) 
        {
            // need an ordered set to loop on it in the correct order
            for(const auto& tc : triggerGeometry_->getOrderedTriggerCellsFromModule(module_hits.first))
            {
                TC_simhit_energies.emplace(tc, 0);
                for(const auto& cell : triggerGeometry_->getCellsFromTriggerCell(tc))
                {
                    double simenergy = simhit_energies[cell];
                    TC_simhit_energies.at(tc)+=simenergy;
                }
            }
        }
        //  Best choice encoding
        data.reset();
        codec_->linearize(dataframes, linearized_dataframes);
        codec_->triggerCellSums(*triggerGeometry_, linearized_dataframes, data);
        HGCalTriggerCellBestChoiceDataPayload data_TCsums_woBestChoice = data;
        codec_->bestChoiceSelect(data);
        HGCalTriggerCellBestChoiceDataPayload data_TCsums_BestChoice = data;
        std::vector<bool> dataword = codec_->encode(data, *triggerGeometry_);
        HGCalTriggerCellBestChoiceDataPayload datadecoded = codec_->decode(dataword, module_hits.first, *triggerGeometry_);
        fillModule(dataframes, linearized_dataframes, data_TCsums_woBestChoice,data_TCsums_BestChoice, datadecoded,  simhit_energies, TC_simhit_energies);

    } //end loop on EE modules
    for( const auto& module_hits : hit_modules_fh ) 
    {        
        // prepare input data
        std::vector<HGCDataFrame<HGCalDetId,HGCSample>> dataframes;
        std::vector<std::pair<HGCalDetId, uint32_t > > linearized_dataframes;
        // loop over FH digis and fill digis belonging to that module
        for(const auto& fhdata : module_hits.second)
        {
            dataframes.emplace_back(fhdata.id());
            for(int i=0; i<fhdata.size(); i++) 
            {
                dataframes.back().setSample(i, fhdata.sample(i));
            }
        }  
        // Association simhit energies with trigger cells
        std::unordered_map<uint32_t, double> TC_simhit_energies;
        if (is_Simhit_comp_) 
        {
            // need an ordered set to loop on it in the correct order
            for(const auto& tc : triggerGeometry_->getOrderedTriggerCellsFromModule(module_hits.first))
            {
                TC_simhit_energies.emplace(tc, 0);
                for(const auto& cell : triggerGeometry_->getCellsFromTriggerCell(tc))
                {
                    double simenergy = simhit_energies[cell];
                    TC_simhit_energies.at(tc)+=simenergy;
                }
            }
        }
        //  Best choice encoding
        data.reset();
        codec_->linearize(dataframes, linearized_dataframes);
        codec_->triggerCellSums(*triggerGeometry_, linearized_dataframes, data);
        HGCalTriggerCellBestChoiceDataPayload data_TCsums_woBestChoice = data;
        codec_->bestChoiceSelect(data);
        HGCalTriggerCellBestChoiceDataPayload data_TCsums_BestChoice = data;
        std::vector<bool> dataword = codec_->encode(data, *triggerGeometry_);
        HGCalTriggerCellBestChoiceDataPayload datadecoded = codec_->decode(dataword, module_hits.first, *triggerGeometry_);
        fillModule(dataframes, linearized_dataframes, data_TCsums_woBestChoice,data_TCsums_BestChoice, datadecoded,  simhit_energies, TC_simhit_energies);
    } //end loop on FH modules   


}


void HGCalTriggerBestChoiceTriggerCellTester::fillModule( const std::vector<HGCDataFrame<HGCalDetId,HGCSample>>& dataframes,  const std::vector<std::pair<HGCalDetId, uint32_t > >& linearized_dataframes, const HGCalTriggerCellBestChoiceDataPayload& fe_payload_TCsums_woBestChoice, const HGCalTriggerCellBestChoiceDataPayload& fe_payload_TCsums_BestChoice, const HGCalTriggerCellBestChoiceDataPayload& fe_payload, const std::map <HGCalDetId,double>& simhit_energies, const std::unordered_map<uint32_t, double>& TC_simhit_energies)
{

    // HGC cells part
    size_t nHGCDigi = 0;
    unsigned hgcCellModuleSum = 0;
    // digis, cell based info
    for(const auto& frame : dataframes)
    {
        uint32_t value = frame[2].data();
        nHGCDigi++;
        hgcCellModuleSum += value;
        hgcCellData_->Fill(value);
        if (is_Simhit_comp_){
            double sim_energy= simhit_energies.at(frame.id());
            if (sim_energy >0){
                hgcCellData_SimHitasso_->Fill(value);
                hgcCellSimHits_->Fill(sim_energy);
                hgcCellData_vsSimHits_->Fill(sim_energy,value);
            }
        }
    }
    hgcCellsPerModule_->Fill(nHGCDigi);
    hgcCellModuleSum_->Fill(hgcCellModuleSum);

    // linearized samples, cell based info
    for(const auto& frame : linearized_dataframes){
        hgcCellData_linampl_-> Fill(frame.second);
        if (is_Simhit_comp_){
            double sim_energy= simhit_energies.at(frame.first);
            if (sim_energy >0){ 
                hgcCellData_linampl_vsSimHits_-> Fill(sim_energy,frame.second);
                hgcCellData_linampl_vsSimHits_zoom_-> Fill(sim_energy,frame.second);
            }
        }
    }

    // trigger cells part
    // after sum, no best choice, no encode/decode
    for(const auto& tc : fe_payload_TCsums_woBestChoice.payload)
    {
        if(tc.hwPt()>0)
        {
            triggerCellData_noBestChoice_->Fill(tc.hwPt());
            if (is_Simhit_comp_){
                if (TC_simhit_energies.at(tc.detId()) >0){
                    triggerCellSimHits_noBestChoice_->Fill(TC_simhit_energies.at(tc.detId()));
                    triggerCellData_noBestChoice_vsSimHits_->Fill(TC_simhit_energies.at(tc.detId()),tc.hwPt());
                }
            }
        }
    }

    // after sum, best choice, no encode/decode
    for(const auto& tc : fe_payload_TCsums_BestChoice.payload)
    {
        if(tc.hwPt()>0)
        {
            triggerCellData_BestChoice_->Fill(tc.hwPt());
            if (is_Simhit_comp_){
                if (TC_simhit_energies.at(tc.detId())>0)  triggerCellData_BestChoice_vsSimHits_->Fill(TC_simhit_energies.at(tc.detId()),tc.hwPt());
            }
        }
    }

    // after sum, best choice, encode/decode
    size_t nFEDigi = 0;
    unsigned triggerCellModuleSum = 0;
    for(const auto& tc : fe_payload.payload)
    {
        uint32_t tcShifted = (tc.hwPt()<<codec_->triggerCellTruncationBits());
        if(tc.hwPt()>0)
        {          
            nFEDigi++;
            triggerCellModuleSum += tcShifted;
            triggerCellData_->Fill(tc.hwPt());
            if (is_Simhit_comp_){
                if (TC_simhit_energies.at(tc.detId())>0)  triggerCellData_vsSimHits_->Fill(TC_simhit_energies.at(tc.detId()),tc.hwPt());
            }
        }
    }
    triggerCellsPerModule_->Fill(nFEDigi);
    triggerCellModuleSum_->Fill(triggerCellModuleSum);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalTriggerBestChoiceTriggerCellTester);
