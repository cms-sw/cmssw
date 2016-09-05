#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/CaloTest/interface/HGCalTestNumbering.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
//#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCTriggerDetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerFECodecBase.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerBackendProcessor.h"
#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalBestChoiceCodecImpl.h"

#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"


#include "TTree.h"

#include <memory>
#include <fstream>
#include <chrono>
#include <functional>


class HGCalTriggerBestChoiceMonitor : public edm::EDAnalyzer
{  
    public:    
        HGCalTriggerBestChoiceMonitor(const edm::ParameterSet&);
        ~HGCalTriggerBestChoiceMonitor() { }

        virtual void beginRun(const edm::Run&, const edm::EventSetup&);
        virtual void analyze(const edm::Event&, const edm::EventSetup&);

    private:
        void clear_module_loop();
        void clear_optimized_loop();
        void clear_summary();
        //void moduleLoop(const edm::Event& e, const edm::EventSetup& es);
        void optimizedLoop(const edm::Event& e, const edm::EventSetup& es);
        void lightweightLoop(const edm::Event& e, const edm::EventSetup& es);

        // inputs
        HGCalTriggerGeometryBase::es_info es_info_;
        edm::EDGetToken inputgen_, inputee_, inputfh_, inputbh_;
        edm::EDGetToken inputee_simhits_, inputfh_simhits_;
        // tools
        std::unique_ptr<HGCalTriggerGeometryBase> triggerGeometry_;
        std::unique_ptr<HGCalTriggerGeometryBase> triggerLightweightGeometry_;
        std::unique_ptr<HGCalBestChoiceCodecImpl> codec_;
        // Output
        edm::Service<TFileService> fs_;
        TTree* tree_module_loop_;
        TTree* tree_optimized_loop_;
        //TTree* tree_lightweight_optimized_loop_;
        TTree* tree_summary_;
        int run_;
        int event_;
        int lumi_;
        // module loop
        int module_cell_selection_time_;
        int module_processing_time_;
        // optimized loop
        int cell_module_selection_time_;
        int module_cell_optimized_selection_time_;
        // summary timing
        int module_loop_total_time_;
        int optimized_loop_total_time_;
        int lightweight_loop_total_time_;

};

DEFINE_FWK_MODULE(HGCalTriggerBestChoiceMonitor);

/*****************************************************************/
HGCalTriggerBestChoiceMonitor::HGCalTriggerBestChoiceMonitor(const edm::ParameterSet& conf):
    inputee_ (consumes<HGCEEDigiCollection>(conf.getParameter<edm::InputTag>("eeDigis"))),
    inputfh_ (consumes<HGCHEDigiCollection>(conf.getParameter<edm::InputTag>("fhDigis")))
/*****************************************************************/
{
    //setup geometry configuration
    const edm::ParameterSet& geometryConfig = conf.getParameterSet("TriggerGeometry");
    const std::string& trigGeomName = geometryConfig.getParameter<std::string>("TriggerGeometryName");
    HGCalTriggerGeometryBase* geometry = HGCalTriggerGeometryFactory::get()->create(trigGeomName,geometryConfig);
    triggerGeometry_.reset(geometry);

    //setup lightweight geometry configuration
    const edm::ParameterSet& lightweightGeometryConfig = conf.getParameterSet("TriggerLightweightGeometry");
    const std::string& trigLightGeomName = lightweightGeometryConfig.getParameter<std::string>("TriggerGeometryName");
    HGCalTriggerGeometryBase* lightweightGeometry = HGCalTriggerGeometryFactory::get()->create(trigLightGeomName,lightweightGeometryConfig);
    triggerLightweightGeometry_.reset(lightweightGeometry);

    //setup FE codec
    const edm::ParameterSet& feCodecConfig = conf.getParameterSet("FECodec");
    codec_.reset( new HGCalBestChoiceCodecImpl(feCodecConfig) );


    tree_module_loop_ = fs_->make<TTree>("Tree_Module_Loop","Time monitoring tree");
    tree_optimized_loop_ = fs_->make<TTree>("Tree_Optimized_Loop","Time monitoring tree");
    tree_summary_ = fs_->make<TTree>("Tree_Summary","Time monitoring tree");


    tree_module_loop_->Branch("run", &run_, "run/I");
    tree_module_loop_->Branch("event", &event_, "event/I");
    tree_module_loop_->Branch("lumi", &lumi_, "lumi/I");
    tree_module_loop_->Branch("module_cell_selection_time", &module_cell_selection_time_, "module_cell_selection_time/I");
    tree_module_loop_->Branch("module_processing_time", &module_processing_time_, "module_processing_time/I");


    tree_optimized_loop_->Branch("run", &run_, "run/I");
    tree_optimized_loop_->Branch("event", &event_, "event/I");
    tree_optimized_loop_->Branch("lumi", &lumi_, "lumi/I");
    //tree_module_loop_->Branch("cell_module_selection_time", &cell_module_selection_time_, "cell_module_selection_time/I");
    tree_optimized_loop_->Branch("module_cell_optimized_selection_time", &module_cell_optimized_selection_time_, "module_cell_optimized_selection_time/I");


    tree_summary_->Branch("run", &run_, "run/I");
    tree_summary_->Branch("event", &event_, "event/I");
    tree_summary_->Branch("lumi", &lumi_, "lumi/I");
    tree_summary_->Branch("module_loop_total_time", &module_loop_total_time_, "module_loop_total_time/I");
    tree_summary_->Branch("optimized_loop_total_time", &optimized_loop_total_time_, "optimized_loop_total_time/I");
    tree_summary_->Branch("lightweight_loop_total_time", &lightweight_loop_total_time_, "lightweight_loop_total_time/I");

}

/*****************************************************************/
void HGCalTriggerBestChoiceMonitor::beginRun(const edm::Run& /*run*/, const edm::EventSetup& es) 
/*****************************************************************/
{
    triggerGeometry_->reset();
    const std::string& ee_sd_name = triggerGeometry_->eeSDName();
    const std::string& fh_sd_name = triggerGeometry_->fhSDName();
    const std::string& bh_sd_name = triggerGeometry_->bhSDName();
    es.get<IdealGeometryRecord>().get(ee_sd_name,es_info_.geom_ee);
    es.get<IdealGeometryRecord>().get(fh_sd_name,es_info_.geom_fh);
    es.get<IdealGeometryRecord>().get(bh_sd_name,es_info_.geom_bh);
    es.get<IdealGeometryRecord>().get(ee_sd_name,es_info_.topo_ee);
    es.get<IdealGeometryRecord>().get(fh_sd_name,es_info_.topo_fh);
    es.get<IdealGeometryRecord>().get(bh_sd_name,es_info_.topo_bh);
    triggerGeometry_->initialize(es_info_);

    triggerLightweightGeometry_->reset();
    triggerLightweightGeometry_->initialize(es_info_);
}


void HGCalTriggerBestChoiceMonitor::analyze(const edm::Event& e, const edm::EventSetup& es) 
{

    clear_summary();

    run_    = e.id().run();
    lumi_   = e.luminosityBlock();
    event_  = e.id().event();

    std::chrono::steady_clock::time_point begin_time;
    std::chrono::steady_clock::time_point end_time;

    begin_time = std::chrono::steady_clock::now();
    //moduleLoop(e, es);  
    end_time = std::chrono::steady_clock::now();
    module_loop_total_time_ = std::chrono::duration_cast<std::chrono::microseconds>(end_time - begin_time).count();

    begin_time = std::chrono::steady_clock::now();
    optimizedLoop(e, es);
    end_time = std::chrono::steady_clock::now();
    optimized_loop_total_time_ = std::chrono::duration_cast<std::chrono::microseconds>(end_time - begin_time).count();

    begin_time = std::chrono::steady_clock::now();
    lightweightLoop(e, es);
    end_time = std::chrono::steady_clock::now();
    lightweight_loop_total_time_ = std::chrono::duration_cast<std::chrono::microseconds>(end_time - begin_time).count();

    tree_summary_->Fill();

}

//void HGCalTriggerBestChoiceMonitor::moduleLoop(const edm::Event& e, const edm::EventSetup& es) 
//{
    //std::cout<<"Old module loop\n";

    //clear_module_loop();

    //run_    = e.id().run();
    //lumi_   = e.luminosityBlock();
    //event_  = e.id().event();

    //edm::Handle<HGCEEDigiCollection> ee_digis_h;
    //edm::Handle<HGCHEDigiCollection> fh_digis_h;
    //e.getByToken(inputee_,ee_digis_h);
    //e.getByToken(inputfh_,fh_digis_h);

    //const HGCEEDigiCollection& ee_digis = *ee_digis_h;
    //const HGCHEDigiCollection& fh_digis = *fh_digis_h;


    //std::chrono::steady_clock::time_point begin_time;
    //std::chrono::steady_clock::time_point end_time;

    //HGCalBestChoiceDataPayload data;

    ////loop on modules
    //for( const auto& module : triggerGeometry_->modules() ) 
    //{        
        //clear_module_loop();
        //run_    = e.id().run();
        //lumi_   = e.luminosityBlock();
        //event_  = e.id().event();
        //HGCalDetId moduleId(module.first);
        //// prepare input data
        //std::vector<HGCDataFrame<HGCalDetId,HGCSample>> dataframes;
        //std::vector<std::pair<HGCalDetId, uint32_t > > linearized_dataframes;

        //begin_time = std::chrono::steady_clock::now();

        //// loop over EE or FH digis and fill digis belonging to that module
        //if(moduleId.subdetId()==ForwardSubdetector::HGCEE) 
        //{
            //for(const auto& eedata : ee_digis)
            //{
                //if(module.second->containsCell(eedata.id())) 
                //{
                    //dataframes.emplace_back(eedata.id());
                    //for(int i=0; i<eedata.size(); i++) 
                    //{
                        //dataframes.back().setSample(i, eedata.sample(i));
                    //}
                //}
            //}  
        //}
        //else if(moduleId.subdetId()==ForwardSubdetector::HGCHEF) 
        //{
            //for(const auto& fhdata : fh_digis) 
            //{
                //if(module.second->containsCell(fhdata.id())) 
                //{
                    //dataframes.emplace_back(fhdata.id());
                    //for(int i=0; i<fhdata.size(); i++) 
                    //{
                        //dataframes.back().setSample(i, fhdata.sample(i));
                    //}
                //}
            //}  
        //}
        //end_time = std::chrono::steady_clock::now();
        //module_cell_selection_time_ = std::chrono::duration_cast<std::chrono::microseconds>(end_time - begin_time).count();

        ////  Best choice encoding
        //begin_time = std::chrono::steady_clock::now();
        //data.reset();
        //codec_->linearize(*(module.second), dataframes, linearized_dataframes);
        //codec_->triggerCellSums(*(module.second), linearized_dataframes, data);
        //codec_->bestChoiceSelect(data);
        //end_time = std::chrono::steady_clock::now();
        //module_processing_time_ = std::chrono::duration_cast<std::chrono::microseconds>(end_time - begin_time).count();

        //tree_module_loop_->Fill();

    //} //end loop on modules
    


//}


void HGCalTriggerBestChoiceMonitor::optimizedLoop(const edm::Event& e, const edm::EventSetup& es) 
{
    std::cout<<"Optimized loop\n";

    clear_optimized_loop();

    run_    = e.id().run();
    lumi_   = e.luminosityBlock();
    event_  = e.id().event();

    edm::Handle<HGCEEDigiCollection> ee_digis_h;
    edm::Handle<HGCHEDigiCollection> fh_digis_h;
    e.getByToken(inputee_,ee_digis_h);
    e.getByToken(inputfh_,fh_digis_h);

    const HGCEEDigiCollection& ee_digis = *ee_digis_h;
    const HGCHEDigiCollection& fh_digis = *fh_digis_h;


    std::chrono::steady_clock::time_point begin_time;
    std::chrono::steady_clock::time_point end_time;

    HGCalBestChoiceDataPayload data;

    std::unordered_map<uint32_t, std::vector<HGCEEDataFrame>> hit_modules_ee;
    for(const auto& eedata : ee_digis)
    {
        const unsigned module = triggerGeometry_->getModuleFromCell(eedata.id());
        auto itr_insert = hit_modules_ee.insert(std::make_pair(module,std::vector<HGCEEDataFrame>()));
        itr_insert.first->second.push_back(eedata);
    }
    std::unordered_map<uint32_t,std::vector<HGCHEDataFrame>> hit_modules_fh;
    for(const auto& fhdata : fh_digis)
    {
        const unsigned module = triggerGeometry_->getModuleFromCell(fhdata.id());
        auto itr_insert = hit_modules_fh.insert(std::make_pair(module, std::vector<HGCHEDataFrame>()));
        itr_insert.first->second.push_back(fhdata);
    }
    //loop on modules
    for( const auto& module_hits : hit_modules_ee ) 
    {        
        //clear_optimized_loop();
        run_    = e.id().run();
        lumi_   = e.luminosityBlock();
        event_  = e.id().event();
        HGCTriggerHexDetId moduleId(module_hits.first);
        // prepare input data
        std::vector<HGCDataFrame<HGCalDetId,HGCSample>> dataframes;
        std::vector<std::pair<HGCalDetId, uint32_t > > linearized_dataframes;

        begin_time = std::chrono::steady_clock::now();

        // loop over EE and fill digis belonging to that module
        if(moduleId.subdetId()==ForwardSubdetector::HGCEE) 
        {
            for(const auto& eedata : module_hits.second)
            {
                dataframes.emplace_back(eedata.id());
                for(int i=0; i<eedata.size(); i++) 
                {
                    dataframes.back().setSample(i, eedata.sample(i));
                }
            }  
        }

        end_time = std::chrono::steady_clock::now();
        module_cell_optimized_selection_time_ = std::chrono::duration_cast<std::chrono::microseconds>(end_time - begin_time).count();

        //  Best choice encoding
        begin_time = std::chrono::steady_clock::now();
        data.reset();
        codec_->linearize(dataframes, linearized_dataframes);
        codec_->triggerCellSums(*triggerGeometry_, linearized_dataframes, data);
        codec_->bestChoiceSelect(data);
        end_time = std::chrono::steady_clock::now();
        module_processing_time_ = std::chrono::duration_cast<std::chrono::microseconds>(end_time - begin_time).count();

        //tree_optimized_loop_->Fill();

    } //end loop on EE modules
    for( const auto& module_hits : hit_modules_fh ) 
    {        
        clear_optimized_loop();
        run_    = e.id().run();
        lumi_   = e.luminosityBlock();
        event_  = e.id().event();
        HGCTriggerHexDetId moduleId(module_hits.first);
        // prepare input data
        std::vector<HGCDataFrame<HGCalDetId,HGCSample>> dataframes;
        std::vector<std::pair<HGCalDetId, uint32_t > > linearized_dataframes;

        begin_time = std::chrono::steady_clock::now();

        // loop over FH digis and fill digis belonging to that module
        if(moduleId.subdetId()==ForwardSubdetector::HGCHEF) 
        {
            for(const auto& fhdata : module_hits.second)
            {
                dataframes.emplace_back(fhdata.id());
                for(int i=0; i<fhdata.size(); i++) 
                {
                    dataframes.back().setSample(i, fhdata.sample(i));
                }
            }  
        }

        end_time = std::chrono::steady_clock::now();
        module_cell_optimized_selection_time_ = std::chrono::duration_cast<std::chrono::microseconds>(end_time - begin_time).count();

        //  Best choice encoding
        begin_time = std::chrono::steady_clock::now();
        data.reset();
        codec_->linearize(dataframes, linearized_dataframes);
        codec_->triggerCellSums(*triggerGeometry_, linearized_dataframes, data);
        codec_->bestChoiceSelect(data);
        end_time = std::chrono::steady_clock::now();
        module_processing_time_ = std::chrono::duration_cast<std::chrono::microseconds>(end_time - begin_time).count();

        //tree_optimized_loop_->Fill();

    } //end loop on FH modules   

}


void HGCalTriggerBestChoiceMonitor::lightweightLoop(const edm::Event& e, const edm::EventSetup& es) 
{
    std::cout<<"Lightweight loop\n";

    //clear_optimized_loop();

    run_    = e.id().run();
    lumi_   = e.luminosityBlock();
    event_  = e.id().event();

    edm::Handle<HGCEEDigiCollection> ee_digis_h;
    edm::Handle<HGCHEDigiCollection> fh_digis_h;
    e.getByToken(inputee_,ee_digis_h);
    e.getByToken(inputfh_,fh_digis_h);

    const HGCEEDigiCollection& ee_digis = *ee_digis_h;
    const HGCHEDigiCollection& fh_digis = *fh_digis_h;


    std::chrono::steady_clock::time_point begin_time;
    std::chrono::steady_clock::time_point end_time;

    HGCalBestChoiceDataPayload data;

    std::unordered_map<uint32_t, std::vector<HGCEEDataFrame>> hit_modules_ee;
    for(const auto& eedata : ee_digis)
    {
        const unsigned module = triggerLightweightGeometry_->getModuleFromCell(eedata.id());
        auto itr_insert = hit_modules_ee.insert(std::make_pair(module,std::vector<HGCEEDataFrame>()));
        itr_insert.first->second.push_back(eedata);
    }
    std::unordered_map<uint32_t,std::vector<HGCHEDataFrame>> hit_modules_fh;
    for(const auto& fhdata : fh_digis)
    {
        const unsigned module = triggerLightweightGeometry_->getModuleFromCell(fhdata.id());
        auto itr_insert = hit_modules_fh.insert(std::make_pair(module, std::vector<HGCHEDataFrame>()));
        itr_insert.first->second.push_back(fhdata);
    }
    //loop on modules
    for( const auto& module_hits : hit_modules_ee ) 
    {        
        //clear_optimized_loop();
        run_    = e.id().run();
        lumi_   = e.luminosityBlock();
        event_  = e.id().event();
        HGCTriggerHexDetId moduleId(module_hits.first);
        // prepare input data
        std::vector<HGCDataFrame<HGCalDetId,HGCSample>> dataframes;
        std::vector<std::pair<HGCalDetId, uint32_t > > linearized_dataframes;

        begin_time = std::chrono::steady_clock::now();

        // loop over EE and fill digis belonging to that module
        if(moduleId.subdetId()==ForwardSubdetector::HGCEE) 
        {
            for(const auto& eedata : module_hits.second)
            {
                dataframes.emplace_back(eedata.id());
                for(int i=0; i<eedata.size(); i++) 
                {
                    dataframes.back().setSample(i, eedata.sample(i));
                }
            }  
        }

        end_time = std::chrono::steady_clock::now();
        module_cell_optimized_selection_time_ = std::chrono::duration_cast<std::chrono::microseconds>(end_time - begin_time).count();

        //  Best choice encoding
        begin_time = std::chrono::steady_clock::now();
        data.reset();
        codec_->linearize(dataframes, linearized_dataframes);
        codec_->triggerCellSums(*triggerLightweightGeometry_, linearized_dataframes, data);
        codec_->bestChoiceSelect(data);
        end_time = std::chrono::steady_clock::now();
        module_processing_time_ = std::chrono::duration_cast<std::chrono::microseconds>(end_time - begin_time).count();

        //tree_optimized_loop_->Fill();

    } //end loop on EE modules
    for( const auto& module_hits : hit_modules_fh ) 
    {        
        clear_optimized_loop();
        run_    = e.id().run();
        lumi_   = e.luminosityBlock();
        event_  = e.id().event();
        HGCTriggerHexDetId moduleId(module_hits.first);
        // prepare input data
        std::vector<HGCDataFrame<HGCalDetId,HGCSample>> dataframes;
        std::vector<std::pair<HGCalDetId, uint32_t > > linearized_dataframes;

        begin_time = std::chrono::steady_clock::now();

        // loop over FH digis and fill digis belonging to that module
        if(moduleId.subdetId()==ForwardSubdetector::HGCHEF) 
        {
            for(const auto& fhdata : module_hits.second)
            {
                dataframes.emplace_back(fhdata.id());
                for(int i=0; i<fhdata.size(); i++) 
                {
                    dataframes.back().setSample(i, fhdata.sample(i));
                }
            }  
        }

        end_time = std::chrono::steady_clock::now();
        module_cell_optimized_selection_time_ = std::chrono::duration_cast<std::chrono::microseconds>(end_time - begin_time).count();

        //  Best choice encoding
        begin_time = std::chrono::steady_clock::now();
        data.reset();
        codec_->linearize(dataframes, linearized_dataframes);
        codec_->triggerCellSums(*triggerLightweightGeometry_, linearized_dataframes, data);
        codec_->bestChoiceSelect(data);
        end_time = std::chrono::steady_clock::now();
        module_processing_time_ = std::chrono::duration_cast<std::chrono::microseconds>(end_time - begin_time).count();

        //tree_optimized_loop_->Fill();

    } //end loop on FH modules   


}




/*****************************************************************/
void HGCalTriggerBestChoiceMonitor::clear_module_loop()
/*****************************************************************/
{
    run_ = 0;
    event_ = 0;
    lumi_ = 0;
    module_cell_selection_time_ = 0;
    module_processing_time_ = 0;
}

/*****************************************************************/
void HGCalTriggerBestChoiceMonitor::clear_optimized_loop()
/*****************************************************************/
{
    run_ = 0;
    event_ = 0;
    lumi_ = 0;
    cell_module_selection_time_ = 0;
    module_cell_optimized_selection_time_ = 0;
}

/*****************************************************************/
void HGCalTriggerBestChoiceMonitor::clear_summary()
/*****************************************************************/
{
    run_ = 0;
    event_ = 0;
    lumi_ = 0;
    module_loop_total_time_ = 0;
    optimized_loop_total_time_ = 0;
    lightweight_loop_total_time_ = 0;
}


