#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
//#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "FWCore/Framework/interface/EventSetup.h"
//#include "FWCore/Framework/interface/ESTransientHandle.h"
//#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerNtupleBase.h"
//#include "TTree.h"
//#include <iostream>
//#include <string>
//#include <vector>

class HGCalTriggerNtupleManager : public edm::EDAnalyzer 
{
    public:
        typedef std::unique_ptr<HGCalTriggerNtupleBase> ntuple_ptr;

    public:
        explicit HGCalTriggerNtupleManager(const edm::ParameterSet& conf);
        ~HGCalTriggerNtupleManager(){};
        virtual void beginRun(const edm::Run&, const edm::EventSetup&) {};
        virtual void analyze(const edm::Event&, const edm::EventSetup&);

    private:
        edm::Service<TFileService> file_service_;
        std::vector<ntuple_ptr> hgc_ntuples_;
        TTree* tree_;
};


DEFINE_FWK_MODULE(HGCalTriggerNtupleManager);


HGCalTriggerNtupleManager::
HGCalTriggerNtupleManager(const edm::ParameterSet& conf) 
{
    tree_ = file_service_->make<TTree>("HGCalTriggerNtuple","HGCalTriggerNtuple");    
    const std::vector<edm::ParameterSet> ntuple_cfgs = conf.getParameterSetVector("Ntuples");
    for(const auto& ntuple_cfg : ntuple_cfgs) 
    {
        const std::string& ntuple_name = ntuple_cfg.getParameter<std::string>("NtupleName");
        hgc_ntuples_.emplace_back( HGCalTriggerNtupleFactory::get()->create(ntuple_name, ntuple_cfg) );
        hgc_ntuples_.back()->initialize(*tree_, ntuple_cfg , consumesCollector());
    }
}


void 
HGCalTriggerNtupleManager::
analyze(const edm::Event& e, const edm::EventSetup& es) 
{
    for(auto& hgc_ntuple : hgc_ntuples_)
    {
        hgc_ntuple->fill(e,es);
    }
    tree_->Fill();
}

