#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1TCalorimeter/interface/CaloCluster.h"
#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"
//#include "DataFormats/L1Trigger/interface/Tau.h"

namespace l1tpf {
    class CandProducerFromStage2 : public edm::stream::EDProducer<> {
        public:
            explicit CandProducerFromStage2(const edm::ParameterSet&) ;
            ~CandProducerFromStage2() {}

        private:
            edm::EDGetTokenT<l1t::CaloClusterBxCollection> srcCluster_;
            edm::EDGetTokenT<l1t::CaloTowerBxCollection> srcTower_;
            edm::EDGetTokenT<l1t::JetBxCollection> srcJet_;
            bool isMP_;

            virtual void produce(edm::Event&, const edm::EventSetup&) override;
    }; // class
} // namespace

l1tpf::CandProducerFromStage2::CandProducerFromStage2(const edm::ParameterSet & iConfig) :
    srcCluster_(consumes<l1t::CaloClusterBxCollection>(iConfig.getParameter<edm::InputTag>("srcCluster"))),
    srcTower_(consumes<l1t::CaloTowerBxCollection>(iConfig.getParameter<edm::InputTag>("srcTower"))),
    srcJet_(consumes<l1t::JetBxCollection>(iConfig.getParameter<edm::InputTag>("srcJet"))),
    isMP_(iConfig.getParameter<bool>("MP"))
{
    produces<l1t::CaloClusterBxCollection>("CaloCluster");
    produces<l1t::CaloTowerBxCollection>("CaloTower");
    produces<l1t::JetBxCollection>("Jet");
}


void 
l1tpf::CandProducerFromStage2::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) 
{
    auto outCluster = std::make_unique<l1t::CaloClusterBxCollection>();
    edm::Handle<l1t::CaloClusterBxCollection> hCluster;
    iEvent.getByToken(srcCluster_, hCluster);
    for(auto it = hCluster->begin(0), ed = hCluster->end(0); it != ed; ++it) {
        if (it->hwPt() == 0) continue;
        l1t::CaloCluster copy(*it);
        copy.setP4(isMP_ ? l1t::CaloTools::p4MP(&copy) : l1t::CaloTools::p4Demux(&copy));
        copy.setPdgId(130);
        outCluster->push_back(0, copy);
    }
    iEvent.put(std::move(outCluster), "CaloCluster");

    auto outTower = std::make_unique<l1t::CaloTowerBxCollection>();
    edm::Handle<l1t::CaloTowerBxCollection> hTower;
    iEvent.getByToken(srcTower_, hTower);
    for(auto it = hTower->begin(0), ed = hTower->end(0); it != ed; ++it) {
        if (it->hwPt() == 0) continue;
        l1t::CaloTower copy(*it);
        copy.setP4(isMP_ ? l1t::CaloTools::p4MP(&copy) : l1t::CaloTools::p4Demux(&copy));
        copy.setPdgId(130);
        outTower->push_back(0, copy);
    }
    iEvent.put(std::move(outTower), "CaloTower");

    auto outJet = std::make_unique<l1t::JetBxCollection>();
    edm::Handle<l1t::JetBxCollection> hJet;
    iEvent.getByToken(srcJet_, hJet);
    for(auto it = hJet->begin(0), ed = hJet->end(0); it != ed; ++it) {
        //l1t::Jet copy(*it);
        //auto p4Dem = l1t::CaloTools::p4Demux(&copy);
        //auto p4MP  = l1t::CaloTools::p4MP(&copy);
        //printf("Tower  demux pt %8.3f  eta %+5.2f  phi %+5.2f    MP pt %8.3f  eta %+5.2f  phi %+5.2f   float pt %8.3f  eta %+5.2f  phi %+5.2f \n", p4Dem.Pt(), p4Dem.Eta(), p4Dem.Phi(), p4MP.Pt(), p4MP.Eta(), p4MP.Phi(), it->pt(), it->eta(), it->phi());
        if (it->pt() > 0) outJet->push_back(0, *it);
    }
    iEvent.put(std::move(outJet), "Jet");
}
using l1tpf::CandProducerFromStage2;
DEFINE_FWK_MODULE(CandProducerFromStage2);
