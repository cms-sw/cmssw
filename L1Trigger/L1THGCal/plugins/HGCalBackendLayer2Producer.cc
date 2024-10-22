#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerSums.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

#include "L1Trigger/L1THGCal/interface/HGCalProcessorBase.h"

#include <memory>
#include <utility>

class HGCalBackendLayer2Producer : public edm::stream::EDProducer<> {
public:
  HGCalBackendLayer2Producer(const edm::ParameterSet&);
  ~HGCalBackendLayer2Producer() override {}

  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  // inputs
  edm::EDGetToken input_clusters_;
  edm::ESHandle<HGCalTriggerGeometryBase> triggerGeometry_;
  edm::ESGetToken<HGCalTriggerGeometryBase, CaloGeometryRecord> triggerGeomToken_;

  std::unique_ptr<HGCalBackendLayer2ProcessorBase> backendProcess_;
};

DEFINE_FWK_MODULE(HGCalBackendLayer2Producer);

HGCalBackendLayer2Producer::HGCalBackendLayer2Producer(const edm::ParameterSet& conf)
    : input_clusters_(consumes<l1t::HGCalClusterBxCollection>(conf.getParameter<edm::InputTag>("InputCluster"))),
      triggerGeomToken_(esConsumes<HGCalTriggerGeometryBase, CaloGeometryRecord, edm::Transition::BeginRun>()) {
  //setup Backend parameters
  const edm::ParameterSet& beParamConfig = conf.getParameterSet("ProcessorParameters");
  const std::string& beProcessorName = beParamConfig.getParameter<std::string>("ProcessorName");
  backendProcess_ = std::unique_ptr<HGCalBackendLayer2ProcessorBase>{
      HGCalBackendLayer2Factory::get()->create(beProcessorName, beParamConfig)};

  produces<l1t::HGCalMulticlusterBxCollection>(backendProcess_->name());
  produces<l1t::HGCalClusterBxCollection>(backendProcess_->name() + "Unclustered");
}

void HGCalBackendLayer2Producer::beginRun(const edm::Run& /*run*/, const edm::EventSetup& es) {
  triggerGeometry_ = es.getHandle(triggerGeomToken_);
  backendProcess_->setGeometry(triggerGeometry_.product());
}

void HGCalBackendLayer2Producer::produce(edm::Event& e, const edm::EventSetup& es) {
  // Output collections
  std::pair<l1t::HGCalMulticlusterBxCollection, l1t::HGCalClusterBxCollection> be_output;

  // Input collections
  edm::Handle<l1t::HGCalClusterBxCollection> trigCluster2DBxColl;

  e.getByToken(input_clusters_, trigCluster2DBxColl);

  backendProcess_->run(trigCluster2DBxColl, be_output);

  e.put(std::make_unique<l1t::HGCalMulticlusterBxCollection>(std::move(be_output.first)), backendProcess_->name());
  e.put(std::make_unique<l1t::HGCalClusterBxCollection>(std::move(be_output.second)),
        backendProcess_->name() + "Unclustered");
}
