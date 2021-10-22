#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

#include "L1Trigger/L1THGCal/interface/HGCalProcessorBase.h"

#include <memory>

class HFNoseVFEProducer : public edm::stream::EDProducer<> {
public:
  HFNoseVFEProducer(const edm::ParameterSet&);
  ~HFNoseVFEProducer() override {}

  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  // inputs
  edm::EDGetToken inputnose_;
  edm::ESHandle<HGCalTriggerGeometryBase> triggerGeometry_;
  edm::ESGetToken<HGCalTriggerGeometryBase, CaloGeometryRecord> triggerGeomToken_;

  std::unique_ptr<HGCalVFEProcessorBase> vfeProcess_;
};

DEFINE_FWK_MODULE(HFNoseVFEProducer);

HFNoseVFEProducer::HFNoseVFEProducer(const edm::ParameterSet& conf)
    : inputnose_(consumes<HGCalDigiCollection>(conf.getParameter<edm::InputTag>("noseDigis"))),
      triggerGeomToken_(esConsumes<HGCalTriggerGeometryBase, CaloGeometryRecord, edm::Transition::BeginRun>()) {
  // setup VFE parameters
  const edm::ParameterSet& vfeParamConfig = conf.getParameterSet("ProcessorParameters");
  const std::string& vfeProcessorName = vfeParamConfig.getParameter<std::string>("ProcessorName");
  vfeProcess_ = std::unique_ptr<HGCalVFEProcessorBase>{
      HGCalVFEProcessorBaseFactory::get()->create(vfeProcessorName, vfeParamConfig)};

  produces<l1t::HGCalTriggerCellBxCollection>(vfeProcess_->name());
}

void HFNoseVFEProducer::beginRun(const edm::Run& /*run*/, const edm::EventSetup& es) {
  triggerGeometry_ = es.getHandle(triggerGeomToken_);
  vfeProcess_->setGeometry(triggerGeometry_.product());
}

void HFNoseVFEProducer::produce(edm::Event& e, const edm::EventSetup& es) {
  // Output collection
  auto vfe_trigcell_output = std::make_unique<l1t::HGCalTriggerCellBxCollection>();

  edm::Handle<HGCalDigiCollection> nose_digis_h;
  e.getByToken(inputnose_, nose_digis_h);

  if (nose_digis_h.isValid()) {
    const HGCalDigiCollection& nose_digis = *nose_digis_h;
    vfeProcess_->run(nose_digis, *vfe_trigcell_output);
  }

  // Put in the event
  e.put(std::move(vfe_trigcell_output), vfeProcess_->name());
}
