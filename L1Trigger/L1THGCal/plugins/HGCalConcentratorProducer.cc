#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerSums.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

#include "L1Trigger/L1THGCal/interface/HGCalProcessorBase.h"

#include <memory>

class HGCalConcentratorProducer : public edm::stream::EDProducer<> {
public:
  HGCalConcentratorProducer(const edm::ParameterSet&);
  ~HGCalConcentratorProducer() override {}

  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  // inputs
  edm::EDGetToken input_cell_, input_sums_;
  edm::ESHandle<HGCalTriggerGeometryBase> triggerGeometry_;

  std::unique_ptr<HGCalConcentratorProcessorBase> concentratorProcess_;
};

DEFINE_FWK_MODULE(HGCalConcentratorProducer);

HGCalConcentratorProducer::HGCalConcentratorProducer(const edm::ParameterSet& conf)
    : input_cell_(consumes<l1t::HGCalTriggerCellBxCollection>(conf.getParameter<edm::InputTag>("InputTriggerCells"))),
      input_sums_(consumes<l1t::HGCalTriggerSumsBxCollection>(conf.getParameter<edm::InputTag>("InputTriggerSums"))) {
  //setup Concentrator parameters
  const edm::ParameterSet& concParamConfig = conf.getParameterSet("ProcessorParameters");
  const std::string& concProcessorName = concParamConfig.getParameter<std::string>("ProcessorName");
  concentratorProcess_ = std::unique_ptr<HGCalConcentratorProcessorBase>{
      HGCalConcentratorFactory::get()->create(concProcessorName, concParamConfig)};

  produces<l1t::HGCalTriggerCellBxCollection>(concentratorProcess_->name());
  produces<l1t::HGCalTriggerSumsBxCollection>(concentratorProcess_->name());
}

void HGCalConcentratorProducer::beginRun(const edm::Run& /*run*/, const edm::EventSetup& es) {
  es.get<CaloGeometryRecord>().get(triggerGeometry_);

  concentratorProcess_->setGeometry(triggerGeometry_.product());
}

void HGCalConcentratorProducer::produce(edm::Event& e, const edm::EventSetup& es) {
  // Output collections
  auto cc_trigcell_output = std::make_unique<l1t::HGCalTriggerCellBxCollection>();
  auto cc_trigsums_output = std::make_unique<l1t::HGCalTriggerSumsBxCollection>();

  // Input collections
  edm::Handle<l1t::HGCalTriggerCellBxCollection> trigCellBxColl;

  e.getByToken(input_cell_, trigCellBxColl);
  concentratorProcess_->run(trigCellBxColl, *cc_trigcell_output, es);
  // Put in the event
  // At the moment the HGCalTriggerSumsBxCollection is empty
  e.put(std::move(cc_trigcell_output), concentratorProcess_->name());
  e.put(std::move(cc_trigsums_output), concentratorProcess_->name());
}
