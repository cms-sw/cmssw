#include "L1Trigger/L1THGCal/interface/HGCalTriggerBackendProcessor.h"

HGCalTriggerBackendProcessor::HGCalTriggerBackendProcessor(const edm::ParameterSet& conf, edm::ConsumesCollector&& cc) {
  const std::vector<edm::ParameterSet>& be_confs = conf.getParameterSetVector("algorithms");
  for (const auto& algo_cfg : be_confs) {
    const std::string& algo_name = algo_cfg.getParameter<std::string>("AlgorithmName");
    algorithms_.emplace_back(HGCalTriggerBackendAlgorithmFactory::get()->create(algo_name, algo_cfg, cc));
  }
}

void HGCalTriggerBackendProcessor::setGeometry(const HGCalTriggerGeometryBase* const geom) {
  for (const auto& algo : algorithms_) {
    algo->setGeometry(geom);
  }
}

void HGCalTriggerBackendProcessor::setProduces(edm::stream::EDProducer<>& prod) const {
  for (const auto& algo : algorithms_) {
    algo->setProduces(prod);
  }
}

void HGCalTriggerBackendProcessor::run(const l1t::HGCFETriggerDigiCollection& coll,
                                       const edm::EventSetup& es,
                                       edm::Event& e) {
  for (auto& algo : algorithms_) {
    algo->run(coll, es, e);
  }
}

void HGCalTriggerBackendProcessor::putInEvent(edm::Event& evt) {
  for (auto& algo : algorithms_) {
    algo->putInEvent(evt);
  }
}

void HGCalTriggerBackendProcessor::reset() {
  for (auto& algo : algorithms_) {
    algo->reset();
  }
}
