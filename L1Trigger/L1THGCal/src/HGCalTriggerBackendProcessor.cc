#include "L1Trigger/L1THGCal/interface/HGCalTriggerBackendProcessor.h"

HGCalTriggerBackendProcessor::
HGCalTriggerBackendProcessor(const edm::ParameterSet& conf) {
  const std::vector<edm::ParameterSet> be_confs = 
    conf.getParameterSetVector("algorithms");
  for( const auto& algo_cfg : be_confs ) {
    const std::string& algo_name = 
      algo_cfg.getParameter<std::string>("AlgorithmName");
    HGCalTriggerBackendAlgorithmBase* algo = 
      HGCalTriggerBackendAlgorithmFactory::get()->create(algo_name,algo_cfg);
    algorithms_.emplace_back(algo);
  }
}

void HGCalTriggerBackendProcessor::setGeometry(const HGCalTriggerGeometryBase* const geom) {
  for( const auto& algo : algorithms_ ) {
    algo->setGeometry(geom);
  }
}


void HGCalTriggerBackendProcessor::setProduces(edm::EDProducer& prod) const {
  for( const auto& algo : algorithms_ ) {
    algo->setProduces(prod);
  }
}

void HGCalTriggerBackendProcessor::
run(const l1t::HGCFETriggerDigiCollection& coll, const edm::EventSetup& es) {
  for( auto& algo : algorithms_ ) {
    algo->run(coll, es);
  }
}

void HGCalTriggerBackendProcessor::putInEvent(edm::Event& evt) {
  for( auto& algo : algorithms_ ) {
    algo->putInEvent(evt);
  }
}

void HGCalTriggerBackendProcessor::reset() {
  for( auto& algo : algorithms_ ) {
    algo->reset();
  }
}

