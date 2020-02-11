#include "GeneratorInterface/GenFilters/plugins/BCToEFilter.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

BCToEFilter::BCToEFilter(const edm::ParameterSet& iConfig)
    : BCToEAlgo_(iConfig.getParameter<edm::ParameterSet>("filterAlgoPSet"), consumesCollector()) {}

bool BCToEFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  return BCToEAlgo_.filter(iEvent);
}
