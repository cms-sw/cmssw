#include "GeneratorInterface/GenFilters/plugins/BCToEFilter.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

BCToEFilter::BCToEFilter(const edm::ParameterSet& iConfig) {
  edm::ParameterSet filterPSet = iConfig.getParameter<edm::ParameterSet>("filterAlgoPSet");

  BCToEAlgo_.reset(new BCToEFilterAlgo(filterPSet, consumesCollector()));
}

BCToEFilter::~BCToEFilter() {}

bool BCToEFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  bool result = BCToEAlgo_->filter(iEvent);

  return result;
}
