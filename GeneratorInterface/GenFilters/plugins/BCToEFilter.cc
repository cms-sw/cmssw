/** \class BCToEFilter
 *
 *  BCToEFilter
 *
 * \author J Lamb, UCSB
 * this is just the wrapper around the filtering algorithm
 * found in BCToEFilterAlgo
 *
 ************************************************************/

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "GeneratorInterface/GenFilters/plugins/BCToEFilterAlgo.h"

class BCToEFilter : public edm::global::EDFilter<> {
public:
  explicit BCToEFilter(const edm::ParameterSet&);
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  const BCToEFilterAlgo BCToEAlgo_;
};

BCToEFilter::BCToEFilter(const edm::ParameterSet& iConfig)
    : BCToEAlgo_(iConfig.getParameter<edm::ParameterSet>("filterAlgoPSet"), consumesCollector()) {}

bool BCToEFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  return BCToEAlgo_.filter(iEvent);
}

DEFINE_FWK_MODULE(BCToEFilter);
