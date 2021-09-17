/** \class EMEnrichingFilter
 *
 *  EMEnrichingFilter
 *
 * \author J Lamb, UCSB
 * this is just the wrapper around the filtering algorithm
 * found in EMEnrichingFilterAlgo
 *
 *
 ************************************************************/

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "GeneratorInterface/GenFilters/plugins/EMEnrichingFilterAlgo.h"

class EMEnrichingFilter : public edm::global::EDFilter<> {
public:
  explicit EMEnrichingFilter(const edm::ParameterSet&);

  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  EMEnrichingFilterAlgo EMEAlgo_;
};

EMEnrichingFilter::EMEnrichingFilter(const edm::ParameterSet& iConfig)
    : EMEAlgo_(iConfig.getParameter<edm::ParameterSet>("filterAlgoPSet"), consumesCollector()) {}

bool EMEnrichingFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  return EMEAlgo_.filter(iEvent, iSetup);
}

DEFINE_FWK_MODULE(EMEnrichingFilter);
