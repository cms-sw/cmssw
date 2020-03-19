//
// Original Author:  Filippo Ambroglini
//         Created:  Fri Sep 29 17:10:41 CEST 2006
//

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "CLHEP/Random/RandomEngine.h"
#include "CLHEP/Random/RandFlat.h"

class MinimumBiasFilter : public edm::global::EDFilter<> {
public:
  MinimumBiasFilter(const edm::ParameterSet&);

  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  const double theEventFraction;
};

MinimumBiasFilter::MinimumBiasFilter(const edm::ParameterSet& iConfig)
    : theEventFraction(iConfig.getUntrackedParameter<double>("EventFraction")) {}

bool MinimumBiasFilter::filter(edm::StreamID streamID, edm::Event&, const edm::EventSetup&) const {
  /**
   * Wainting the real trigger for the
   * MB we have developped
   * a random one
  */
  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine& engine = rng->getEngine(streamID);
  double rnd = CLHEP::RandFlat::shoot(&engine, 0., 1.);

  return (rnd <= theEventFraction);
}

DEFINE_FWK_MODULE(MinimumBiasFilter);
