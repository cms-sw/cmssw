#include "GeneratorInterface/TauolaInterface/interface/TauSpinnerFilter.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/RandomEngineSentry.h"

TauSpinnerFilter::TauSpinnerFilter(const edm::ParameterSet& pset)
    : WTToken_(consumes<double>(pset.getParameter<edm::InputTag>("src"))), fRandomEngine(nullptr), ntaus_(0) {
  if (pset.getParameter<int>("ntaus") == 1)
    ntaus_ = 1.0;
  if (pset.getParameter<int>("ntaus") == 2)
    ntaus_ = 2.0;
}

bool TauSpinnerFilter::filter(edm::Event& e, edm::EventSetup const& es) {
  edm::RandomEngineSentry<TauSpinnerFilter> randomEngineSentry(this, e.streamID());
  const edm::Handle<double>& WT = e.getHandle(WTToken_);
  if (*(WT.product()) >= 0 && *(WT.product()) <= 4.0) {
    double weight = (*(WT.product()));
    if (fRandomEngine->flat() * ntaus_ * 2.0 < weight) {
      return true;
    }
  }
  return false;
}

DEFINE_FWK_MODULE(TauSpinnerFilter);
