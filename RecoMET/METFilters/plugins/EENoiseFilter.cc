
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"


class EENoiseFilter : public edm::EDFilter {

  public:

    explicit EENoiseFilter(const edm::ParameterSet & iConfig);
    ~EENoiseFilter() override {}

  private:

    bool filter(edm::Event & iEvent, const edm::EventSetup & iSetup) override;

    edm::EDGetTokenT<EcalRecHitCollection> ebRHSrcToken_;
    edm::EDGetTokenT<EcalRecHitCollection> eeRHSrcToken_;
    const double slope_, intercept_;

    const bool taggingMode_, debug_;
};


EENoiseFilter::EENoiseFilter(const edm::ParameterSet & iConfig)
  : ebRHSrcToken_     (consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("EBRecHitSource")))
  , eeRHSrcToken_     (consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("EERecHitSource")))
  , slope_       (iConfig.getParameter<double>("Slope"))
  , intercept_   (iConfig.getParameter<double>("Intercept"))
  , taggingMode_ (iConfig.getParameter<bool>("taggingMode"))
  , debug_       (iConfig.getParameter<bool>("debug"))
{
  produces<bool>();
}


bool EENoiseFilter::filter(edm::Event & iEvent, const edm::EventSetup & iSetup) {
  edm::Handle<EcalRecHitCollection> ebRHs, eeRHs;
  iEvent.getByToken(ebRHSrcToken_, ebRHs);
  iEvent.getByToken(eeRHSrcToken_, eeRHs);

  const bool pass = eeRHs->size() < slope_ * ebRHs->size() + intercept_;

  iEvent.put(std::make_unique<bool>(pass));

  return taggingMode_ || pass;
}


#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(EENoiseFilter);
