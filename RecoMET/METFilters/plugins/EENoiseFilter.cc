
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"


class EENoiseFilter : public edm::EDFilter {

  public:

    explicit EENoiseFilter(const edm::ParameterSet & iConfig);
    ~EENoiseFilter() {}

  private:

    virtual bool filter(edm::Event & iEvent, const edm::EventSetup & iSetup);
    
    const edm::InputTag ebRHSrc_, eeRHSrc_;
    const double slope_, intercept_;

    const bool taggingMode_, debug_;
};


EENoiseFilter::EENoiseFilter(const edm::ParameterSet & iConfig)
  : ebRHSrc_     (iConfig.getParameter<edm::InputTag>("EBRecHitSource"))
  , eeRHSrc_     (iConfig.getParameter<edm::InputTag>("EERecHitSource"))
  , slope_       (iConfig.getParameter<double>("Slope"))
  , intercept_   (iConfig.getParameter<double>("Intercept"))
  , taggingMode_ (iConfig.getParameter<bool>("taggingMode"))
  , debug_       (iConfig.getParameter<bool>("debug"))
{
  produces<bool>();
}


bool EENoiseFilter::filter(edm::Event & iEvent, const edm::EventSetup & iSetup) {
  edm::Handle<EcalRecHitCollection> ebRHs, eeRHs;
  iEvent.getByLabel(ebRHSrc_, ebRHs);
  iEvent.getByLabel(eeRHSrc_, eeRHs);

  const bool pass = eeRHs->size() < slope_ * ebRHs->size() + intercept_;

  iEvent.put( std::auto_ptr<bool>(new bool(pass)) );

  return taggingMode_ || pass;
}


#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(EENoiseFilter);
