#include "RecoEgamma/ElectronIdentification/plugins/ElectronIDSelectorLikelihood.h"

ElectronIDSelectorLikelihood::ElectronIDSelectorLikelihood (const edm::ParameterSet& conf) : conf_ (conf) 
{
  doLikelihood_ = conf_.getParameter<bool> ("doLikelihood"); 
  reducedBarrelRecHitCollection_ = conf.getParameter<edm::InputTag>("reducedBarrelRecHitCollection");
  reducedEndcapRecHitCollection_ = conf.getParameter<edm::InputTag>("reducedEndcapRecHitCollection");

}

ElectronIDSelectorLikelihood::~ElectronIDSelectorLikelihood () 
{
}

void ElectronIDSelectorLikelihood::newEvent (const edm::Event& e, const edm::EventSetup& es)
{
  
  es.getData(likelihoodAlgo_) ;

}

double ElectronIDSelectorLikelihood::operator () (const reco::GsfElectron & ele, const edm::Event& e, const edm::EventSetup& es) 
{

  if (doLikelihood_) {
    edm::Handle< EcalRecHitCollection > pEBRecHits;
    e.getByLabel( reducedBarrelRecHitCollection_, pEBRecHits );

    edm::Handle< EcalRecHitCollection > pEERecHits;
    e.getByLabel( reducedEndcapRecHitCollection_, pEERecHits );

    EcalClusterLazyTools lazyTools( e, es, reducedBarrelRecHitCollection_, reducedEndcapRecHitCollection_ ) ;    
    return static_cast<double>(likelihoodAlgo_->resultLog (ele,lazyTools)) ;
  }
  return 0; 

}
