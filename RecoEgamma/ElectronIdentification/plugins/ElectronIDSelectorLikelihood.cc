#include "RecoEgamma/ElectronIdentification/plugins/ElectronIDSelectorLikelihood.h"

ElectronIDSelectorLikelihood::ElectronIDSelectorLikelihood (const edm::ParameterSet& conf, edm::ConsumesCollector & iC) : conf_ (conf)
{
  doLikelihood_ = conf_.getParameter<bool> ("doLikelihood");
  reducedBarrelRecHitCollection_ = conf.getParameter<edm::InputTag>("reducedBarrelRecHitCollection");
  reducedEndcapRecHitCollection_ = conf.getParameter<edm::InputTag>("reducedEndcapRecHitCollection");
  reducedBarrelRecHitCollectionToken_ = iC.consumes<EcalRecHitCollection>(reducedBarrelRecHitCollection_);
  reducedEndcapRecHitCollectionToken_ = iC.consumes<EcalRecHitCollection>(reducedEndcapRecHitCollection_);

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
    e.getByToken( reducedBarrelRecHitCollectionToken_, pEBRecHits );

    edm::Handle< EcalRecHitCollection > pEERecHits;
    e.getByToken( reducedEndcapRecHitCollectionToken_, pEERecHits );

    EcalClusterLazyTools lazyTools( e, es, reducedBarrelRecHitCollectionToken_, reducedEndcapRecHitCollectionToken_ ) ;
    return static_cast<double>(likelihoodAlgo_->resultLog (ele,lazyTools)) ;
  }
  return 0;

}
