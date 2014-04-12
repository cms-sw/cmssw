#include "RecoEgamma/ElectronIdentification/interface/ElectronIDAlgo.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"

//void ElectronIDAlgo::baseSetup(const edm::ParameterSet& conf) {
//  reducedBarrelRecHitCollection_ = conf.getParameter<edm::InputTag>("reducedBarrelRecHitCollection");
//  reducedEndcapRecHitCollection_ = conf.getParameter<edm::InputTag>("reducedEndcapRecHitCollection");
//}
//
//EcalClusterLazyTools ElectronIDAlgo::getClusterShape(const edm::Event& ev, 
//                                                     const edm::EventSetup& es) 
//{
//
//  edm::Handle< EcalRecHitCollection > pEBRecHits;
//  ev.getByLabel( reducedBarrelRecHitCollection_, pEBRecHits );
//
//  edm::Handle< EcalRecHitCollection > pEERecHits;
//  ev.getByLabel( reducedEndcapRecHitCollection_, pEERecHits );
//
//  EcalClusterLazyTools lazyTools( ev, es, reducedBarrelRecHitCollection_, reducedEndcapRecHitCollection_ ) ;
//  return lazyTools ;
//
//}
