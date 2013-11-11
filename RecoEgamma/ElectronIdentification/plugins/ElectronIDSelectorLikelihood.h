#ifndef ElectronIDSelectorLikelihood_h
#define ElectronIDSelectorLikelihood_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "RecoEgamma/ElectronIdentification/interface/ElectronLikelihood.h"

class ElectronIDSelectorLikelihood
{
 public:

  explicit ElectronIDSelectorLikelihood (const edm::ParameterSet& conf, edm::ConsumesCollector && iC) :
    ElectronIDSelectorLikelihood(conf, iC) {}
  explicit ElectronIDSelectorLikelihood (const edm::ParameterSet& conf, edm::ConsumesCollector & iC) ;
  virtual ~ElectronIDSelectorLikelihood () ;

  void newEvent (const edm::Event&, const edm::EventSetup&) ;
  double operator() (const reco::GsfElectron&, const edm::Event&, const edm::EventSetup&) ;

 private:

  edm::ESHandle<ElectronLikelihood> likelihoodAlgo_ ;

  edm::ParameterSet conf_;

  edm::InputTag reducedBarrelRecHitCollection_;
  edm::InputTag reducedEndcapRecHitCollection_;
  edm::EDGetTokenT<EcalRecHitCollection> reducedBarrelRecHitCollectionToken_;
  edm::EDGetTokenT<EcalRecHitCollection> reducedEndcapRecHitCollectionToken_;

  bool doLikelihood_;

};
#endif
