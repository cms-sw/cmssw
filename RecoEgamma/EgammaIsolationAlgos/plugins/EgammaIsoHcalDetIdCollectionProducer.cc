#include "RecoEgamma/EgammaIsolationAlgos/plugins/EgammaIsoHcalDetIdCollectionProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

EgammaIsoHcalDetIdCollectionProducer::EgammaIsoHcalDetIdCollectionProducer(const edm::ParameterSet& iConfig):
  hcalHitSelector_(iConfig.getParameter<edm::ParameterSet>("hitSelection"))
{

  recHitsToken_ = 
	  consumes<HBHERecHitCollection>(iConfig.getParameter< edm::InputTag > ("recHitsLabel"));
  elesToken_ = 
	  consumes<reco::GsfElectronCollection>(iConfig.getParameter< edm::InputTag >("elesLabel"));

  phosToken_ = 
  	  consumes<reco::PhotonCollection>(iConfig.getParameter< edm::InputTag >("phosLabel"));

  superClustersToken_ = 
          consumes<reco::SuperClusterCollection>(iConfig.getParameter< edm::InputTag >("superClustersLabel"));

  minSCEt_ = iConfig.getParameter<double>("minSCEt");
  minEleEt_ = iConfig.getParameter<double>("minEleEt");
  minPhoEt_ = iConfig.getParameter<double>("minPhoEt");
  

  interestingDetIdCollection_ = iConfig.getParameter<std::string>("interestingDetIdCollection");
  
   //register your products
  produces< DetIdCollection > (interestingDetIdCollection_) ;
}

void EgammaIsoHcalDetIdCollectionProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions)
{
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("recHitsLabel",edm::InputTag("hbhereco"));
  desc.add<edm::InputTag>("elesLabel",edm::InputTag("gedGsfElectrons"));
  desc.add<edm::InputTag>("phosLabel",edm::InputTag("gedPhotons"));
  desc.add<edm::InputTag>("superClustersLabel",edm::InputTag("particleFlowEGamma"));
  desc.add<double>("minSCEt",20);
  desc.add<double>("minEleEt",20);
  desc.add<double>("minPhoEt",20);
  desc.add<std::string>("interestingDetIdCollection","");
  desc.add<edm::ParameterSetDescription>("hitSelection",EGHcalRecHitSelector::makePSetDescription());
  descriptions.add(("interestingGedEgammaIsoHCALDetId"), desc); 
}


void EgammaIsoHcalDetIdCollectionProducer::beginRun (edm::Run const& run, const edm::EventSetup & iSetup)  
{
  hcalHitSelector_.setup(iSetup);
}

// ------------ method called to produce the data  ------------
void
EgammaIsoHcalDetIdCollectionProducer::produce (edm::Event& iEvent, 
                                const edm::EventSetup& iSetup)
{
  edm::Handle<reco::SuperClusterCollection> superClusters;
  iEvent.getByToken(superClustersToken_, superClusters);
  
  edm::Handle<reco::GsfElectronCollection> eles;
  iEvent.getByToken(elesToken_,eles);
  
  edm::Handle<reco::PhotonCollection> phos;
  iEvent.getByToken(phosToken_,phos);

  edm::Handle<HBHERecHitCollection> recHits;
  iEvent.getByToken(recHitsToken_,recHits);

  std::vector<DetId> indexToStore;
  indexToStore.reserve(100);

  if(eles.isValid() && recHits.isValid()){
    for(auto& ele : *eles){
      float scEt = ele.superCluster()->energy()*std::sin(ele.superCluster()->position().theta());
      if(scEt > minEleEt_ || ele.et()> minEleEt_){
	hcalHitSelector_.addDetIds(*ele.superCluster(),*recHits,indexToStore);
      }
    }
  }
  if(phos.isValid() && recHits.isValid()){
    for(auto& pho : *phos){
      float scEt = pho.superCluster()->energy()*std::sin(pho.superCluster()->position().theta());
      if(scEt > minPhoEt_ || pho.et()> minPhoEt_){
	hcalHitSelector_.addDetIds(*pho.superCluster(),*recHits,indexToStore);
      }
    }
  }
  if(superClusters.isValid() && recHits.isValid()){
    for(auto& sc : *superClusters){
      float scEt = sc.energy()*std::sin(sc.position().theta());
      if(scEt > minSCEt_){
	hcalHitSelector_.addDetIds(sc,*recHits,indexToStore);
      }
    }
  }
  
  //unify the vector
  std::sort(indexToStore.begin(),indexToStore.end());
  std::unique(indexToStore.begin(),indexToStore.end());
  
  auto detIdCollection = std::make_unique<DetIdCollection>(indexToStore);
   
  iEvent.put(std::move(detIdCollection), interestingDetIdCollection_ );

}
