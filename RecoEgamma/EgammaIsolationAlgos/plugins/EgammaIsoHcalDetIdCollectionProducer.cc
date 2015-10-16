#include "RecoEgamma/EgammaIsolationAlgos/plugins/EgammaIsoHcalDetIdCollectionProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"




#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"


EgammaIsoHcalDetIdCollectionProducer::EgammaIsoHcalDetIdCollectionProducer(const edm::ParameterSet& iConfig) 
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
  
  maxDIEta_ = iConfig.getParameter<int>("maxDIEta");
  maxDIPhi_ = iConfig.getParameter<int>("maxDIPhi");

  minEnergyHCAL_= iConfig.getParameter<double>("minEnergyHCAL");
  
   //register your products
  produces< DetIdCollection > (interestingDetIdCollection_) ;


}


void EgammaIsoHcalDetIdCollectionProducer::beginRun (edm::Run const& run, const edm::EventSetup & iSetup)  
{
   iSetup.get<HcalRecNumberingRecord>().get(towerMap_);
   //  std::cout <<" got geom "<<towerMap_.isValid()<<" "<<&(*towerMap_)<<std::endl;
}

// ------------ method called to produce the data  ------------
void
EgammaIsoHcalDetIdCollectionProducer::produce (edm::Event& iEvent, 
                                const edm::EventSetup& iSetup)
{

   // take BasicClusters
  edm::Handle<reco::SuperClusterCollection> superClusters;
  iEvent.getByToken(superClustersToken_, superClusters);
  
  edm::Handle<reco::GsfElectronCollection> eles;
  iEvent.getByToken(elesToken_,eles);
  
  edm::Handle<reco::PhotonCollection> phos;
  iEvent.getByToken(phosToken_,phos);

  edm::Handle<HBHERecHitCollection> recHits;
  iEvent.getByToken(recHitsToken_,recHits);

  //Create empty output collections
  std::vector<DetId> indexToStore;
  indexToStore.reserve(100);

  if(eles.isValid() && recHits.isValid()){
    for(auto& ele : *eles){
   
      float scEt = ele.superCluster()->energy()*std::sin(ele.superCluster()->position().theta());
      if(scEt > minEleEt_ || ele.et()> minEleEt_) addDetIds(*ele.superCluster(),*recHits,indexToStore);
    }
  }
  if(phos.isValid() && recHits.isValid()){
    for(auto& pho : *phos){
      float scEt = pho.superCluster()->energy()*std::sin(pho.superCluster()->position().theta());
      if(scEt > minPhoEt_ || pho.et()> minPhoEt_) addDetIds(*pho.superCluster(),*recHits,indexToStore);
    }
  }
  if(superClusters.isValid() && recHits.isValid()){
    for(auto& sc : *superClusters){
      float scEt = sc.energy()*std::sin(sc.position().theta());
      if(scEt > minSCEt_) addDetIds(sc,*recHits,indexToStore);
    }
  }
  
  //unify the vector
  std::sort(indexToStore.begin(),indexToStore.end());
  std::unique(indexToStore.begin(),indexToStore.end());
  
  std::auto_ptr< DetIdCollection > detIdCollection (new DetIdCollection(indexToStore) ) ;
   
  iEvent.put( detIdCollection, interestingDetIdCollection_ );

}

//some nasty hardcoded badness
int calDIEta(int iEta1,int iEta2)
{
  
  int dEta = iEta1-iEta2;
  if(iEta1*iEta2<0) {//-ve to +ve transistion and no crystal at zero
    if(dEta<0) dEta++;
    else dEta--;
  }
  return dEta;
}

//some nasty hardcoded badness
int calDIPhi(int iPhi1,int iPhi2)
{

  int dPhi = iPhi1-iPhi2;

  if(dPhi>72/2) dPhi-=72;
  else if(dPhi<-72/2) dPhi+=72;
  
  return dPhi;

}


void
EgammaIsoHcalDetIdCollectionProducer::addDetIds(const reco::SuperCluster& superClus,const HBHERecHitCollection& recHits,std::vector<DetId>& detIdsToStore)
{
  DetId seedId = superClus.seed()->seed();
  if(seedId.det() != DetId::Ecal) {
    edm::LogError("EgammaIsoHcalDetIdCollectionProducerError") << "Somehow the supercluster has a seed which is not ECAL, something is badly wrong";
  }
  //so we are using CaloTowers to get the iEta/iPhi of the HCAL rec hit behind the seed cluster, this might go funny on tower 28 but shouldnt matter there
 
  CaloTowerDetId towerId(towerMap_->towerOf(seedId)); 
  int seedHcalIEta = towerId.ieta();
  int seedHcalIPhi = towerId.iphi();

  for(auto& recHit : recHits){
    int dIEtaAbs = std::abs(calDIEta(seedHcalIEta,recHit.id().ieta()));
    int dIPhiAbs = std::abs(calDIPhi(seedHcalIPhi,recHit.id().iphi()));
  
    if(dIEtaAbs<=maxDIEta_ && dIPhiAbs<=maxDIPhi_ && recHit.energy()>minEnergyHCAL_) detIdsToStore.push_back(recHit.id().rawId());
  }

}
