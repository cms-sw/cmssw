#include "RecoEgamma/EgammaIsolationAlgos/plugins/EgammaIsoESDetIdCollectionProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"


EgammaIsoESDetIdCollectionProducer::EgammaIsoESDetIdCollectionProducer(const edm::ParameterSet& iConfig) 
{

  eeClusToESMapToken_ = 
    consumes<reco::PFCluster::EEtoPSAssociation>(iConfig.getParameter< edm::InputTag > ("eeClusToESMapLabel"));
  
  ecalPFClustersToken_ =
    consumes<reco::PFClusterCollection>(iConfig.getParameter<edm::InputTag>("ecalPFClustersLabel"));
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
  
  maxDR_ = iConfig.getParameter<double>("maxDR");

   //register your products
  produces< DetIdCollection > (interestingDetIdCollection_) ;


}


void EgammaIsoESDetIdCollectionProducer::beginRun (edm::Run const& run, const edm::EventSetup & iSetup)  
{
  
}

// ------------ method called to produce the data  ------------
void
EgammaIsoESDetIdCollectionProducer::produce (edm::Event& iEvent, 
                                const edm::EventSetup& iSetup)
{

   // take BasicClusters
  edm::Handle<reco::SuperClusterCollection> superClusters;
  iEvent.getByToken(superClustersToken_, superClusters);
  
  edm::Handle<reco::GsfElectronCollection> eles;
  iEvent.getByToken(elesToken_,eles);
  
  edm::Handle<reco::PhotonCollection> phos;
  iEvent.getByToken(phosToken_,phos);

  edm::Handle<reco::PFCluster::EEtoPSAssociation> eeClusToESMap;
  iEvent.getByToken(eeClusToESMapToken_,eeClusToESMap);

  edm::Handle<reco::PFClusterCollection> ecalPFClusters;
  iEvent.getByToken(ecalPFClustersToken_,ecalPFClusters);

  //Create empty output collections
  std::vector<DetId> indexToStore;
  indexToStore.reserve(100);

  if(eles.isValid() && eeClusToESMap.isValid() && ecalPFClusters.isValid()){
    for(auto& ele : *eles){
   
      float scEt = ele.superCluster()->energy()*std::sin(ele.superCluster()->position().theta());
      if(scEt > minEleEt_ || ele.et()> minEleEt_) addDetIds(*ele.superCluster(),*ecalPFClusters,*eeClusToESMap,indexToStore);
    }
  }
  if(phos.isValid() && eeClusToESMap.isValid() && ecalPFClusters.isValid()){
    for(auto& pho : *phos){
      float scEt = pho.superCluster()->energy()*std::sin(pho.superCluster()->position().theta());
      if(scEt > minPhoEt_ || pho.et()> minPhoEt_) addDetIds(*pho.superCluster(),*ecalPFClusters,*eeClusToESMap,indexToStore);
    }
  }
  if(superClusters.isValid() && eeClusToESMap.isValid() && ecalPFClusters.isValid()){
    for(auto& sc : *superClusters){
      float scEt = sc.energy()*std::sin(sc.position().theta());
      if(scEt > minSCEt_) addDetIds(sc,*ecalPFClusters,*eeClusToESMap,indexToStore);
    }
  }
  
  //unify the vector
  std::sort(indexToStore.begin(),indexToStore.end());
  std::unique(indexToStore.begin(),indexToStore.end());
  
  std::auto_ptr< DetIdCollection > detIdCollection (new DetIdCollection(indexToStore) ) ;
   
  iEvent.put( detIdCollection, interestingDetIdCollection_ );

}




//find all clusters within dR2<maxDR2_ of supercluster and then save det id's of hits of all es clusters matched to said ecal clusters
void
EgammaIsoESDetIdCollectionProducer::addDetIds(const reco::SuperCluster& superClus,reco::PFClusterCollection clusters,const reco::PFCluster::EEtoPSAssociation& eeClusToESMap,std::vector<DetId>& detIdsToStore)
{
  
  const float scEta = superClus.eta();
  //  if(std::abs(scEta)+maxDR_<1.5) return; //not possible to have a endcap cluster, let alone one with preshower (eta>1.65) so exit without checking further
  const float scPhi = superClus.phi();

  const float maxDR2=maxDR_*maxDR_;

  for (size_t clusNr=0;clusNr<clusters.size();clusNr++){
    const reco::PFCluster& clus = clusters[clusNr];
    if(clus.layer()==PFLayer::ECAL_ENDCAP &&
       reco::deltaR2(scEta,scPhi,clus.eta(),clus.phi())<maxDR2){
    
       auto keyVal = std::make_pair(clusNr,edm::Ptr<reco::PFCluster>());
       const auto esClusters = std::equal_range(eeClusToESMap.begin(),eeClusToESMap.end(),keyVal,
						[](const reco::PFCluster::EEtoPSAssociation::value_type& rhs, //roll on c++14, auto & lambda 4 evar!
						   const reco::PFCluster::EEtoPSAssociation::value_type& lhs)->
						bool{return rhs.first<lhs.first;}
						);
       //   std::cout <<"cluster "<<clus.eta()<<"  had "<<std::distance(esClusters.first,esClusters.second)<<" es clusters"<<std::endl;
       for(auto esIt = esClusters.first;esIt!=esClusters.second;++esIt){
	 //	 std::cout <<"es clus "<<esIt->second->hitsAndFractions().size()<<std::endl;
	 for(const auto& hitAndFrac : esIt->second->hitsAndFractions()){
	   detIdsToStore.push_back(hitAndFrac.first);
	 }
       }

    }//end of endcap & dR check
  }//end of cluster loop
 
  

}
