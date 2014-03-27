#include "RecoEcal/EgammaClusterProducers/interface/InterestingDetIdCollectionProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"


#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"

#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalTools.h"


InterestingDetIdCollectionProducer::InterestingDetIdCollectionProducer(const edm::ParameterSet& iConfig) 
{

  recHitsToken_ = 
	  consumes<EcalRecHitCollection>(iConfig.getParameter< edm::InputTag > ("recHitsLabel"));
  basicClustersToken_ = 
	  consumes<reco::BasicClusterCollection>(iConfig.getParameter< edm::InputTag >("basicClustersLabel"));

  interestingDetIdCollection_ = iConfig.getParameter<std::string>("interestingDetIdCollection");
  
  minimalEtaSize_ = iConfig.getParameter<int> ("etaSize");
  minimalPhiSize_ = iConfig.getParameter<int> ("phiSize");
  if ( minimalPhiSize_ % 2 == 0 ||  minimalEtaSize_ % 2 == 0)
    edm::LogError("InterestingDetIdCollectionProducerError") << "Size of eta/phi should be odd numbers";
 
   //register your products
  produces< DetIdCollection > (interestingDetIdCollection_) ;

  severityLevel_  = iConfig.getParameter<int>("severityLevel");
  keepNextToDead_ = iConfig.getParameter<bool>("keepNextToDead");
  keepNextToBoundary_ = iConfig.getParameter<bool>("keepNextToBoundary");
}


void InterestingDetIdCollectionProducer::beginRun (edm::Run const& run, const edm::EventSetup & iSetup)  
{
  edm::ESHandle<CaloTopology> theCaloTopology;
  iSetup.get<CaloTopologyRecord>().get(theCaloTopology);
  caloTopology_ = &(*theCaloTopology); 

  edm::ESHandle<EcalSeverityLevelAlgo> sevLv;
  iSetup.get<EcalSeverityLevelAlgoRcd>().get(sevLv);
  severity_ = sevLv.product();
}

// ------------ method called to produce the data  ------------
void
InterestingDetIdCollectionProducer::produce (edm::Event& iEvent, 
                                const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;

   // take BasicClusters
  Handle<reco::BasicClusterCollection> pClusters;
  iEvent.getByToken(basicClustersToken_, pClusters);
  
  // take EcalRecHits
  Handle<EcalRecHitCollection> recHitsHandle;
  iEvent.getByToken(recHitsToken_,recHitsHandle);

  //Create empty output collections
  std::vector<DetId> indexToStore;
  indexToStore.reserve(1000);

  reco::BasicClusterCollection::const_iterator clusIt;

  std::vector<DetId> xtalsToStore;
  xtalsToStore.reserve(50);
  for (clusIt=pClusters->begin(); clusIt!=pClusters->end(); clusIt++) {
    const std::vector<std::pair<DetId,float> > &clusterDetIds = clusIt->hitsAndFractions();
    for (const auto &detidpair : clusterDetIds) {
      indexToStore.push_back(detidpair.first);
    }
    
    //below checks and additional code are only relevant for EB/EE, not for ES
    if (clusterDetIds.front().first.subdetId()==EcalBarrel || clusterDetIds.front().first.subdetId()==EcalEndcap) {
      std::vector<std::pair<DetId,float> >::const_iterator posCurrent;
      
      float eMax=0.;
      DetId eMaxId(0);        

      EcalRecHit testEcalRecHit;
      
      for(posCurrent = clusterDetIds.begin(); posCurrent != clusterDetIds.end(); posCurrent++)
        {
          EcalRecHitCollection::const_iterator itt = recHitsHandle->find((*posCurrent).first);
          if ((!((*posCurrent).first.null())) && (itt != recHitsHandle->end()) && ((*itt).energy() > eMax) )
            {
              eMax = (*itt).energy();
              eMaxId = (*itt).id();
            }
        }
      
      if (eMaxId.null())
      continue;
      
      const CaloSubdetectorTopology* topology  = caloTopology_->getSubdetectorTopology(eMaxId.det(),eMaxId.subdetId());

      xtalsToStore=topology->getWindow(eMaxId,minimalEtaSize_,minimalPhiSize_);
      
      for (const auto &detid : xtalsToStore) {
        indexToStore.push_back(detid);
      }
      
    }
  }


  if (severityLevel_>=0 || keepNextToDead_ || keepNextToBoundary_) {
    for (EcalRecHitCollection::const_iterator it = recHitsHandle->begin(); it != recHitsHandle->end(); ++it) {
      // also add recHits of dead TT if the corresponding TP is saturated
      if ( it->checkFlag(EcalRecHit::kTPSaturated) ) {
        indexToStore.push_back(it->id());
      }
      // add hits for severities above a threshold
      if ( severityLevel_>=0 && 
          severity_->severityLevel(*it) >=severityLevel_){
        
        indexToStore.push_back(it->id());
      } 
      if (keepNextToDead_) {
        // also keep channels next to dead ones
        if (EcalTools::isNextToDead(it->id(), iSetup)) {
          indexToStore.push_back(it->id());
        }
      } 

      if (keepNextToBoundary_){
        // keep channels around EB/EE boundary
        if (it->id().subdetId() == EcalBarrel){
          EBDetId ebid(it->id());
          if (abs(ebid.ieta())== 85)
            indexToStore.push_back(it->id());
        } else {
      
          if (EEDetId::isNextToRingBoundary(it->id()))
            indexToStore.push_back(it->id());
        }

      }
      
    }
  }

  //unify the vector
  std::sort(indexToStore.begin(),indexToStore.end());
  std::unique(indexToStore.begin(),indexToStore.end());
  
  std::auto_ptr< DetIdCollection > detIdCollection (new DetIdCollection(indexToStore) ) ;

 
  iEvent.put( detIdCollection, interestingDetIdCollection_ );

}
