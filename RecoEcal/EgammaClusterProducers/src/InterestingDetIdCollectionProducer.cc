#include "RecoEcal/EgammaClusterProducers/interface/InterestingDetIdCollectionProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"

#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalTools.h"

// $Id$

InterestingDetIdCollectionProducer::InterestingDetIdCollectionProducer(const edm::ParameterSet& iConfig) 
{

  recHitsLabel_ = iConfig.getParameter< edm::InputTag > ("recHitsLabel");
  basicClusters_ = iConfig.getParameter< edm::InputTag > ("basicClustersLabel");

  interestingDetIdCollection_ = iConfig.getParameter<std::string>("interestingDetIdCollection");
  
  minimalEtaSize_ = iConfig.getParameter<int> ("etaSize");
  minimalPhiSize_ = iConfig.getParameter<int> ("phiSize");
  if ( minimalPhiSize_ % 2 == 0 ||  minimalEtaSize_ % 2 == 0)
    edm::LogError("InterestingDetIdCollectionProducerError") << "Size of eta/phi should be odd numbers";
 
   //register your products
  produces< DetIdCollection > (interestingDetIdCollection_) ;

  severityLevel_  = iConfig.getParameter<int>("severityLevel");
  keepNextToDead_ = iConfig.getParameter<bool>("keepNextToDead");
}


InterestingDetIdCollectionProducer::~InterestingDetIdCollectionProducer()
{}

void InterestingDetIdCollectionProducer::beginRun (edm::Run & run, const edm::EventSetup & iSetup)  
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
  iEvent.getByLabel(basicClusters_, pClusters);
  
  // take EcalRecHits
  Handle<EcalRecHitCollection> recHitsHandle;
  iEvent.getByLabel(recHitsLabel_,recHitsHandle);

  //Create empty output collections
  std::vector<DetId> indexToStore;
  indexToStore.reserve(1000);

  reco::BasicClusterCollection::const_iterator clusIt;

  std::vector<DetId> xtalsToStore;
  xtalsToStore.reserve(50);
  for (clusIt=pClusters->begin(); clusIt!=pClusters->end(); clusIt++) {
    //PG barrel
    
    float eMax=0.;
    DetId eMaxId(0);

    std::vector<std::pair<DetId,float> > clusterDetIds = (*clusIt).hitsAndFractions();
    std::vector<std::pair<DetId,float> >::iterator posCurrent;

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
    std::vector<std::pair<DetId,float > > xtalsInClus=(*clusIt).hitsAndFractions();
    
    for (unsigned int ii=0;ii<xtalsInClus.size();ii++)
      {
	  xtalsToStore.push_back(xtalsInClus[ii].first);
      }
    
    indexToStore.insert(indexToStore.end(),xtalsToStore.begin(),xtalsToStore.end());
  }


  // also add recHits of dead TT if the corresponding TP is saturated
  for (EcalRecHitCollection::const_iterator it = recHitsHandle->begin(); it != recHitsHandle->end(); ++it) {
          if ( it->checkFlag(EcalRecHit::kTPSaturated) ) {
	    indexToStore.push_back(it->id());
	  }
	  else if ( severityLevel_>=0 && severity_->severityLevel(*it) >=severityLevel_){
	    indexToStore.push_back(it->id());
	  } 
          else if (keepNextToDead_) {
	    // also keep channels next to dead ones
	    if (EcalTools::isNextToDead(it->id(), iSetup)) {
	      indexToStore.push_back(it->id());
	    }
	  }

	
  }

  //unify the vector
  std::sort(indexToStore.begin(),indexToStore.end());
  std::unique(indexToStore.begin(),indexToStore.end());
  
  std::auto_ptr< DetIdCollection > detIdCollection (new DetIdCollection(indexToStore) ) ;

 
  iEvent.put( detIdCollection, interestingDetIdCollection_ );

}
