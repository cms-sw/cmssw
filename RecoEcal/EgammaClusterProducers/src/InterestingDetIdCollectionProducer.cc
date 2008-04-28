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
  
}


InterestingDetIdCollectionProducer::~InterestingDetIdCollectionProducer()
{}

void InterestingDetIdCollectionProducer::beginJob (const edm::EventSetup& iSetup)  
{
  edm::ESHandle<CaloTopology> theCaloTopology;
  iSetup.get<CaloTopologyRecord>().get(theCaloTopology);
  caloTopology_ = &(*theCaloTopology); 
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
  std::auto_ptr< DetIdCollection > detIdCollection (new DetIdCollection() ) ;
  
  reco::BasicClusterCollection::const_iterator clusIt;
  
  for (clusIt=pClusters->begin(); clusIt!=pClusters->end(); clusIt++) {
    //PG barrel
    
    float eMax=0.;
    DetId eMaxId(0);

    std::vector<DetId> clusterDetIds = (*clusIt).getHitsByDetId();
    std::vector<DetId>::iterator posCurrent;

    EcalRecHit testEcalRecHit;
    
    for(posCurrent = clusterDetIds.begin(); posCurrent != clusterDetIds.end(); posCurrent++)
      {
	EcalRecHitCollection::const_iterator itt = recHitsHandle->find(*posCurrent);
	if ((!((*posCurrent).null())) && (itt != recHitsHandle->end()) && ((*itt).energy() > eMax) )
	  {
	    eMax = (*itt).energy();
	    eMaxId = (*itt).id();
	  }
      }
    
    if (eMaxId.null())
    continue;
    
    const CaloSubdetectorTopology* topology  = caloTopology_->getSubdetectorTopology(eMaxId.det(),eMaxId.subdetId());
    std::vector<DetId> xtalsToStore=topology->getWindow(eMaxId,minimalEtaSize_,minimalPhiSize_);
    std::vector<DetId> xtalsInClus=(*clusIt).getHitsByDetId();
    
    for (unsigned int ii=0;ii<xtalsInClus.size();ii++)
      {
	if (std::find(xtalsToStore.begin(),xtalsToStore.end(),xtalsInClus[ii]) == xtalsToStore.end())
	  xtalsToStore.push_back(xtalsInClus[ii]);
      }
    
    for (unsigned int iCry=0;iCry<xtalsToStore.size();iCry++)
      {
	if ( 
	    std::find(detIdCollection->begin(),detIdCollection->end(),xtalsToStore[iCry]) == detIdCollection->end()
	    )
	  detIdCollection->push_back(xtalsToStore[iCry]);
      }     
  }
  
  //  std::cout << "Interesting DetId Collection size is " << detIdCollection->size() <<  " BCs are " << pClusters->size() << std::endl;
  iEvent.put( detIdCollection, interestingDetIdCollection_ );

}
