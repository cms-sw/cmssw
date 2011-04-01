#include "RecoEcal/EgammaClusterProducers/interface/ReducedRecHitCollectionProducer.h"

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

#include <iostream>

ReducedRecHitCollectionProducer::ReducedRecHitCollectionProducer(const edm::ParameterSet& iConfig) 
{

  recHitsLabel_ = iConfig.getParameter< edm::InputTag > ("recHitsLabel");
  interestingDetIdCollections_ = iConfig.getParameter< std::vector<edm::InputTag> > ("interestingDetIdCollections");

  reducedHitsCollection_ = iConfig.getParameter<std::string>("reducedHitsCollection");
  
   //register your products
  produces< EcalRecHitCollection > (reducedHitsCollection_) ;
  
}


ReducedRecHitCollectionProducer::~ReducedRecHitCollectionProducer()
{}


// ------------ method called to produce the data  ------------
void
ReducedRecHitCollectionProducer::produce (edm::Event& iEvent, 
                                const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;

   if (interestingDetIdCollections_.size() < 1)
     {
       edm::LogError("ReducedRecHitCollectionProducer") << "VInputTag collections empty" ;
       return;
     }
   

   Handle< DetIdCollection > detIds;
   iEvent.getByLabel(interestingDetIdCollections_[0],detIds);
   std::vector<DetId> xtalsToStore((*detIds).size());
   std::copy( (*detIds).begin() , (*detIds).end() , xtalsToStore.begin() );   
   
   //Merging DetIds from different collections 
   for( unsigned int t = 1; t < interestingDetIdCollections_.size(); ++t )
     {
       Handle< DetIdCollection > detId;
       iEvent.getByLabel(interestingDetIdCollections_[t],detId);
       if( !detId.isValid() ){
	 edm::LogError("MissingInput")<<"no reason to skip detid from :"<<interestingDetIdCollections_[t];
	 continue;
       }
       
       for (unsigned int ii=0;ii<(*detId).size();ii++)
	 {
	   if (std::find(xtalsToStore.begin(),xtalsToStore.end(),(*detId)[ii]) == xtalsToStore.end())
	     xtalsToStore.push_back((*detId)[ii]);
	 }
     }
   
   Handle<EcalRecHitCollection> recHitsHandle;
   iEvent.getByLabel(recHitsLabel_,recHitsHandle);
   if( !recHitsHandle.isValid() ) 
     {
       edm::LogError("ReducedRecHitCollectionProducer") << "RecHit collection not found";
       return;
     }
   
   //Create empty output collections
   std::auto_ptr< EcalRecHitCollection > miniRecHitCollection (new EcalRecHitCollection) ;
   
   for (unsigned int iCry=0;iCry<xtalsToStore.size();iCry++)
     {
       EcalRecHitCollection::const_iterator iRecHit = recHitsHandle->find(xtalsToStore[iCry]);
       if ( (iRecHit != recHitsHandle->end()) && (miniRecHitCollection->find(xtalsToStore[iCry]) == miniRecHitCollection->end()) )
	 miniRecHitCollection->push_back(*iRecHit);
     }     
   
   //   std::cout << "New Collection " << reducedHitsCollection_ << " size is " << miniRecHitCollection->size() << " original is " << recHitsHandle->size() << std::endl;
   iEvent.put( miniRecHitCollection,reducedHitsCollection_ );
}
