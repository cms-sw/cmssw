#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTRemoveDuplicatedSC.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/Handle.h"

#include <string>

EgammaHLTRemoveDuplicatedSC::EgammaHLTRemoveDuplicatedSC(const edm::ParameterSet& ps)
{
 
 
  // the input producers

  sCInputProducer_ = ps.getParameter<edm::InputTag>("L1NonIsoUskimmedSC");
  alreadyExistingSC_= ps.getParameter<edm::InputTag>("L1IsoSC");
  // set the producer parameters
  outputCollection_ = ps.getParameter<std::string>("L1NonIsoSkimmedCollection");
  produces<reco::SuperClusterCollection>(outputCollection_);

  
}

EgammaHLTRemoveDuplicatedSC::~EgammaHLTRemoveDuplicatedSC()
{
  ;
}

void
EgammaHLTRemoveDuplicatedSC::produce(edm::Event& evt, const edm::EventSetup& es)
{
  using namespace edm;

 
  // Get raw SuperClusters from the event    
  Handle<reco::SuperClusterCollection> UnskimmedSuperCluster;
  evt.getByLabel(sCInputProducer_, UnskimmedSuperCluster);
  Handle<reco::SuperClusterCollection> L1IsoSuperCluster;
  evt.getByLabel(alreadyExistingSC_, L1IsoSuperCluster);
  /*
    edm::LogError("EgammaHLTRemoveDuplicatedSCError") 
      << "Error! can't get the rawSuperClusters " 
      << sCInputProducer_.label() ;
  */
  

  // Create a pointer to the existing SuperClusters
    const reco::SuperClusterCollection *UnskimmedL1NonIsoSC = UnskimmedSuperCluster.product();
    const reco::SuperClusterCollection *L1IsoSC = L1IsoSuperCluster.product();

    /*
    for(reco::SuperClusterCollection::const_iterator it = L1IsoSC->begin(); it != L1IsoSC->end(); it++){
      std::cout<<"L1 iso  E, eta, phi: "<<it->energy()<<" "<<it->eta()<<" "<<it->phi()<<std::endl;
    }
    std::cout<<std::endl;
    for(reco::SuperClusterCollection::const_iterator it = UnskimmedL1NonIsoSC->begin(); it != UnskimmedL1NonIsoSC->end(); it++){
      std::cout<<"L1 Non iso (not skimmed) E, eta, phi: "<<it->energy()<<" "<<it->eta()<<" "<<it->phi()<<std::endl;
    }
    std::cout<<std::endl;
    */

  // Define a collection of corrected SuperClusters to put back into the event
  std::auto_ptr<reco::SuperClusterCollection> corrClusters(new reco::SuperClusterCollection);
  
  //  Loop over raw clusters and make corrected ones
  reco::SuperClusterCollection::const_iterator aClus;
  reco::SuperClusterCollection::const_iterator cit;
  for(aClus = UnskimmedL1NonIsoSC->begin(); aClus != UnskimmedL1NonIsoSC->end(); aClus++)
    {
      bool AlreadyThere = false;
      //reco::SuperCluster newClus;
      for(cit = L1IsoSC->begin(); cit != L1IsoSC->end(); cit++){
	if( fabs(aClus->energy()- cit->energy()) < 0.5 && fabs(aClus->eta()- cit->eta()) < 0.0175 ){
	  float deltaphi=fabs( aClus->phi() - cit->phi() );
	  if(deltaphi>6.283185308) deltaphi -= 6.283185308;
	  if(deltaphi>3.141592654) deltaphi = 6.283185308-deltaphi;

	  if( deltaphi < 0.035 ){AlreadyThere = true; break;}
	}
      }
      // if(AlreadyThere){std::cout<<"AAAA: SC removed: "<<aClus->energy()<<" "<<aClus->eta()<<" "<<aClus->phi()<<std::endl;}
      if(!AlreadyThere){corrClusters->push_back(*aClus);}
    }

  // Put collection of corrected SuperClusters into the event
  evt.put(corrClusters, outputCollection_);   
  
}



