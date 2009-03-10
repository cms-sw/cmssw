#include "RecoTauTag/RecoTau/interface/PFRecoTauDiscriminationByNeutralHadrons.h"

void PFRecoTauDiscriminationByNeutralHadrons::produce(Event& iEvent,const EventSetup& iEventSetup){
  Handle<PFTauCollection> thePFTauCollection;
  iEvent.getByLabel(PFTauProducer_,thePFTauCollection);
  
  // fill the AssociationVector object
  auto_ptr<PFTauDiscriminator> thePFTauDiscriminatorByNeutralHadrons(new PFTauDiscriminator(PFTauRefProd(thePFTauCollection)));
  
  for(size_t iPFTau=0;iPFTau<thePFTauCollection->size();++iPFTau) {
    PFTauRef thePFTauRef(thePFTauCollection,iPFTau);
    PFTau thePFTau=*thePFTauRef;
    if(thePFTau.signalPFNeutrHadrCands().size() > neutralHadrons_){
      thePFTauDiscriminatorByNeutralHadrons->setValue(iPFTau,0);
    }else thePFTauDiscriminatorByNeutralHadrons->setValue(iPFTau,1);
  }    
  
  iEvent.put(thePFTauDiscriminatorByNeutralHadrons);
  
}


