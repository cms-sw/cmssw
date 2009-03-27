#include "RecoTauTag/RecoTau/interface/PFRecoTauDiscriminationByLeadingTrackFinding.h"
void PFRecoTauDiscriminationByLeadingTrackFinding::produce(Event& iEvent,const EventSetup& iEventSetup){

   Handle<PFTauCollection> thePFTauCollection;
   iEvent.getByLabel(PFTauProducer_,thePFTauCollection);

   auto_ptr<PFTauDiscriminator> thePFTauDiscriminatorByLeadingTrackFinding(new PFTauDiscriminator(PFTauRefProd(thePFTauCollection)));

   //loop over the PFTau candidates
   for(size_t iPFTau=0;iPFTau<thePFTauCollection->size();++iPFTau) {
      PFTauRef thePFTauRef(thePFTauCollection,iPFTau);
      PFTau thePFTau=*thePFTauRef;
      double theleadTrackFindingDiscriminator = 0.;

      // fill the AssociationVector object
      if (thePFTau.leadPFChargedHadrCand().isNonnull()) 
         theleadTrackFindingDiscriminator = 1.;
      else
         theleadTrackFindingDiscriminator = 0.;


      thePFTauDiscriminatorByLeadingTrackFinding->setValue(iPFTau,theleadTrackFindingDiscriminator);
   }

   iEvent.put(thePFTauDiscriminatorByLeadingTrackFinding);

}
   
