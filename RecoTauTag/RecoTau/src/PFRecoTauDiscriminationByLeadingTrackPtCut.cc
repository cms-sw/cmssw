#include "RecoTauTag/RecoTau/interface/PFRecoTauDiscriminationByLeadingTrackPtCut.h"
void PFRecoTauDiscriminationByLeadingTrackPtCut::produce(Event& iEvent,const EventSetup& iEventSetup){
   Handle<PFTauCollection> thePFTauCollection;
   iEvent.getByLabel(PFTauProducer_,thePFTauCollection);


   auto_ptr<PFTauDiscriminator> thePFTauDiscriminatorByLeadingTrackPtCut(new PFTauDiscriminator(PFTauRefProd(thePFTauCollection)));

   //loop over the PFTau candidates
   for(size_t iPFTau=0;iPFTau<thePFTauCollection->size();++iPFTau) {
      PFTauRef thePFTauRef(thePFTauCollection,iPFTau);
      PFTau thePFTau=*thePFTauRef;
      double theleadTrackPtCutDiscriminator = 0.;   
      // fill the AssociationVector object
      if (!thePFTau.leadPFChargedHadrCand()) 
      {
         theleadTrackPtCutDiscriminator=0.;
      }else if(thePFTau.leadPFChargedHadrCand()->pt() > minPtLeadTrack_) theleadTrackPtCutDiscriminator=1.;

      thePFTauDiscriminatorByLeadingTrackPtCut->setValue(iPFTau,theleadTrackPtCutDiscriminator);
   }

   iEvent.put(thePFTauDiscriminatorByLeadingTrackPtCut);

}

