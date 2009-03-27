#include "RecoTauTag/RecoTau/interface/PFRecoTauDiscriminationByLeadingPionPtCut.h"
void PFRecoTauDiscriminationByLeadingPionPtCut::produce(Event& iEvent,const EventSetup& iEventSetup){
   Handle<PFTauCollection> thePFTauCollection;
   iEvent.getByLabel(PFTauProducer_,thePFTauCollection);


   auto_ptr<PFTauDiscriminator> thePFTauDiscriminatorByLeadingPionPtCut(new PFTauDiscriminator(PFTauRefProd(thePFTauCollection)));

   //loop over the PFTau candidates
   for(size_t iPFTau=0;iPFTau<thePFTauCollection->size();++iPFTau) {
      PFTauRef thePFTauRef(thePFTauCollection,iPFTau);
      PFTau thePFTau=*thePFTauRef;
      double theleadTrackPtCutDiscriminator = 0.;   
      // fill the AssociationVector object
      if (!thePFTau.leadPFCand()) 
      {
         theleadTrackPtCutDiscriminator=0.;
      }
      else if(thePFTau.leadPFCand()->pt() > minPtLeadTrack_) theleadTrackPtCutDiscriminator=1.;

      thePFTauDiscriminatorByLeadingPionPtCut->setValue(iPFTau,theleadTrackPtCutDiscriminator);
   }

   iEvent.put(thePFTauDiscriminatorByLeadingPionPtCut);

}
   
