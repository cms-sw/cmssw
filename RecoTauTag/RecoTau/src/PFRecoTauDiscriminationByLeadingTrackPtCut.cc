#include "DataFormats/TauReco/interface/PFRecoTauDiscriminationByLeadingTrackPtCut.h"
void PFRecoTauDiscriminationByIsolation::produce(Event& iEvent,const EventSetup& iEventSetup){
  Handle<PFTauCollection> thePFTauCollection;
 iEvent.getByLabel(PFTauProducer_,thePFTauCollection);
 
 double theleadTrackPtCutDiscriminator = 0.;
 auto_ptr<PFTauDiscriminator> thePFTauDiscriminatorByLeadingTrackPtCut(new PFTauDiscriminator);

 //loop over the PFTau candidates
 for(size_t iPFTau=0;iPFTau<thePFTauCollection->size();++iPFTau) {
   PFTauRef thePFTauRef(thePFTauCollection,iPFTau);
   PFTau thePFTau=*thePFTauRef;
   
   // fill the AssociationVector object
   if (!thePFTau.leadTrack()) 
     {
       theleadTrackPtCutDiscriminator=0.;
     }else if(thePFTau.leadTrack().pt() > minPtLeadTrack_) theleadTrackPtCutDiscriminator=1.;

   thePFTauDiscriminatorByLeadingTrackPtCut->setValue(iPFTau,theleadTrackPtCutDiscriminator);
 }
 
 iEvent.put(thePFTauDiscriminatorByLeadingTrackPtCut);

}
   
