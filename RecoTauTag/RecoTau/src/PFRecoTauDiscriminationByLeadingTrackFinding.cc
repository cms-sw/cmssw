#include "RecoTauTag/RecoTau/interface/PFRecoTauDiscriminationByLeadingTrackFinding.h"
void PFRecoTauDiscriminationByLeadingTrackPtCut::produce(Event& iEvent,const EventSetup& iEventSetup){
  Handle<PFTauCollection> thePFTauCollection;
  iEvent.getByLabel(PFTauProducer_,thePFTauCollection);
  
 auto_ptr<PFTauDiscriminator> thePFTauDiscriminatorByLeadingTrackFinding(new PFTauDiscriminator);
 double theleadTrackFindingDiscriminator = 0.;


 //loop over the PFTau candidates
 for(size_t iPFTau=0;iPFTau<thePFTauCollection->size();++iPFTau) {
   PFTauRef thePFTauRef(thePFTauCollection,iPFTau);
   PFTau thePFTau=*thePFTauRef;
   
   // fill the AssociationVector object
   if (!thePFTau.leadPFChargedHadrCand()) 
     {
       theleadTrackFindingDiscriminator=0.;
     }else{  
       theleadTrackPtCutDiscriminator=1.;
     }

   thePFTauDiscriminatorByLeadingTrackFinding->setValue(iPFTau,theleadTrackFindingDiscriminator);
 }
 
 iEvent.put(thePFTauDiscriminatorByLeadingTrackFinding);

}
   
