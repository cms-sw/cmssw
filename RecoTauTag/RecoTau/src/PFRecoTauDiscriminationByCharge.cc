#include "RecoTauTag/RecoTau/interface/PFRecoTauDiscriminationByCharge.h"
#include "DataFormats/TrackReco/interface/Track.h"
void PFRecoTauDiscriminationByCharge::produce(Event& iEvent,const EventSetup& iEventSetup){
  Handle<PFTauCollection> thePFTauCollection;
  iEvent.getByLabel(PFTauProducer_,thePFTauCollection);
 

 auto_ptr<PFTauDiscriminator> thePFTauDiscriminatorByCharge(new PFTauDiscriminator(PFTauRefProd(thePFTauCollection)));

 //loop over the PFTau candidates
 for(size_t iPFTau=0;iPFTau<thePFTauCollection->size();++iPFTau) {
   PFTauRef thePFTauRef(thePFTauCollection,iPFTau);
   uint nSigTk=thePFTauRef->signalTracks().size();
   bool chargeok=fabs(thePFTauRef->charge())==1;
   bool oneOrThreeProng = (nSigTk==1)||(nSigTk==3);
   bool ptok = (thePFTauRef->pt()>ptcut_);
   bool  ptsum=true;
   float leadpt=0;
   float sigsum;
   uint nhits=0;
   for (TrackRefVector::iterator it =thePFTauRef->signalTracks().begin();it!=thePFTauRef->signalTracks().end(); ++it){
     sigsum+=(*it)->pt();
     if ((*it)->pt()>leadpt){
       leadpt=(*it)->pt();
       nhits=(*it)->found();
     }
   }
   bool NHits=nhits>minHitsLeadTk_;
   if (applySigTkSumPt_){
     ptsum=(sigsum/thePFTauRef->pt())>minSigPtTkRatio_;
   }
   if (chargeok && oneOrThreeProng && ptok &&ptsum && NHits)
     thePFTauDiscriminatorByCharge->setValue(iPFTau,1);
   else
     thePFTauDiscriminatorByCharge->setValue(iPFTau,0);   
 }
 
 iEvent.put(thePFTauDiscriminatorByCharge);

}
   
