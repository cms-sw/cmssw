#include "RecoTauTag/RecoTau/interface/CaloRecoTauDiscriminationByLeadingTrackPtCut.h"
void CaloRecoTauDiscriminationByLeadingTrackPtCut::produce(Event& iEvent,const EventSetup& iEventSetup){
   Handle<CaloTauCollection> theCaloTauCollection;
   iEvent.getByLabel(CaloTauProducer_,theCaloTauCollection);

   double theleadTrackPtCutDiscriminator = 0.;
   auto_ptr<CaloTauDiscriminator> theCaloTauDiscriminatorByLeadingTrackPtCut(new CaloTauDiscriminator(CaloTauRefProd(theCaloTauCollection)));

   //loop over the CaloTau candidates
   for(size_t iCaloTau=0;iCaloTau<theCaloTauCollection->size();++iCaloTau) {
      CaloTauRef theCaloTauRef(theCaloTauCollection,iCaloTau);
      CaloTau theCaloTau=*theCaloTauRef;

      // fill the AssociationVector object
      if (!theCaloTau.leadTrack()) 
         theleadTrackPtCutDiscriminator=0.;
      else if(theCaloTau.leadTrack()->pt() > minPtLeadTrack_) theleadTrackPtCutDiscriminator=1.;

      theCaloTauDiscriminatorByLeadingTrackPtCut->setValue(iCaloTau,theleadTrackPtCutDiscriminator);
   }

   iEvent.put(theCaloTauDiscriminatorByLeadingTrackPtCut);

}
   
