#include "RecoTauTag/RecoTau/interface/CaloRecoTauDiscriminationByLeadingTrackFinding.h"
void CaloRecoTauDiscriminationByLeadingTrackFinding::produce(Event& iEvent,const EventSetup& iEventSetup){

   Handle<CaloTauCollection> theCaloTauCollection;
   iEvent.getByLabel(CaloTauProducer_,theCaloTauCollection);

   auto_ptr<CaloTauDiscriminator> theCaloTauDiscriminatorByLeadingTrackFinding(new CaloTauDiscriminator(CaloTauRefProd(theCaloTauCollection)));
   double theleadTrackFindingDiscriminator = 0.;

   //loop over the CaloTau candidates
   for(size_t iCaloTau=0;iCaloTau<theCaloTauCollection->size();++iCaloTau) {
      CaloTauRef theCaloTauRef(theCaloTauCollection,iCaloTau);
      CaloTau theCaloTau=*theCaloTauRef;

      // fill the AssociationVector object
      if (theCaloTau.leadTrack().isNonnull()) 
         theleadTrackFindingDiscriminator = 1.;
      else 
         theleadTrackFindingDiscriminator = 0.;


      theCaloTauDiscriminatorByLeadingTrackFinding->setValue(iCaloTau,theleadTrackFindingDiscriminator);
   }

   iEvent.put(theCaloTauDiscriminatorByLeadingTrackFinding);

}
   
