#include "RecoTauTag/RecoTau/interface/CaloRecoTauDiscriminationByIsolation.h"

void CaloRecoTauDiscriminationByIsolation::produce(Event& iEvent,const EventSetup& iEventSetup){
  Handle<CaloTauCollection> theCaloTauCollection;
  iEvent.getByLabel(CaloTauProducer_,theCaloTauCollection);

  // fill the AssociationVector object
  auto_ptr<CaloTauDiscriminatorByIsolation> theCaloTauDiscriminatorByIsolation(new CaloTauDiscriminatorByIsolation(CaloTauRefProd(theCaloTauCollection)));

  for(size_t iCaloTau=0;iCaloTau<theCaloTauCollection->size();++iCaloTau) {
    CaloTauRef theCaloTauRef(theCaloTauCollection,iCaloTau);
    CaloTau theCaloTau=*theCaloTauRef;
    math::XYZVector theCaloTau_XYZVector=theCaloTau.momentum();   
    CaloTauElementsOperators theCaloTauElementsOperators(theCaloTau); 	
    
    if (ApplyDiscriminationByTrackerIsolation_){  
      // optional selection by a tracker isolation : ask for 0 reco::Track in an isolation annulus around a leading reco::Track axis
      double theTrackerIsolationDiscriminator;
      theTrackerIsolationDiscriminator=theCaloTauElementsOperators.discriminatorByIsolTracksN(TrackerIsolAnnulus_Tracksmaxn_);
      if (theTrackerIsolationDiscriminator==0){
	theCaloTauDiscriminatorByIsolation->setValue(iCaloTau,0);
	continue;
      }
    }
    
    // not optional selection : ask for a leading (Pt>minPt) reco::Track in a matching cone around the CaloJet axis
    double theleadTkDiscriminator=NAN;
    if (!theCaloTau.leadTrack()) theleadTkDiscriminator=0;
    if (theleadTkDiscriminator==0) theCaloTauDiscriminatorByIsolation->setValue(iCaloTau,0);
    else theCaloTauDiscriminatorByIsolation->setValue(iCaloTau,1);
  }
  
  iEvent.put(theCaloTauDiscriminatorByIsolation);
}
