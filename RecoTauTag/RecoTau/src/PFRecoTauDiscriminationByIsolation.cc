#include "RecoTauTag/RecoTau/interface/PFRecoTauDiscriminationByIsolation.h"

void PFRecoTauDiscriminationByIsolation::produce(Event& iEvent,const EventSetup& iEventSetup){
  Handle<PFTauCollection> thePFTauCollection;
  iEvent.getByLabel(PFTauProducer_,thePFTauCollection);
  
  // fill the AssociationVector object
  auto_ptr<PFTauDiscriminatorByIsolation> thePFTauDiscriminatorByIsolation(new PFTauDiscriminatorByIsolation(PFTauRefProd(thePFTauCollection)));
  
  for(size_t iPFTau=0;iPFTau<thePFTauCollection->size();++iPFTau) {
    PFTauRef thePFTauRef(thePFTauCollection,iPFTau);
    PFTau thePFTau=*thePFTauRef;
    math::XYZVector thePFTau_XYZVector=thePFTau.momentum();   
    PFTauElementsOperators thePFTauElementsOperators(thePFTau);
    
    if (ApplyDiscriminationByTrackerIsolation_){  
      // optional selection by a tracker isolation : ask for 0 charged hadron PFCand / reco::Track in an isolation annulus around a leading PFCand / reco::Track axis
      double theTrackerIsolationDiscriminator;
      if (ManipulateTracks_insteadofChargedHadrCands_){
	theTrackerIsolationDiscriminator=thePFTauElementsOperators.discriminatorByIsolTracksN(TrackerIsolAnnulus_Tracksmaxn_);
      } else theTrackerIsolationDiscriminator=thePFTauElementsOperators.discriminatorByIsolPFChargedHadrCandsN(TrackerIsolAnnulus_Candsmaxn_);      
      if (theTrackerIsolationDiscriminator==0){
	thePFTauDiscriminatorByIsolation->setValue(iPFTau,0);
	continue;
      }
    }    
    
    if (ApplyDiscriminationByECALIsolation_){
      // optional selection by an ECAL isolation : ask for 0 gamma PFCand in an isolation annulus around a leading PFCand
      double theECALIsolationDiscriminator;
      theECALIsolationDiscriminator=thePFTauElementsOperators.discriminatorByIsolPFGammaCandsN(ECALIsolAnnulus_Candsmaxn_);
      if (theECALIsolationDiscriminator==0){
	thePFTauDiscriminatorByIsolation->setValue(iPFTau,0);
	continue;
      }
    }
    
    // not optional selection : ask for a leading (Pt>minPt) PFCand / reco::Track in a matching cone around the PFJet axis
    double theleadElementDiscriminator=NAN;
    if (ManipulateTracks_insteadofChargedHadrCands_){
      if (!thePFTau.leadTrack()) theleadElementDiscriminator=0;
    }else{
      if (!thePFTau.leadPFChargedHadrCand()) theleadElementDiscriminator=0;
    }
    if (theleadElementDiscriminator==0) thePFTauDiscriminatorByIsolation->setValue(iPFTau,0);
    else thePFTauDiscriminatorByIsolation->setValue(iPFTau,1);
  }    
  
  iEvent.put(thePFTauDiscriminatorByIsolation);
  
}


