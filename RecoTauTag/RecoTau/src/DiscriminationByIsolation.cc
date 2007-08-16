#include "RecoTauTag/RecoTau/interface/DiscriminationByIsolation.h"

void DiscriminationByIsolation::produce(Event& iEvent,const EventSetup& iEventSetup){
  Handle<TauCollection> theTauCollection;
  iEvent.getByLabel(TauProducer_,theTauCollection);
  // fill the AssociationVector object
  auto_ptr<TauDiscriminatorByIsolation> theTauDiscriminatorByIsolation(new TauDiscriminatorByIsolation(TauRefProd(theTauCollection)));
  //for (JetTagCollection::const_iterator iTau=theTauCollection.begin();iTau!=myJetTagCollection.end();++iTau) {
  for(size_t iTau=0;iTau<theTauCollection->size();++iTau) {
    TauRef theTauRef(theTauCollection,iTau);
    Tau theTau=*theTauRef;
    math::XYZVector theTau_XYZVector=theTau.momentum();   
    if (!theTau.getpfjetRef()){}else{
      PFTauElementsOperators thePFTauElementsOperators(theTau);
      if (ApplyDiscriminationByTrackerIsolation_){  
	// optional selection by a tracker isolation : ask for 0 charged hadron PFCand / reco::Track in an isolation annulus around a leading PFCand / reco::Track axis
	double theTrackerIsolationDiscriminator;
	if (ReCompute_leadElementSignalIsolationElements_){
	  TFormula theMatchingConeSizeTFormula=thePFTauElementsOperators.computeConeSizeTFormula(MatchingConeSizeFormula_,"Matching cone size");
	  double theMatchingConeSize=thePFTauElementsOperators.computeConeSize(theMatchingConeSizeTFormula,MatchingConeVariableSize_min_,MatchingConeVariableSize_max_);
	  TFormula theTrackerSignalConeSizeTFormula=thePFTauElementsOperators.computeConeSizeTFormula(TrackerSignalConeSizeFormula_,"Tracker signal cone size");
	  double theTrackerSignalConeSize=thePFTauElementsOperators.computeConeSize(theTrackerSignalConeSizeTFormula,TrackerSignalConeVariableSize_min_,TrackerSignalConeVariableSize_max_);
	  TFormula theTrackerIsolConeSizeTFormula=thePFTauElementsOperators.computeConeSizeTFormula(TrackerIsolConeSizeFormula_,"Tracker isolation cone size");
	  double theTrackerIsolConeSize=thePFTauElementsOperators.computeConeSize(theTrackerIsolConeSizeTFormula,TrackerIsolConeVariableSize_min_,TrackerIsolConeVariableSize_max_);     		  	    
	  if (ManipulateTracks_insteadofChargedHadrCands_){
	    CaloTauElementsOperators theCaloTauElementsOperators(theTau); 
	    theTrackerIsolationDiscriminator=theCaloTauElementsOperators.discriminator(theTau_XYZVector,MatchingConeMetric_,theMatchingConeSize,LeadTrack_minPt_,Track_minPt_,TrackerSignalConeMetric_,theTrackerSignalConeSize,TrackerIsolConeMetric_,theTrackerIsolConeSize,TrackerIsolAnnulus_Tracksmaxn_);
	  }else
	    theTrackerIsolationDiscriminator=thePFTauElementsOperators.discriminatorByIsolPFChargedHadrCandsN(theTau_XYZVector,MatchingConeMetric_,theMatchingConeSize,TrackerSignalConeMetric_,theTrackerSignalConeSize,TrackerIsolConeMetric_,theTrackerIsolConeSize,LeadCand_minPt_,ChargedHadrCand_minPt_,TrackerIsolAnnulus_Candsmaxn_);
	}else{
	  if (ManipulateTracks_insteadofChargedHadrCands_){
	    CaloTauElementsOperators theCaloTauElementsOperators(theTau); 
	    theTrackerIsolationDiscriminator=theCaloTauElementsOperators.discriminator(TrackerIsolAnnulus_Tracksmaxn_);
	  } else theTrackerIsolationDiscriminator=thePFTauElementsOperators.discriminatorByIsolPFChargedHadrCandsN(TrackerIsolAnnulus_Candsmaxn_);
	}
	if (theTrackerIsolationDiscriminator==0){
	  theTauDiscriminatorByIsolation->setValue(iTau,0);
	  continue;
	}
      }
      
      if (ApplyDiscriminationByECALIsolation_){
	// optional selection by an ECAL isolation : ask for 0 gamma PFCand in an isolation annulus around a leading PFCand
	double theECALIsolationDiscriminator;
	if (ReCompute_leadElementSignalIsolationElements_){
	  TFormula theMatchingConeSizeTFormula=thePFTauElementsOperators.computeConeSizeTFormula(MatchingConeSizeFormula_,"Matching cone size");
	  double theMatchingConeSize=thePFTauElementsOperators.computeConeSize(theMatchingConeSizeTFormula,MatchingConeVariableSize_min_,MatchingConeVariableSize_max_);
	  TFormula theECALSignalConeSizeTFormula=thePFTauElementsOperators.computeConeSizeTFormula(ECALSignalConeSizeFormula_,"ECAL signal cone size");
	  double theECALSignalConeSize=thePFTauElementsOperators.computeConeSize(theECALSignalConeSizeTFormula,ECALSignalConeVariableSize_min_,ECALSignalConeVariableSize_max_);
	  TFormula theECALIsolConeSizeTFormula=thePFTauElementsOperators.computeConeSizeTFormula(ECALIsolConeSizeFormula_,"ECAL isolation cone size");
	  double theECALIsolConeSize=thePFTauElementsOperators.computeConeSize(theECALIsolConeSizeTFormula,ECALIsolConeVariableSize_min_,ECALIsolConeVariableSize_max_);     	
	  theECALIsolationDiscriminator=thePFTauElementsOperators.discriminatorByIsolPFGammaCandsN(theTau_XYZVector,MatchingConeMetric_,theMatchingConeSize,ECALSignalConeMetric_,theECALSignalConeSize,ECALIsolConeMetric_,theECALIsolConeSize,UseOnlyChargedHadr_for_LeadCand_,LeadCand_minPt_,GammaCand_minPt_,ECALIsolAnnulus_Candsmaxn_);
	}else theECALIsolationDiscriminator=thePFTauElementsOperators.discriminatorByIsolPFGammaCandsN(ECALIsolAnnulus_Candsmaxn_);
	if (theECALIsolationDiscriminator==0){
	  theTauDiscriminatorByIsolation->setValue(iTau,0);
	  continue;
	}
      }
      
      // not optional selection : ask for a leading (Pt>minPt) PFCand / reco::Track in a matching cone around the PFJet axis
      double theleadElementDiscriminator=NAN;
      if (ReCompute_leadElementSignalIsolationElements_){
	TFormula theMatchingConeSizeTFormula=thePFTauElementsOperators.computeConeSizeTFormula(MatchingConeSizeFormula_,"Matching cone size");
	double theMatchingConeSize=thePFTauElementsOperators.computeConeSize(theMatchingConeSizeTFormula,MatchingConeVariableSize_min_,MatchingConeVariableSize_max_);
	if (ManipulateTracks_insteadofChargedHadrCands_){
	  CaloTauElementsOperators theCaloTauElementsOperators(theTau); 
	  if (!theCaloTauElementsOperators.leadTk(MatchingConeMetric_,theMatchingConeSize,LeadTrack_minPt_)) theleadElementDiscriminator=0;
	}else{
	  if (UseOnlyChargedHadr_for_LeadCand_){
	    if (!thePFTauElementsOperators.leadPFChargedHadrCand(MatchingConeMetric_,theMatchingConeSize,LeadCand_minPt_)) theleadElementDiscriminator=0;
	  }else{
	    if (!thePFTauElementsOperators.leadPFCand(MatchingConeMetric_,theMatchingConeSize,LeadCand_minPt_)) theleadElementDiscriminator=0;
	  }
	}
      }else{
	if (ManipulateTracks_insteadofChargedHadrCands_){
	  CaloTauElementsOperators theCaloTauElementsOperators(theTau); 
	  if (!theTau.getleadTrack()) theleadElementDiscriminator=0;
	}else{
	  if (!theTau.getleadPFChargedHadrCand()) theleadElementDiscriminator=0;
	}
      }
      if (theleadElementDiscriminator==0) theTauDiscriminatorByIsolation->setValue(iTau,0);
      else theTauDiscriminatorByIsolation->setValue(iTau,1);
      continue;
    }
    
    if (!theTau.getcalojetRef()){}else{
      CaloTauElementsOperators theCaloTauElementsOperators(theTau); 	
      if (ApplyDiscriminationByTrackerIsolation_){  
	// optional selection by a tracker isolation : ask for 0 reco::Track in an isolation annulus around a leading reco::Track axis
	double theTrackerIsolationDiscriminator;
	if (ReCompute_leadElementSignalIsolationElements_){
	  TFormula theMatchingConeSizeTFormula=theCaloTauElementsOperators.computeConeSizeTFormula(MatchingConeSizeFormula_,"Matching cone size");
	  double theMatchingConeSize=theCaloTauElementsOperators.computeConeSize(theMatchingConeSizeTFormula,MatchingConeVariableSize_min_,MatchingConeVariableSize_max_);
	  TFormula theTrackerSignalConeSizeTFormula=theCaloTauElementsOperators.computeConeSizeTFormula(TrackerSignalConeSizeFormula_,"Tracker signal cone size");
	  double theTrackerSignalConeSize=theCaloTauElementsOperators.computeConeSize(theTrackerSignalConeSizeTFormula,TrackerSignalConeVariableSize_min_,TrackerSignalConeVariableSize_max_);
	  TFormula theTrackerIsolConeSizeTFormula=theCaloTauElementsOperators.computeConeSizeTFormula(TrackerIsolConeSizeFormula_,"Tracker isolation cone size");
	  double theTrackerIsolConeSize=theCaloTauElementsOperators.computeConeSize(theTrackerIsolConeSizeTFormula,TrackerIsolConeVariableSize_min_,TrackerIsolConeVariableSize_max_);     	
	  theTrackerIsolationDiscriminator=theCaloTauElementsOperators.discriminator(theTau_XYZVector,MatchingConeMetric_,theMatchingConeSize,LeadTrack_minPt_,Track_minPt_,TrackerSignalConeMetric_,theTrackerSignalConeSize,TrackerIsolConeMetric_,theTrackerIsolConeSize,TrackerIsolAnnulus_Tracksmaxn_);
	}else{
	  theTrackerIsolationDiscriminator=theCaloTauElementsOperators.discriminator(TrackerIsolAnnulus_Tracksmaxn_);
	}
	if (theTrackerIsolationDiscriminator==0){
	  theTauDiscriminatorByIsolation->setValue(iTau,0);
	  continue;
	}
      }
      
      // not optional selection : ask for a leading (Pt>minPt) reco::Track in a matching cone around the CaloJet axis
      double theleadTkDiscriminator=NAN;
      if (ReCompute_leadElementSignalIsolationElements_){
	TFormula theMatchingConeSizeTFormula=theCaloTauElementsOperators.computeConeSizeTFormula(MatchingConeSizeFormula_,"Matching cone size");
	double theMatchingConeSize=theCaloTauElementsOperators.computeConeSize(theMatchingConeSizeTFormula,MatchingConeVariableSize_min_,MatchingConeVariableSize_max_);
	if (!theCaloTauElementsOperators.leadTk(MatchingConeMetric_,theMatchingConeSize,LeadTrack_minPt_)) theleadTkDiscriminator=0;
      }else{
	if (!theTau.getleadTrack()) theleadTkDiscriminator=0;
      }
      if (theleadTkDiscriminator==0) theTauDiscriminatorByIsolation->setValue(iTau,0);
      else theTauDiscriminatorByIsolation->setValue(iTau,1);
      continue;
    }
  }
  iEvent.put(theTauDiscriminatorByIsolation);
}
