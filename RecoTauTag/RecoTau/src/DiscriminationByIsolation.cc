#include "RecoTauTag/RecoTau/interface/DiscriminationByIsolation.h"

void DiscriminationByIsolation::produce(Event& iEvent,const EventSetup& iEventSetup){
  Handle<TauCollection> theTauCollection;
  iEvent.getByLabel(TauProducer_,theTauCollection);
  // fill the AssociationVector object
  auto_ptr<TauDiscriminatorByIsolation> theTauDiscriminatorByIsolation(new TauDiscriminatorByIsolation(TauRefProd(theTauCollection)));
  for(size_t iTau=0;iTau<theTauCollection->size();++iTau) {
    TauRef theTauRef(theTauCollection,iTau);
    math::XYZVector theTauRef_XYZVector=(*theTauRef).momentum();   
    double theMatchingConeSize=computeConeSize(theTauRef,MatchingConeSizeTFormula_,MatchingConeVariableSize_min_,MatchingConeVariableSize_max_);
    if (!(*theTauRef).getpfjetRef()){}else{
      PFTauElementsOperators thePFTauElementsOperators(theTauRef); 
      if (ApplyDiscriminationByTrackerIsolation_){  
	double trackerdiscriminator;
	double theTrackerSignalConeSize=computeConeSize(theTauRef,TrackerSignalConeSizeTFormula_,TrackerSignalConeVariableSize_min_,TrackerSignalConeVariableSize_max_);
	double theTrackerIsolConeSize=computeConeSize(theTauRef,TrackerIsolConeSizeTFormula_,TrackerIsolConeVariableSize_min_,TrackerIsolConeVariableSize_max_);     	
	if (ManipulateTracks_insteadofChargedHadrCands_){
	  CaloTauElementsOperators theCaloTauElementsOperators(theTauRef); 
	  trackerdiscriminator=theCaloTauElementsOperators.discriminator(theTauRef_XYZVector,MatchingConeMetric_,theMatchingConeSize,LeadTrack_minPt_,Track_minPt_,TrackerSignalConeMetric_,theTrackerSignalConeSize,TrackerIsolConeMetric_,theTrackerIsolConeSize,TrackerIsolAnnulus_Tracksmaxn_);
	  if (trackerdiscriminator==0){
	    theTauDiscriminatorByIsolation->setValue(iTau,0);
	    continue;
	  }
	}else{
	  trackerdiscriminator=thePFTauElementsOperators.discriminatorByIsolPFChargedHadrCandsN(theTauRef_XYZVector,MatchingConeMetric_,theMatchingConeSize,TrackerSignalConeMetric_,theTrackerSignalConeSize,TrackerIsolConeMetric_,theTrackerIsolConeSize,LeadTrack_minPt_,Track_minPt_,TrackerIsolAnnulus_Tracksmaxn_);
	}
	if (trackerdiscriminator==0){
	  theTauDiscriminatorByIsolation->setValue(iTau,0);
	  continue;
	}
      }
      if (ApplyDiscriminationByECALIsolation_){
	double theECALSignalConeSize=computeConeSize(theTauRef,ECALSignalConeSizeTFormula_,ECALSignalConeVariableSize_min_,ECALSignalConeVariableSize_max_);
	double theECALIsolConeSize=computeConeSize(theTauRef,ECALIsolConeSizeTFormula_,ECALIsolConeVariableSize_min_,ECALIsolConeVariableSize_max_);     	
	double ECALdiscriminator=thePFTauElementsOperators.discriminatorByIsolPFGammaCandsN(theTauRef_XYZVector,MatchingConeMetric_,theMatchingConeSize,ECALSignalConeMetric_,theECALSignalConeSize,ECALIsolConeMetric_,theECALIsolConeSize,UseOnlyChargedHadr_for_LeadCand_,LeadCand_minPt_,GammaCand_minPt_,ECALIsolAnnulus_Candsmaxn_);
	if (ECALdiscriminator==0){
	  theTauDiscriminatorByIsolation->setValue(iTau,0);
	  continue;
	}
      }
      if (ManipulateTracks_insteadofChargedHadrCands_){
	CaloTauElementsOperators theCaloTauElementsOperators(theTauRef); 
	if (!theCaloTauElementsOperators.leadTk(MatchingConeMetric_,theMatchingConeSize,LeadCand_minPt_)) theTauDiscriminatorByIsolation->setValue(iTau,0);
	else theTauDiscriminatorByIsolation->setValue(iTau,1);
      }else{
	if (UseOnlyChargedHadr_for_LeadCand_){
	  if (!thePFTauElementsOperators.leadPFChargedHadrCand(MatchingConeMetric_,theMatchingConeSize,LeadCand_minPt_)) theTauDiscriminatorByIsolation->setValue(iTau,0);
	  else theTauDiscriminatorByIsolation->setValue(iTau,1);
	}else{
	  if (!thePFTauElementsOperators.leadPFCand(MatchingConeMetric_,theMatchingConeSize,LeadCand_minPt_)) theTauDiscriminatorByIsolation->setValue(iTau,0);
	  else theTauDiscriminatorByIsolation->setValue(iTau,1);
	}
      }
    }
    
    if (!(*theTauRef).getcalojetRef()){}else{
      CaloTauElementsOperators theCaloTauElementsOperators(theTauRef); 
      if (ApplyDiscriminationByTrackerIsolation_){  
	double trackerdiscriminator;
	double theTrackerSignalConeSize=computeConeSize(theTauRef,TrackerSignalConeSizeTFormula_,TrackerSignalConeVariableSize_min_,TrackerSignalConeVariableSize_max_);
	double theTrackerIsolConeSize=computeConeSize(theTauRef,TrackerIsolConeSizeTFormula_,TrackerIsolConeVariableSize_min_,TrackerIsolConeVariableSize_max_);     	
	trackerdiscriminator=theCaloTauElementsOperators.discriminator(theTauRef_XYZVector,MatchingConeMetric_,theMatchingConeSize,LeadTrack_minPt_,Track_minPt_,TrackerSignalConeMetric_,theTrackerSignalConeSize,TrackerIsolConeMetric_,theTrackerIsolConeSize,TrackerIsolAnnulus_Tracksmaxn_);
	if (trackerdiscriminator==0){
	  theTauDiscriminatorByIsolation->setValue(iTau,0);
	  continue;
	}
      }
      if (!theCaloTauElementsOperators.leadTk(MatchingConeMetric_,theMatchingConeSize,LeadCand_minPt_)) theTauDiscriminatorByIsolation->setValue(iTau,0);
      else theTauDiscriminatorByIsolation->setValue(iTau,1);
    }
  }
  iEvent.put(theTauDiscriminatorByIsolation);
}
double DiscriminationByIsolation::computeConeSize(const TauRef& theTau,const TFormula& ConeSizeTFormula,double ConeSizeMin,double ConeSizeMax){
  double x=theTau->energy();
  double y=theTau->et();
  double ConeSize=ConeSizeTFormula.Eval(x,y);
  if (ConeSize<ConeSizeMin)ConeSize=ConeSizeMin;
  if (ConeSize>ConeSizeMax)ConeSize=ConeSizeMax;
  return ConeSize;
}
TFormula DiscriminationByIsolation::computeConeSizeTFormula(const string& ConeSizeFormula,const char* errorMessage){
//--- check functional form 
//    given as configuration parameter for matching and signal cone sizes;
//
//    The size of a cone may depend on the energy "E" and/or transverse energy "ET" of the tau-jet candidate.
//    Any functional form that is supported by ROOT's TFormula class can be used (e.g. "3.0/E", "0.25/sqrt(ET)")
//
//    replace "E"  by TFormula variable "x"
//            "ET"                      "y"
  string ConeSizeFormulaStr = ConeSizeFormula;
  replaceSubStr(ConeSizeFormulaStr,"E","x");
  replaceSubStr(ConeSizeFormulaStr,"ET","y");
  TFormula ConeSizeTFormula;
  ConeSizeTFormula.SetName("ConeSize");
  ConeSizeTFormula.SetTitle(ConeSizeFormulaStr.data()); // the function definition is actually stored in the "Title" data-member of the TFormula object
  int errorFlag = ConeSizeTFormula.Compile();
  if (errorFlag!= 0) {
    throw cms::Exception("") << "\n unsupported functional Form for " << errorMessage << " " << ConeSizeFormula << endl
			     << "Please check that the Definition in \"" << ConeSizeTFormula.GetName() << "\" only contains the variables \"E\" or \"ET\""
			     << " and Functions that are supported by ROOT's TFormular Class." << endl;
  }else return ConeSizeTFormula;
}
void DiscriminationByIsolation::replaceSubStr(string& s,const string& oldSubStr,const string& newSubStr)
{
//--- protect replacement algorithm
//    from case that oldSubStr and newSubStr are equal
//    (nothing to be done anyway)
  if ( oldSubStr == newSubStr ) return;

//--- protect replacement algorithm
//    from case that oldSubStr contains no characters
//    (i.e. matches everything)
  if ( oldSubStr.empty() ) return;

  const string::size_type lengthOldSubStr = oldSubStr.size();
  const string::size_type lengthNewSubStr = newSubStr.size();

  string::size_type positionPreviousMatch = 0;
  string::size_type positionNextMatch = 0;

//--- consecutively replace all occurences of oldSubStr by newSubStr;
//    keep iterating until no occurence of oldSubStr left
  while ( (positionNextMatch = s.find(oldSubStr, positionPreviousMatch)) != string::npos ) {
    s.replace(positionNextMatch, lengthOldSubStr, newSubStr);
    positionPreviousMatch = positionNextMatch + lengthNewSubStr;
  } 
}


