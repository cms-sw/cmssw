#include "RecoTauTag/TauTagTools/interface/TauElementsOperators.h"

using namespace reco;
using std::string;

TauElementsOperators::TauElementsOperators(BaseTau& theBaseTau) : BaseTau_(theBaseTau),AreaMetric_recoElements_maxabsEta_(2.5){
  IsolTracks_=theBaseTau.isolationTracks();
}
  
double TauElementsOperators::computeConeSize(const TFormula& ConeSizeTFormula,double ConeSizeMin,double ConeSizeMax){
  double x=BaseTau_.energy();
  double y=BaseTau_.et();
  double ConeSize=ConeSizeTFormula.Eval(x,y);
  if (ConeSize<ConeSizeMin)ConeSize=ConeSizeMin;
  if (ConeSize>ConeSizeMax)ConeSize=ConeSizeMax;
  return ConeSize;
}

double TauElementsOperators::computeConeSize(const TFormula& ConeSizeTFormula,double ConeSizeMin,double ConeSizeMax, double transverseEnergy, double energy, double jetOpeningAngle){
  double ConeSize=ConeSizeTFormula.Eval(energy, transverseEnergy, jetOpeningAngle);
  if (ConeSize<ConeSizeMin)ConeSize=ConeSizeMin;
  if (ConeSize>ConeSizeMax)ConeSize=ConeSizeMax;
  return ConeSize;
}

/*
 * DEPRECATED????
TFormula  TauElementsOperators::computeConeSizeTFormula(const string& ConeSizeFormula,const char* errorMessage){
  //--- check functional form 
  //    given as configuration parameter for matching and signal cone sizes;
  //
  //    The size of a cone may depend on the energy "E" and/or transverse energy "ET" of the tau-jet candidate.
  //    Any functional form that is supported by ROOT's TFormula class can be used (e.g. "3.0/E", "0.25/sqrt(ET)")
  //
  //    replace "E"  by TFormula variable "x"
  //            "ET"                      "y"
  string ConeSizeFormulaStr = ConeSizeFormula;
  replaceSubStr(ConeSizeFormulaStr,"ET","y");
  replaceSubStr(ConeSizeFormulaStr,"E","x");
  ConeSizeTFormula.SetName("ConeSize");
  ConeSizeTFormula.SetTitle(ConeSizeFormulaStr.data()); // the function definition is actually stored in the "Title" data-member of the TFormula object
  int errorFlag = ConeSizeTFormula.Compile();
  if (errorFlag!= 0) {
    throw cms::Exception("") << "\n unsupported functional Form for " << errorMessage << " " << ConeSizeFormula << endl
			     << "Please check that the Definition in \"" << ConeSizeTFormula.GetName() << "\" only contains the variables \"E\" or \"ET\""
			     << " and Functions that are supported by ROOT's TFormular Class." << endl;
  }else return ConeSizeTFormula;
}
*/



void TauElementsOperators::replaceSubStr(string& s,const string& oldSubStr,const string& newSubStr){
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

const TrackRefVector TauElementsOperators::tracksInCone(const math::XYZVector& coneAxis,const string coneMetric,const double coneSize,const double ptTrackMin) const{
  TrackRefVector theFilteredTracks;
  for (TrackRefVector::const_iterator iTrack=Tracks_.begin();iTrack!=Tracks_.end();++iTrack) {
    if ((**iTrack).pt()>ptTrackMin)theFilteredTracks.push_back(*iTrack);
  }  
  TrackRefVector theFilteredTracksInCone;
  if (coneMetric=="DR"){
    theFilteredTracksInCone=TracksinCone_DRmetric_(coneAxis,metricDR_,coneSize,theFilteredTracks);
  }else if(coneMetric=="angle"){
    theFilteredTracksInCone=TracksinCone_Anglemetric_(coneAxis,metricAngle_,coneSize,theFilteredTracks);
  }else if(coneMetric=="area"){
    int errorFlag = 0;
    FixedAreaIsolationCone fixedAreaCone;
    fixedAreaCone.setAcceptanceLimit(AreaMetric_recoElements_maxabsEta_);
    double coneAngle=fixedAreaCone(coneAxis.theta(),coneAxis.phi(),0,coneSize,errorFlag);
    if (errorFlag!=0) return TrackRefVector();
    theFilteredTracksInCone=TracksinCone_Anglemetric_(coneAxis,metricAngle_,coneAngle,theFilteredTracks);
  }else return TrackRefVector(); 
  return theFilteredTracksInCone;
}
const TrackRefVector TauElementsOperators::tracksInCone(const math::XYZVector& coneAxis,const string coneMetric,const double coneSize,const double ptTrackMin,const double tracktorefpoint_maxDZ,const double refpoint_Z, const Vertex &myPV) const{
  TrackRefVector theFilteredTracks;
  for (TrackRefVector::const_iterator iTrack=Tracks_.begin();iTrack!=Tracks_.end();++iTrack) {
    if ((**iTrack).pt()>ptTrackMin && fabs((**iTrack).dz(myPV.position())-refpoint_Z)<=tracktorefpoint_maxDZ)theFilteredTracks.push_back(*iTrack);
  }  
  TrackRefVector theFilteredTracksInCone;
  if (coneMetric=="DR"){
    theFilteredTracksInCone=TracksinCone_DRmetric_(coneAxis,metricDR_,coneSize,theFilteredTracks);
  }else if(coneMetric=="angle"){
    theFilteredTracksInCone=TracksinCone_Anglemetric_(coneAxis,metricAngle_,coneSize,theFilteredTracks);
  }else if(coneMetric=="area"){
    int errorFlag = 0;
    FixedAreaIsolationCone fixedAreaCone;
    fixedAreaCone.setAcceptanceLimit(AreaMetric_recoElements_maxabsEta_);
    double coneAngle=fixedAreaCone(coneAxis.theta(),coneAxis.phi(),0,coneSize,errorFlag);
    if (errorFlag!=0) return TrackRefVector();
    theFilteredTracksInCone=TracksinCone_Anglemetric_(coneAxis,metricAngle_,coneAngle,theFilteredTracks);
  }else return TrackRefVector(); 
  return theFilteredTracksInCone;
}
const TrackRefVector TauElementsOperators::tracksInAnnulus(const math::XYZVector& myVector,const string innercone_metric,const double innercone_size,const string outercone_metric,const double outercone_size,const double minPt)const{     
  TrackRefVector theFilteredTracks;
  for (TrackRefVector::const_iterator iTrack=Tracks_.begin();iTrack!=Tracks_.end();++iTrack) {
    if ((**iTrack).pt()>minPt)theFilteredTracks.push_back(*iTrack);
  }  
  TrackRefVector theFilteredTracksInAnnulus;
  if (outercone_metric=="DR"){
    if (innercone_metric=="DR"){
      theFilteredTracksInAnnulus=TracksinAnnulus_innerDRouterDRmetrics_(myVector,metricDR_,innercone_size,metricDR_,outercone_size,theFilteredTracks);
    }else if(innercone_metric=="angle"){
      theFilteredTracksInAnnulus=TracksinAnnulus_innerAngleouterDRmetrics_(myVector,metricAngle_,innercone_size,metricDR_,outercone_size,theFilteredTracks);
    }else if(innercone_metric=="area"){
      int errorFlag=0;
      FixedAreaIsolationCone theFixedAreaSignalCone;
      theFixedAreaSignalCone.setAcceptanceLimit(AreaMetric_recoElements_maxabsEta_);
      double innercone_angle=theFixedAreaSignalCone(myVector.theta(),myVector.phi(),0,innercone_size,errorFlag);
      if (errorFlag!=0)return TrackRefVector();
      theFilteredTracksInAnnulus=TracksinAnnulus_innerAngleouterDRmetrics_(myVector,metricAngle_,innercone_angle,metricDR_,outercone_size,theFilteredTracks);
    }else return TrackRefVector();
  }else if(outercone_metric=="angle"){
    if (innercone_metric=="DR"){
      theFilteredTracksInAnnulus=TracksinAnnulus_innerDRouterAnglemetrics_(myVector,metricDR_,innercone_size,metricAngle_,outercone_size,theFilteredTracks);
    }else if(innercone_metric=="angle"){
      theFilteredTracksInAnnulus=TracksinAnnulus_innerAngleouterAnglemetrics_(myVector,metricAngle_,innercone_size,metricAngle_,outercone_size,theFilteredTracks);
    }else if(innercone_metric=="area"){
      int errorFlag=0;
      FixedAreaIsolationCone theFixedAreaSignalCone;
      theFixedAreaSignalCone.setAcceptanceLimit(AreaMetric_recoElements_maxabsEta_);
      double innercone_angle=theFixedAreaSignalCone(myVector.theta(),myVector.phi(),0,innercone_size,errorFlag);
      if (errorFlag!=0)return theFilteredTracksInAnnulus;
      theFilteredTracksInAnnulus=TracksinAnnulus_innerAngleouterAnglemetrics_(myVector,metricAngle_,innercone_angle,metricAngle_,outercone_size,theFilteredTracks);
    }else return TrackRefVector();
  }else if(outercone_metric=="area"){
    int errorFlag=0;
    FixedAreaIsolationCone theFixedAreaSignalCone;
    theFixedAreaSignalCone.setAcceptanceLimit(AreaMetric_recoElements_maxabsEta_);
    if (innercone_metric=="DR"){
      // not implemented yet
    }else if(innercone_metric=="angle"){
      double outercone_angle=theFixedAreaSignalCone(myVector.theta(),myVector.phi(),innercone_size,outercone_size,errorFlag);    
      if (errorFlag!=0)return theFilteredTracksInAnnulus;
      theFilteredTracksInAnnulus=TracksinAnnulus_innerAngleouterAnglemetrics_(myVector,metricAngle_,innercone_size,metricAngle_,outercone_angle,theFilteredTracks);
    }else if(innercone_metric=="area"){
      double innercone_angle=theFixedAreaSignalCone(myVector.theta(),myVector.phi(),0,innercone_size,errorFlag);    
      if (errorFlag!=0)return theFilteredTracksInAnnulus;
      double outercone_angle=theFixedAreaSignalCone(myVector.theta(),myVector.phi(),innercone_angle,outercone_size,errorFlag);
      if (errorFlag!=0)return theFilteredTracksInAnnulus;
      theFilteredTracksInAnnulus=TracksinAnnulus_innerAngleouterAnglemetrics_(myVector,metricAngle_,innercone_angle,metricAngle_,outercone_angle,theFilteredTracks);
    }else return TrackRefVector();
  }
  return theFilteredTracksInAnnulus;
}
const TrackRefVector TauElementsOperators::tracksInAnnulus(const math::XYZVector& myVector,const string innercone_metric,const double innercone_size,const string outercone_metric,const double outercone_size,const double minPt,const double tracktorefpoint_maxDZ,const double refpoint_Z, const Vertex &myPV)const{     
  TrackRefVector theFilteredTracks;
  for (TrackRefVector::const_iterator iTrack=Tracks_.begin();iTrack!=Tracks_.end();++iTrack) {
    if ((**iTrack).pt()>minPt && fabs((**iTrack).dz(myPV.position())-refpoint_Z)<=tracktorefpoint_maxDZ)theFilteredTracks.push_back(*iTrack);
  }  
  TrackRefVector theFilteredTracksInAnnulus;
  if (outercone_metric=="DR"){
    if (innercone_metric=="DR"){
      theFilteredTracksInAnnulus=TracksinAnnulus_innerDRouterDRmetrics_(myVector,metricDR_,innercone_size,metricDR_,outercone_size,theFilteredTracks);
    }else if(innercone_metric=="angle"){
      theFilteredTracksInAnnulus=TracksinAnnulus_innerAngleouterDRmetrics_(myVector,metricAngle_,innercone_size,metricDR_,outercone_size,theFilteredTracks);
    }else if(innercone_metric=="area"){
      int errorFlag=0;
      FixedAreaIsolationCone theFixedAreaSignalCone;
      theFixedAreaSignalCone.setAcceptanceLimit(AreaMetric_recoElements_maxabsEta_);
      double innercone_angle=theFixedAreaSignalCone(myVector.theta(),myVector.phi(),0,innercone_size,errorFlag);
      if (errorFlag!=0)return TrackRefVector();
      theFilteredTracksInAnnulus=TracksinAnnulus_innerAngleouterDRmetrics_(myVector,metricAngle_,innercone_angle,metricDR_,outercone_size,theFilteredTracks);
    }else return TrackRefVector();
  }else if(outercone_metric=="angle"){
    if (innercone_metric=="DR"){
      theFilteredTracksInAnnulus=TracksinAnnulus_innerDRouterAnglemetrics_(myVector,metricDR_,innercone_size,metricAngle_,outercone_size,theFilteredTracks);
    }else if(innercone_metric=="angle"){
      theFilteredTracksInAnnulus=TracksinAnnulus_innerAngleouterAnglemetrics_(myVector,metricAngle_,innercone_size,metricAngle_,outercone_size,theFilteredTracks);
    }else if(innercone_metric=="area"){
      int errorFlag=0;
      FixedAreaIsolationCone theFixedAreaSignalCone;
      theFixedAreaSignalCone.setAcceptanceLimit(AreaMetric_recoElements_maxabsEta_);
      double innercone_angle=theFixedAreaSignalCone(myVector.theta(),myVector.phi(),0,innercone_size,errorFlag);
      if (errorFlag!=0)return theFilteredTracksInAnnulus;
      theFilteredTracksInAnnulus=TracksinAnnulus_innerAngleouterAnglemetrics_(myVector,metricAngle_,innercone_angle,metricAngle_,outercone_size,theFilteredTracks);
    }else return TrackRefVector();
  }else if(outercone_metric=="area"){
    int errorFlag=0;
    FixedAreaIsolationCone theFixedAreaSignalCone;
    theFixedAreaSignalCone.setAcceptanceLimit(AreaMetric_recoElements_maxabsEta_);
    if (innercone_metric=="DR"){
      // not implemented yet
    }else if(innercone_metric=="angle"){
      double outercone_angle=theFixedAreaSignalCone(myVector.theta(),myVector.phi(),innercone_size,outercone_size,errorFlag);    
      if (errorFlag!=0)return theFilteredTracksInAnnulus;
      theFilteredTracksInAnnulus=TracksinAnnulus_innerAngleouterAnglemetrics_(myVector,metricAngle_,innercone_size,metricAngle_,outercone_angle,theFilteredTracks);
    }else if(innercone_metric=="area"){
      double innercone_angle=theFixedAreaSignalCone(myVector.theta(),myVector.phi(),0,innercone_size,errorFlag);    
      if (errorFlag!=0)return theFilteredTracksInAnnulus;
      double outercone_angle=theFixedAreaSignalCone(myVector.theta(),myVector.phi(),innercone_angle,outercone_size,errorFlag);
      if (errorFlag!=0)return theFilteredTracksInAnnulus;
      theFilteredTracksInAnnulus=TracksinAnnulus_innerAngleouterAnglemetrics_(myVector,metricAngle_,innercone_angle,metricAngle_,outercone_angle,theFilteredTracks);
    }else return TrackRefVector();
  }
  return theFilteredTracksInAnnulus;
}

const TrackRef TauElementsOperators::leadTk(string matchingConeMetric,double matchingConeSize,double ptTrackMin)const{
  return leadTk(BaseTau_.momentum(),matchingConeMetric,matchingConeSize,ptTrackMin);
}

const TrackRef TauElementsOperators::leadTk(const math::XYZVector& jetAxis,string matchingConeMetric,double matchingConeSize,double ptTrackMin)const{
  const TrackRefVector matchingConeTracks=tracksInCone(jetAxis,matchingConeMetric,matchingConeSize,ptTrackMin);
  if ((int)matchingConeTracks.size()==0) return TrackRef();
  TrackRef leadingTrack;
  double leadingTrackPt=0.;
  for (TrackRefVector::const_iterator track=matchingConeTracks.begin();track!=matchingConeTracks.end();++track) {
    if ((*track)->pt()>ptTrackMin && (*track)->pt()>leadingTrackPt){
      leadingTrack=(*track);
      leadingTrackPt=leadingTrack->pt();
    }
  }  
  return leadingTrack;
}
// ***
double TauElementsOperators::discriminatorByIsolTracksN(unsigned int isolationAnnulus_Tracksmaxn)const{
  if ((unsigned int)IsolTracks_.size()>isolationAnnulus_Tracksmaxn)return 0.;
  else return 1.;
}
double TauElementsOperators::discriminatorByIsolTracksN(const math::XYZVector& jetAxis, 
							string matchingConeMetric,double matchingConeSize,double ptLeadingTrackMin,double ptOtherTracksMin,
							string signalConeMetric,double signalConeSize,string isolationConeMetric,double isolationConeSize, 
							unsigned int isolationAnnulus_Tracksmaxn)const{
  const TrackRef leadingTrack=leadTk(jetAxis,matchingConeMetric,matchingConeSize,ptLeadingTrackMin);
  if(!leadingTrack)return 0.; 
  math::XYZVector coneAxis=leadingTrack->momentum();
  TrackRefVector isolationAnnulusTracks=tracksInAnnulus(coneAxis,signalConeMetric,signalConeSize,isolationConeMetric,isolationConeSize,ptOtherTracksMin);
  if ((unsigned int)isolationAnnulusTracks.size()>isolationAnnulus_Tracksmaxn)return 0.;
  else return 1.;
}
double TauElementsOperators::discriminatorByIsolTracksN(string matchingConeMetric,double matchingConeSize,double ptLeadingTrackMin,double ptOtherTracksMin, 
							string signalConeMetric,double signalConeSize,string isolationConeMetric,double isolationConeSize, 
							unsigned int isolationAnnulus_Tracksmaxn)const{
  return discriminatorByIsolTracksN(BaseTau_.momentum(),matchingConeMetric,matchingConeSize,ptLeadingTrackMin,ptOtherTracksMin,signalConeMetric,signalConeSize,isolationConeMetric,isolationConeSize,isolationAnnulus_Tracksmaxn);
}
