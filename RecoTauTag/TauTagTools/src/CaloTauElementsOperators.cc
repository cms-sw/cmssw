#include "RecoTauTag/TauTagTools/interface/CaloTauElementsOperators.h"

const TrackRefVector CaloTauElementsOperators::tracksInCone(const math::XYZVector& coneAxis,const string coneMetric,const double coneSize,const double ptTrackMin) const{
  TrackRefVector matchingTracks;
  if (coneMetric=="DR"){
    matchingTracks=TracksinCone_DRmetric_(coneAxis,metricDR_,coneSize,Tracks_);
  }else if(coneMetric=="angle"){
    matchingTracks=TracksinCone_Anglemetric_(coneAxis,metricAngle_,coneSize,Tracks_);
  }else if(coneMetric=="area"){
    int errorFlag = 0;
    FixedAreaIsolationCone fixedAreaCone;
    fixedAreaCone.setAcceptanceLimit(AreaMetric_recoElements_maxabsEta_);
    double coneAngle=fixedAreaCone(coneAxis.theta(),coneAxis.phi(),0,coneSize,errorFlag);
    if (errorFlag!=0) return TrackRefVector();
    matchingTracks=TracksinCone_Anglemetric_(coneAxis,metricAngle_,coneAngle,Tracks_);
  }else return TrackRefVector(); 
  TrackRefVector selectedTracks;
  for (TrackRefVector::const_iterator iTrack=matchingTracks.begin();iTrack!=matchingTracks.end();++iTrack) {
    if ((**iTrack).pt()>ptTrackMin )selectedTracks.push_back(*iTrack);
  }  
  return selectedTracks;
}
const TrackRefVector CaloTauElementsOperators::tracksInAnnulus(const math::XYZVector& myVector,const string innercone_metric,const double innercone_size,const string outercone_metric,const double outercone_size,const double minPt)const{     
  TrackRefVector tmp;
  if (outercone_metric=="DR"){
    if (innercone_metric=="DR"){
      tmp=TracksinAnnulus_innerDRouterDRmetrics_(myVector,metricDR_,innercone_size,metricDR_,outercone_size,Tracks_);
    }else if(innercone_metric=="angle"){
      tmp=TracksinAnnulus_innerAngleouterDRmetrics_(myVector,metricAngle_,innercone_size,metricDR_,outercone_size,Tracks_);
    }else if(innercone_metric=="area"){
      int errorFlag=0;
      FixedAreaIsolationCone theFixedAreaSignalCone;
      theFixedAreaSignalCone.setAcceptanceLimit(AreaMetric_recoElements_maxabsEta_);
      double innercone_angle=theFixedAreaSignalCone(myVector.theta(),myVector.phi(),0,innercone_size,errorFlag);
      if (errorFlag!=0)return TrackRefVector();
      tmp=TracksinAnnulus_innerAngleouterDRmetrics_(myVector,metricAngle_,innercone_angle,metricDR_,outercone_size,Tracks_);
    }else return TrackRefVector();
  }else if(outercone_metric=="angle"){
    if (innercone_metric=="DR"){
      tmp=TracksinAnnulus_innerDRouterAnglemetrics_(myVector,metricDR_,innercone_size,metricAngle_,outercone_size,Tracks_);
    }else if(innercone_metric=="angle"){
      tmp=TracksinAnnulus_innerAngleouterAnglemetrics_(myVector,metricAngle_,innercone_size,metricAngle_,outercone_size,Tracks_);
    }else if(innercone_metric=="area"){
      int errorFlag=0;
      FixedAreaIsolationCone theFixedAreaSignalCone;
      theFixedAreaSignalCone.setAcceptanceLimit(AreaMetric_recoElements_maxabsEta_);
      double innercone_angle=theFixedAreaSignalCone(myVector.theta(),myVector.phi(),0,innercone_size,errorFlag);
      if (errorFlag!=0)return tmp;
      tmp=TracksinAnnulus_innerAngleouterAnglemetrics_(myVector,metricAngle_,innercone_angle,metricAngle_,outercone_size,Tracks_);
    }else return TrackRefVector();
  }else if(outercone_metric=="area"){
    int errorFlag=0;
    FixedAreaIsolationCone theFixedAreaSignalCone;
    theFixedAreaSignalCone.setAcceptanceLimit(AreaMetric_recoElements_maxabsEta_);
    if (innercone_metric=="DR"){
      // not implemented yet
    }else if(innercone_metric=="angle"){
      double outercone_angle=theFixedAreaSignalCone(myVector.theta(),myVector.phi(),innercone_size,outercone_size,errorFlag);    
      if (errorFlag!=0)return tmp;
      tmp=TracksinAnnulus_innerAngleouterAnglemetrics_(myVector,metricAngle_,innercone_size,metricAngle_,outercone_angle,Tracks_);
    }else if(innercone_metric=="area"){
      double innercone_angle=theFixedAreaSignalCone(myVector.theta(),myVector.phi(),0,innercone_size,errorFlag);    
      if (errorFlag!=0)return tmp;
      double outercone_angle=theFixedAreaSignalCone(myVector.theta(),myVector.phi(),innercone_angle,outercone_size,errorFlag);
      if (errorFlag!=0)return tmp;
      tmp=TracksinAnnulus_innerAngleouterAnglemetrics_(myVector,metricAngle_,innercone_angle,metricAngle_,outercone_angle,Tracks_);
    }else return TrackRefVector();
  }
  TrackRefVector selectedTracks;
  for (TrackRefVector::const_iterator iTrack=tmp.begin();iTrack!=tmp.end();++iTrack) {
    if ((**iTrack).pt()>minPt)selectedTracks.push_back(*iTrack);
  }  
  return selectedTracks;
}

const TrackRef CaloTauElementsOperators::leadTk(string matchingConeMetric,double matchingConeSize,double ptTrackMin)const{
  return leadTk((*TauRef_).momentum(),matchingConeMetric,matchingConeSize,ptTrackMin);
}

const TrackRef CaloTauElementsOperators::leadTk(const math::XYZVector& jetAxis,string matchingConeMetric,double matchingConeSize,double ptTrackMin)const{
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
double CaloTauElementsOperators::discriminator(const math::XYZVector& jetAxis, 
					       string matchingConeMetric,double matchingConeSize,double ptLeadingTrackMin,double ptOtherTracksMin,
					       string signalConeMetric,double signalConeSize,string isolationConeMetric,double isolationConeSize, 
					       unsigned int isolationAnnulus_Tracksmaxn)const{
  const TrackRef leadingTrack=leadTk(jetAxis,matchingConeMetric,matchingConeSize,ptLeadingTrackMin);
  if ( !leadingTrack ) return 0.; 
  math::XYZVector coneAxis=leadingTrack->momentum();
  TrackRefVector isolationAnnulusTracks=tracksInAnnulus(coneAxis,signalConeMetric,signalConeSize,isolationConeMetric,isolationConeSize,ptOtherTracksMin);
  if ((uint)isolationAnnulusTracks.size()>isolationAnnulus_Tracksmaxn)return 0.;
  else return 1.;
}
double CaloTauElementsOperators::discriminator(string matchingConeMetric,double matchingConeSize,double ptLeadingTrackMin,double ptOtherTracksMin, 
					       string signalConeMetric,double signalConeSize,string isolationConeMetric,double isolationConeSize, 
					       unsigned int isolationAnnulus_Tracksmaxn)const{
  return discriminator((*TauRef_).momentum(),matchingConeMetric,matchingConeSize,ptLeadingTrackMin,ptOtherTracksMin,signalConeMetric,signalConeSize,isolationConeMetric,isolationConeSize,isolationAnnulus_Tracksmaxn);
}

