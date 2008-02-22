#include "RecoTauTag/TauTagTools/interface/CaloTauElementsOperators.h"

CaloTauElementsOperators::CaloTauElementsOperators(CaloTau& theCaloTau) : TauElementsOperators(theCaloTau),CaloTau_(theCaloTau),AreaMetric_recoElements_maxabsEta_(2.5){
  Tracks_=theCaloTau.caloTauTagInfoRef()->Tracks();
  EcalRecHits_=theCaloTau.caloTauTagInfoRef()->positionAndEnergyECALRecHits();
}
  vector<pair<math::XYZPoint,float> > CaloTauElementsOperators::EcalRecHitsInCone(const math::XYZVector& coneAxis,const string coneMetric,const double coneSize,const double EcalRecHit_minEt)const{
  vector<pair<math::XYZPoint,float> > theFilteredEcalRecHits;
  for (vector<pair<math::XYZPoint,float> >::const_iterator iEcalRecHit=EcalRecHits_.begin();iEcalRecHit!=EcalRecHits_.end();++iEcalRecHit) {
    if ((*iEcalRecHit).second*fabs(sin((*iEcalRecHit).first.theta()))>EcalRecHit_minEt)theFilteredEcalRecHits.push_back(*iEcalRecHit);
  }  
  vector<pair<math::XYZPoint,float> > theFilteredEcalRecHitsInCone;
  if (coneMetric=="DR"){
    theFilteredEcalRecHitsInCone=EcalRecHitsinCone_DRmetric_(coneAxis,metricDR_,coneSize,theFilteredEcalRecHits);
  }else if(coneMetric=="angle"){
    theFilteredEcalRecHitsInCone=EcalRecHitsinCone_Anglemetric_(coneAxis,metricAngle_,coneSize,theFilteredEcalRecHits);
  }else if(coneMetric=="area"){
    int errorFlag = 0;
    FixedAreaIsolationCone fixedAreaCone;
    fixedAreaCone.setAcceptanceLimit(AreaMetric_recoElements_maxabsEta_);
    double coneAngle=fixedAreaCone(coneAxis.theta(),coneAxis.phi(),0,coneSize,errorFlag);
    if (errorFlag!=0) return vector<pair<math::XYZPoint,float> >();
    theFilteredEcalRecHitsInCone=EcalRecHitsinCone_Anglemetric_(coneAxis,metricAngle_,coneAngle,theFilteredEcalRecHits);
  }else return vector<pair<math::XYZPoint,float> >(); 
  return theFilteredEcalRecHitsInCone;
}
vector<pair<math::XYZPoint,float> > CaloTauElementsOperators::EcalRecHitsInAnnulus(const math::XYZVector& coneAxis,const string innerconeMetric,const double innerconeSize,const string outerconeMetric,const double outerconeSize,const double EcalRecHit_minEt)const{     
  vector<pair<math::XYZPoint,float> > theFilteredEcalRecHits;
  for (vector<pair<math::XYZPoint,float> >::const_iterator iEcalRecHit=EcalRecHits_.begin();iEcalRecHit!=EcalRecHits_.end();++iEcalRecHit) {
    if ((*iEcalRecHit).second*fabs(sin((*iEcalRecHit).first.theta()))>EcalRecHit_minEt)theFilteredEcalRecHits.push_back(*iEcalRecHit);
  }  
  vector<pair<math::XYZPoint,float> > theFilteredEcalRecHitsInAnnulus;
  if (outerconeMetric=="DR"){
    if (innerconeMetric=="DR"){
      theFilteredEcalRecHitsInAnnulus=EcalRecHitsinAnnulus_innerDRouterDRmetrics_(coneAxis,metricDR_,innerconeSize,metricDR_,outerconeSize,theFilteredEcalRecHits);
    }else if(innerconeMetric=="angle"){
      theFilteredEcalRecHitsInAnnulus=EcalRecHitsinAnnulus_innerAngleouterDRmetrics_(coneAxis,metricAngle_,innerconeSize,metricDR_,outerconeSize,theFilteredEcalRecHits);
    }else if(innerconeMetric=="area"){
      int errorFlag=0;
      FixedAreaIsolationCone theFixedAreaSignalCone;
      theFixedAreaSignalCone.setAcceptanceLimit(AreaMetric_recoElements_maxabsEta_);
      double innercone_angle=theFixedAreaSignalCone(coneAxis.theta(),coneAxis.phi(),0,innerconeSize,errorFlag);
      if (errorFlag!=0)return vector<pair<math::XYZPoint,float> >();
      theFilteredEcalRecHitsInAnnulus=EcalRecHitsinAnnulus_innerAngleouterDRmetrics_(coneAxis,metricAngle_,innercone_angle,metricDR_,outerconeSize,theFilteredEcalRecHits);
    }else return vector<pair<math::XYZPoint,float> >();
  }else if(outerconeMetric=="angle"){
    if (innerconeMetric=="DR"){
      theFilteredEcalRecHitsInAnnulus=EcalRecHitsinAnnulus_innerDRouterAnglemetrics_(coneAxis,metricDR_,innerconeSize,metricAngle_,outerconeSize,theFilteredEcalRecHits);
    }else if(innerconeMetric=="angle"){
      theFilteredEcalRecHitsInAnnulus=EcalRecHitsinAnnulus_innerAngleouterAnglemetrics_(coneAxis,metricAngle_,innerconeSize,metricAngle_,outerconeSize,theFilteredEcalRecHits);
    }else if(innerconeMetric=="area"){
      int errorFlag=0;
      FixedAreaIsolationCone theFixedAreaSignalCone;
      theFixedAreaSignalCone.setAcceptanceLimit(AreaMetric_recoElements_maxabsEta_);
      double innercone_angle=theFixedAreaSignalCone(coneAxis.theta(),coneAxis.phi(),0,innerconeSize,errorFlag);
      if (errorFlag!=0)return theFilteredEcalRecHitsInAnnulus;
      theFilteredEcalRecHitsInAnnulus=EcalRecHitsinAnnulus_innerAngleouterAnglemetrics_(coneAxis,metricAngle_,innercone_angle,metricAngle_,outerconeSize,theFilteredEcalRecHits);
    }else return vector<pair<math::XYZPoint,float> >();
  }else if(outerconeMetric=="area"){
    int errorFlag=0;
    FixedAreaIsolationCone theFixedAreaSignalCone;
    theFixedAreaSignalCone.setAcceptanceLimit(AreaMetric_recoElements_maxabsEta_);
    if (innerconeMetric=="DR"){
      // not implemented yet
    }else if(innerconeMetric=="angle"){
      double outercone_angle=theFixedAreaSignalCone(coneAxis.theta(),coneAxis.phi(),innerconeSize,outerconeSize,errorFlag);    
      if (errorFlag!=0)return theFilteredEcalRecHitsInAnnulus;
      theFilteredEcalRecHitsInAnnulus=EcalRecHitsinAnnulus_innerAngleouterAnglemetrics_(coneAxis,metricAngle_,innerconeSize,metricAngle_,outercone_angle,theFilteredEcalRecHits);
    }else if(innerconeMetric=="area"){
      double innercone_angle=theFixedAreaSignalCone(coneAxis.theta(),coneAxis.phi(),0,innerconeSize,errorFlag);    
      if (errorFlag!=0)return theFilteredEcalRecHitsInAnnulus;
      double outercone_angle=theFixedAreaSignalCone(coneAxis.theta(),coneAxis.phi(),innercone_angle,outerconeSize,errorFlag);
      if (errorFlag!=0)return theFilteredEcalRecHitsInAnnulus;
      theFilteredEcalRecHitsInAnnulus=EcalRecHitsinAnnulus_innerAngleouterAnglemetrics_(coneAxis,metricAngle_,innercone_angle,metricAngle_,outercone_angle,theFilteredEcalRecHits);
    }else return vector<pair<math::XYZPoint,float> >();
  }
  return theFilteredEcalRecHitsInAnnulus;
}


