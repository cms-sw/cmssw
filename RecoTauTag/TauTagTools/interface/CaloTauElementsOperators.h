#ifndef RecoTauTag_RecoTau_CaloTauElementsOperators_H_
#define RecoTauTag_RecoTau_CaloTauElementsOperators_H_

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/TauReco/interface/CaloTau.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/JetReco/interface/Jet.h"

#include "RecoTauTag/TauTagTools/interface/TauElementsOperators.h"
#include "RecoTauTag/TauTagTools/interface/ElementsInCone.h"
#include "RecoTauTag/TauTagTools/interface/ElementsInAnnulus.h"
#include "PhysicsTools/IsolationUtils/interface/FixedAreaIsolationCone.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "CommonTools/Utils/interface/Angle.h"

class CaloTauElementsOperators : public TauElementsOperators {
 public:
  CaloTauElementsOperators(reco::CaloTau&);
  ~CaloTauElementsOperators(){} 
  
  // return all Ecal RecHits in a cone of metric* "coneMetric" and size "coneSize" around a direction "coneAxis" 
  std::vector<std::pair<math::XYZPoint,float> > EcalRecHitsInCone(const math::XYZVector& coneAxis,const std::string coneMetric,const double coneSize,const double EcalRecHit_minEt,const std::vector<std::pair<math::XYZPoint,float> >& myEcalRecHits)const;
  // return all Ecal RecHits in an annulus defined by inner(metric* "innerconeMetric" and size "innerconeSize") and outer(metric* "outerconeMetric" and size "outerconeSize") cones around a direction "coneAxis" 
  std::vector<std::pair<math::XYZPoint,float> > EcalRecHitsInAnnulus(const math::XYZVector& coneAxis,const std::string innerconeMetric,const double innerconeSize,const std::string outerconeMetric,const double outerconeSize,const double EcalRecHit_minEt,const std::vector<std::pair<math::XYZPoint,float> >& myEcalRecHits)const;

  // These function kept for out-of-box compatability with 2_2_X
  std::vector<std::pair<math::XYZPoint,float> > EcalRecHitsInCone(const math::XYZVector& coneAxis,const std::string coneMetric,const double coneSize,const double EcalRecHit_minEt)const;
  std::vector<std::pair<math::XYZPoint,float> > EcalRecHitsInAnnulus(const math::XYZVector& coneAxis,const std::string innerconeMetric,const double innerconeSize,const std::string outerconeMetric,const double outerconeSize,const double EcalRecHit_minEt)const; 

  // return all neutral Ecal BasicClusters in a cone of metric* "coneMetric" and size "coneSize" around a direction "coneAxis" 
  std::vector<std::pair<math::XYZPoint,float> > neutralEcalBasicClustersInCone(const math::XYZVector& coneAxis,const std::string coneMetric,const double coneSize,const double neutralEcalBasicCluster_minEt)const;
  // return all neutral Ecal BasicClusters in an annulus defined by inner(metric* "innerconeMetric" and size "innerconeSize") and outer(metric* "outerconeMetric" and size "outerconeSize") cones around a direction "coneAxis" 
  std::vector<std::pair<math::XYZPoint,float> > neutralEcalBasicClustersInAnnulus(const math::XYZVector& coneAxis,const std::string innerconeMetric,const double innerconeSize,const std::string outerconeMetric,const double outerconeSize,const double neutralEcalBasicCluster_minEt)const; 
  
  void setAreaMetricrecoElementsmaxabsEta(const double x) {AreaMetric_recoElements_maxabsEta_=x;}   
 private:
  reco::CaloTau& CaloTau_;
  std::vector<std::pair<math::XYZPoint,float> > EcalRecHits_;
  double AreaMetric_recoElements_maxabsEta_;
  // template objects for DR and Angle metrics
  DeltaR<math::XYZVector,math::XYZPoint> metricDR_;
  Angle<math::XYZVector,math::XYZPoint> metricAngle_;
  ElementsInCone<math::XYZVector,DeltaR<math::XYZVector,math::XYZPoint>,std::pair<math::XYZPoint,float> > EcalRecHitsinCone_DRmetric_;
  ElementsInCone<math::XYZVector,Angle<math::XYZVector,math::XYZPoint>,std::pair<math::XYZPoint,float> > EcalRecHitsinCone_Anglemetric_; 
  ElementsInAnnulus<math::XYZVector,DeltaR<math::XYZVector,math::XYZPoint>,DeltaR<math::XYZVector,math::XYZPoint>,std::pair<math::XYZPoint,float> > EcalRecHitsinAnnulus_innerDRouterDRmetrics_;
  ElementsInAnnulus<math::XYZVector,DeltaR<math::XYZVector,math::XYZPoint>,Angle<math::XYZVector,math::XYZPoint>,std::pair<math::XYZPoint,float> > EcalRecHitsinAnnulus_innerDRouterAnglemetrics_; 
  ElementsInAnnulus<math::XYZVector,Angle<math::XYZVector,math::XYZPoint>,Angle<math::XYZVector,math::XYZPoint>,std::pair<math::XYZPoint,float> > EcalRecHitsinAnnulus_innerAngleouterAnglemetrics_;
  ElementsInAnnulus<math::XYZVector,Angle<math::XYZVector,math::XYZPoint>,DeltaR<math::XYZVector,math::XYZPoint>,std::pair<math::XYZPoint,float> > EcalRecHitsinAnnulus_innerAngleouterDRmetrics_; 
};
#endif

// * different possible metrics for a cone : "DR", "angle", "area"; 
