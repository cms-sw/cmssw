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
#include "PhysicsTools/Utilities/interface/DeltaR.h"
#include "PhysicsTools/Utilities/interface/Angle.h"

using namespace edm;
using namespace std;
using namespace reco;

class CaloTauElementsOperators : public TauElementsOperators {
 public:
  CaloTauElementsOperators(CaloTau&);
  ~CaloTauElementsOperators(){} 
  
  // return all Ecal RecHits in a cone of metric* "coneMetric" and size "coneSize" around a direction "coneAxis" 
  vector<pair<math::XYZPoint,float> > EcalRecHitsInCone(const math::XYZVector& coneAxis,const string coneMetric,const double coneSize,const double EcalRecHit_minEt)const;
  // return all Ecal RecHits in an annulus defined by inner(metric* "innerconeMetric" and size "innerconeSize") and outer(metric* "outerconeMetric" and size "outerconeSize") cones around a direction "coneAxis" 
  vector<pair<math::XYZPoint,float> > EcalRecHitsInAnnulus(const math::XYZVector& coneAxis,const string innerconeMetric,const double innerconeSize,const string outerconeMetric,const double outerconeSize,const double EcalRecHit_minEt)const; 
  
  // return all neutral Ecal BasicClusters in a cone of metric* "coneMetric" and size "coneSize" around a direction "coneAxis" 
  vector<pair<math::XYZPoint,float> > neutralEcalBasicClustersInCone(const math::XYZVector& coneAxis,const string coneMetric,const double coneSize,const double neutralEcalBasicCluster_minEt)const;
  // return all neutral Ecal BasicClusters in an annulus defined by inner(metric* "innerconeMetric" and size "innerconeSize") and outer(metric* "outerconeMetric" and size "outerconeSize") cones around a direction "coneAxis" 
  vector<pair<math::XYZPoint,float> > neutralEcalBasicClustersInAnnulus(const math::XYZVector& coneAxis,const string innerconeMetric,const double innerconeSize,const string outerconeMetric,const double outerconeSize,const double neutralEcalBasicCluster_minEt)const; 
  
  void setAreaMetricrecoElementsmaxabsEta(const double x) {AreaMetric_recoElements_maxabsEta_=x;}   
 private:
  CaloTau& CaloTau_;
  vector<pair<math::XYZPoint,float> > EcalRecHits_;
  double AreaMetric_recoElements_maxabsEta_;
  // template objects for DR and Angle metrics
  DeltaR<math::XYZVector,math::XYZPoint> metricDR_;
  Angle<math::XYZVector,math::XYZPoint> metricAngle_;
  ElementsInCone<math::XYZVector,DeltaR<math::XYZVector,math::XYZPoint>,pair<math::XYZPoint,float> > EcalRecHitsinCone_DRmetric_;
  ElementsInCone<math::XYZVector,Angle<math::XYZVector,math::XYZPoint>,pair<math::XYZPoint,float> > EcalRecHitsinCone_Anglemetric_; 
  ElementsInAnnulus<math::XYZVector,DeltaR<math::XYZVector,math::XYZPoint>,DeltaR<math::XYZVector,math::XYZPoint>,pair<math::XYZPoint,float> > EcalRecHitsinAnnulus_innerDRouterDRmetrics_;
  ElementsInAnnulus<math::XYZVector,DeltaR<math::XYZVector,math::XYZPoint>,Angle<math::XYZVector,math::XYZPoint>,pair<math::XYZPoint,float> > EcalRecHitsinAnnulus_innerDRouterAnglemetrics_; 
  ElementsInAnnulus<math::XYZVector,Angle<math::XYZVector,math::XYZPoint>,Angle<math::XYZVector,math::XYZPoint>,pair<math::XYZPoint,float> > EcalRecHitsinAnnulus_innerAngleouterAnglemetrics_;
  ElementsInAnnulus<math::XYZVector,Angle<math::XYZVector,math::XYZPoint>,DeltaR<math::XYZVector,math::XYZPoint>,pair<math::XYZPoint,float> > EcalRecHitsinAnnulus_innerAngleouterDRmetrics_; 
};
#endif

// * different possible metrics for a cone : "DR", "angle", "area"; 
