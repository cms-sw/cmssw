#ifndef CaloTauElementsOperators_H_
#define CaloTauElementsOperators_H_

#include "RecoTauTag/TauTagTools/interface/ElementsInCone.h"
#include "RecoTauTag/TauTagTools/interface/ElementsInAnnulus.h"
#include "PhysicsTools/IsolationUtils/interface/FixedAreaIsolationCone.h"
#include "PhysicsTools/Utilities/interface/DeltaR.h"
#include "PhysicsTools/Utilities/interface/Angle.h"

#include "DataFormats/TauReco/interface/Tau.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/JetReco/interface/Jet.h"

using namespace edm;
using namespace std;
using namespace reco;

class CaloTauElementsOperators{
 public:
  CaloTauElementsOperators(const TauRef& theTauRef){
    TauRef_=theTauRef;
    Tracks_=(*theTauRef).getSelectedTracks();
  }
  void setAreaMetricrecoElementsmaxabsEta(const double x) {AreaMetric_recoElements_maxabsEta_=x;} 
  // get leading (i.e. highest Pt track)
  const TrackRef leadTk(string matchingConeMetric,double matchingConeSize,double ptTrackMin)const;
  const TrackRef leadTk(const math::XYZVector& jetAxis,string matchingConeMetric,double matchingConeSize,double ptTrackMin)const;
  
  // get list of selected tracks within cone given as function argument;
  const TrackRefVector tracksInCone(const math::XYZVector& coneAxis,const string coneMetric,const double coneSize,const double ptTrackMin)const;
  // get list of selected tracks within annulus given as function argument;
  const TrackRefVector tracksInAnnulus(const math::XYZVector& coneAxis,const string innerconeMetric,const double innerconeSize,const string outerconeMetric,const double outerconeSize,const double ptTrackMin)const;
  
  // functions that allow to recompute discriminator value "on-the-fly"
  // return value: 0 = jet fails tau-jet selection
  //                1 = jet passes tau-jet selection
  double discriminator(const math::XYZVector& coneAxes, 
		       string matchingConeMetric,double matchingConeSize, double ptLeadingTrackMin, double ptOtherTracksMin, 
		       string signalConeMetric,double signalConeSize,string isolationConeMetric,double isolationConeSize, 
		       unsigned int isolationAnnulus_Tracksmaxn)const;
  // axes of signal and isolation cone taken as jet-axis;
  double discriminator(string matchingConeMetric,double matchingConeSize, double ptLeadingTrackMin, double ptOtherTracksMin, 
		       string signalConeMetric,double signalConeSize,string isolationConeMetric,double isolationConeSize, 
		       unsigned int isolationAnnulus_Tracksmaxn)const;
 private:
  TauRef TauRef_;
  double AreaMetric_recoElements_maxabsEta_;
  TrackRefVector Tracks_;  // track selection criteria applied
  // template objects for DR and Angle metrics
  DeltaR<math::XYZVector> metricDR_;
  Angle<math::XYZVector> metricAngle_;
  ElementsInCone<math::XYZVector,DeltaR<math::XYZVector>,reco::TrackCollection> TracksinCone_DRmetric_;
  ElementsInCone<math::XYZVector,Angle<math::XYZVector>,reco::TrackCollection> TracksinCone_Anglemetric_; 
  ElementsInAnnulus<math::XYZVector,DeltaR<math::XYZVector>,DeltaR<math::XYZVector>,reco::TrackCollection> TracksinAnnulus_innerDRouterDRmetrics_;
  ElementsInAnnulus<math::XYZVector,DeltaR<math::XYZVector>,Angle<math::XYZVector>,reco::TrackCollection> TracksinAnnulus_innerDRouterAnglemetrics_; 
  ElementsInAnnulus<math::XYZVector,Angle<math::XYZVector>,Angle<math::XYZVector>,reco::TrackCollection> TracksinAnnulus_innerAngleouterAnglemetrics_;
  ElementsInAnnulus<math::XYZVector,Angle<math::XYZVector>,DeltaR<math::XYZVector>,reco::TrackCollection> TracksinAnnulus_innerAngleouterDRmetrics_; 
  };
#endif

// * different possible metrics for a cone : "DR", "angle", "area"; 
