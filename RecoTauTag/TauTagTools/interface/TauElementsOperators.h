#ifndef RecoTauTag_RecoTau_TauElementsOperators_H_
#define RecoTauTag_RecoTau_TauElementsOperators_H_

#include "DataFormats/TauReco/interface/BaseTau.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "RecoTauTag/TauTagTools/interface/ElementsInCone.h"
#include "RecoTauTag/TauTagTools/interface/ElementsInAnnulus.h"

#include "PhysicsTools/IsolationUtils/interface/FixedAreaIsolationCone.h"
#include "PhysicsTools/Utilities/interface/DeltaR.h"
#include "PhysicsTools/Utilities/interface/Angle.h"

#include "TFormula.h"

using namespace edm;
using namespace std;
using namespace reco;

class TauElementsOperators{
 public:
  TauElementsOperators(BaseTau&);
  ~TauElementsOperators(){}   
  // compute size of signal cone possibly depending on E(energy) and/or ET(transverse energy) of the tau-jet candidate
  double computeConeSize(const TFormula& ConeSizeTFormula,double ConeSizeMin,double ConeSizeMax);
  TFormula computeConeSizeTFormula(const string& ConeSizeFormula,const char* errorMessage);
  void replaceSubStr(string& s,const string& oldSubStr,const string& newSubStr); 
  //return the leading (i.e. highest Pt) Track in a given cone around the jet axis or a given direction
  const TrackRef leadTk(string matchingConeMetric,double matchingConeSize,double ptTrackMin)const;
  const TrackRef leadTk(const math::XYZVector& jetAxis,string matchingConeMetric,double matchingConeSize,double ptTrackMin)const;
  
  // return all Tracks in a cone of metric* "coneMetric" and size "coneSize" around a direction "coneAxis" 
  const TrackRefVector tracksInCone(const math::XYZVector& coneAxis,const string coneMetric,const double coneSize,const double ptTrackMin)const;
  const TrackRefVector tracksInCone(const math::XYZVector& coneAxis,const string coneMetric,const double coneSize,const double ptTrackMin,const double tracktorefpoint_maxDZ,const double refpoint_Z)const;
  // return all Tracks in an annulus defined by inner(metric* "innerconeMetric" and size "innerconeSize") and outer(metric* "outerconeMetric" and size "outerconeSize") cones around a direction "coneAxis" 
  const TrackRefVector tracksInAnnulus(const math::XYZVector& coneAxis,const string innerconeMetric,const double innerconeSize,const string outerconeMetric,const double outerconeSize,const double ptTrackMin)const;  
  const TrackRefVector tracksInAnnulus(const math::XYZVector& coneAxis,const string innerconeMetric,const double innerconeSize,const string outerconeMetric,const double outerconeSize,const double ptTrackMin,const double tracktorefpoint_maxDZ,const double refpoint_Z)const;  
  // return 1 if no/low Tracks activity in an isolation annulus around a leading Track, 0 otherwise; 
  // different possible metrics* for the matching, signal and isolation cones; 
  double discriminatorByIsolTracksN(unsigned int isolationAnnulus_Tracksmaxn)const;
  double discriminatorByIsolTracksN(const math::XYZVector& coneAxis, 
				    string matchingConeMetric,double matchingConeSize, double ptLeadingTrackMin, double ptOtherTracksMin, 
				    string signalConeMetric,double signalConeSize,string isolationConeMetric,double isolationConeSize, 
				    unsigned int isolationAnnulus_Tracksmaxn)const;
  // matching cone axis is the jet axis, signal and isolation cones axes are a leading Track axis;
  double discriminatorByIsolTracksN(string matchingConeMetric,double matchingConeSize, double ptLeadingTrackMin, double ptOtherTracksMin, 
				    string signalConeMetric,double signalConeSize,string isolationConeMetric,double isolationConeSize, 
				    unsigned int isolationAnnulus_Tracksmaxn)const;
 protected:
  BaseTau& BaseTau_;
  double AreaMetric_recoElements_maxabsEta_;
  TrackRefVector Tracks_;  // track selection criteria applied
  TrackRefVector IsolTracks_;  // tracks in an isolation annulus, track selection criteria applied; 
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
