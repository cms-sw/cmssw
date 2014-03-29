#ifndef RecoTauTag_RecoTau_TauElementsOperators_H_
#define RecoTauTag_RecoTau_TauElementsOperators_H_

#include "DataFormats/TauReco/interface/BaseTau.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "RecoTauTag/TauTagTools/interface/ElementsInCone.h"
#include "RecoTauTag/TauTagTools/interface/ElementsInConeRef.h"
#include "RecoTauTag/TauTagTools/interface/ElementsInAnnulus.h"
#include "RecoTauTag/TauTagTools/interface/ElementsInAnnulusRef.h"

#include "PhysicsTools/IsolationUtils/interface/FixedAreaIsolationCone.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "CommonTools/Utils/interface/Angle.h"

#include "TFormula.h"

class TauElementsOperators{
 public:
  TauElementsOperators();
  TauElementsOperators(reco::BaseTau&);
  ~TauElementsOperators(){}   
  // compute size of signal cone possibly depending on E(energy) and/or ET(transverse energy), and/or seed Jet Opening DR of the tau-jet candidate
  double computeConeSize(const TFormula& ConeSizeTFormula,double ConeSizeMin,double ConeSizeMax);
  double computeConeSize(const TFormula& ConeSizeTFormula,double ConeSizeMin,double ConeSizeMax, double transverseEnergy, double energy, double jetOpeningAngle = 0.);

  //TFormula computeConeSizeTFormula(const std::string& ConeSizeFormula,const char* errorMessage);
  void replaceSubStr(std::string& s,const std::string& oldSubStr,const std::string& newSubStr); 
  //return the leading (i.e. highest Pt) Track in a given cone around the jet axis or a given direction
  const reco::TrackRef leadTk(std::string matchingConeMetric,double matchingConeSize,double ptTrackMin)const;
  const reco::TrackRef leadTk(const math::XYZVector& jetAxis,std::string matchingConeMetric,double matchingConeSize,double ptTrackMin)const;
  
  // return all Tracks in a cone of metric* "coneMetric" and size "coneSize" around a direction "coneAxis" 
  const reco::TrackRefVector tracksInCone(const math::XYZVector& coneAxis,const std::string coneMetric,const double coneSize,const double ptTrackMin)const;
  const reco::TrackRefVector tracksInCone(const math::XYZVector& coneAxis,const std::string coneMetric,const double coneSize,const double ptTrackMin,const double tracktorefpoint_maxDZ,const double refpoint_Z, const reco::Vertex &myPV)const;
  // return all Tracks in an annulus defined by inner(metric* "innerconeMetric" and size "innerconeSize") and outer(metric* "outerconeMetric" and size "outerconeSize") cones around a direction "coneAxis" 
  const reco::TrackRefVector tracksInAnnulus(const math::XYZVector& coneAxis,const std::string innerconeMetric,const double innerconeSize,const std::string outerconeMetric,const double outerconeSize,const double ptTrackMin)const;  
  const reco::TrackRefVector tracksInAnnulus(const math::XYZVector& coneAxis,const std::string innerconeMetric,const double innerconeSize,const std::string outerconeMetric,const double outerconeSize,const double ptTrackMin,const double tracktorefpoint_maxDZ,const double refpoint_Z, const reco::Vertex &myPV)const;  
  // return 1 if no/low Tracks activity in an isolation annulus around a leading Track, 0 otherwise; 
  // different possible metrics* for the matching, signal and isolation cones; 
  double discriminatorByIsolTracksN(unsigned int isolationAnnulus_Tracksmaxn)const;
  double discriminatorByIsolTracksN(const math::XYZVector& coneAxis, 
				    std::string matchingConeMetric,double matchingConeSize, double ptLeadingTrackMin, double ptOtherTracksMin, 
				    std::string signalConeMetric,double signalConeSize,std::string isolationConeMetric,double isolationConeSize, 
				    unsigned int isolationAnnulus_Tracksmaxn)const;
  // matching cone axis is the jet axis, signal and isolation cones axes are a leading Track axis;
  double discriminatorByIsolTracksN(std::string matchingConeMetric,double matchingConeSize, double ptLeadingTrackMin, double ptOtherTracksMin, 
				    std::string signalConeMetric,double signalConeSize,std::string isolationConeMetric,double isolationConeSize, 
				    unsigned int isolationAnnulus_Tracksmaxn)const;
 protected:
    TFormula ConeSizeTFormula;

  reco::BaseTau& BaseTau_;
  double AreaMetric_recoElements_maxabsEta_;
  reco::TrackRefVector Tracks_;  // track selection criteria applied
  reco::TrackRefVector IsolTracks_;  // tracks in an isolation annulus, track selection criteria applied; 
  // template objects for DR and Angle metrics
  DeltaR<math::XYZVector> metricDR_;
  Angle<math::XYZVector> metricAngle_;
  ElementsInConeRef<math::XYZVector,DeltaR<math::XYZVector>,reco::TrackCollection> TracksinCone_DRmetric_;
  ElementsInConeRef<math::XYZVector,Angle<math::XYZVector>,reco::TrackCollection> TracksinCone_Anglemetric_; 
  ElementsInAnnulusRef<math::XYZVector,DeltaR<math::XYZVector>,DeltaR<math::XYZVector>,reco::TrackCollection> TracksinAnnulus_innerDRouterDRmetrics_;
  ElementsInAnnulusRef<math::XYZVector,DeltaR<math::XYZVector>,Angle<math::XYZVector>,reco::TrackCollection> TracksinAnnulus_innerDRouterAnglemetrics_; 
  ElementsInAnnulusRef<math::XYZVector,Angle<math::XYZVector>,Angle<math::XYZVector>,reco::TrackCollection> TracksinAnnulus_innerAngleouterAnglemetrics_;
  ElementsInAnnulusRef<math::XYZVector,Angle<math::XYZVector>,DeltaR<math::XYZVector>,reco::TrackCollection> TracksinAnnulus_innerAngleouterDRmetrics_; 
};


#endif

// * different possible metrics for a cone : "DR", "angle", "area"; 
