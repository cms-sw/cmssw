#ifndef RecoLocalTracker_SiStripRecHitConverter_StripCPEfromTrackAngle_H
#define RecoLocalTracker_SiStripRecHitConverter_StripCPEfromTrackAngle_H

#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPE.h"

class StripCPEfromTrackAngle : public StripCPE 
{

private:
  using StripCPE::localParameters;
  
  //Error parameterization, low cluster width function
  float mLC_P[3];
  float mHC_P[4][2];

  //High cluster width is broken down by sub-det
  std::map<SiStripDetId::SubDetector, float> mHC_P0;
  std::map<SiStripDetId::SubDetector, float> mHC_P1;

  //Set to true if we are using the old error parameterization
  const bool useLegacyError;

  //Clusters with charge/path > this cut will use old error parameterization
  // (overridden by useLegacyError; negative value disables the cut)
  const float maxChgOneMIP;

  enum class Algo { legacy, mergeCK, chargeCK };

  Algo m_algo;

public:  
  using AlgoParam = StripCPE::AlgoParam;
  using AClusters = StripClusterParameterEstimator::AClusters;
  using ALocalValues  = StripClusterParameterEstimator::ALocalValues;
  
  void localParameters(AClusters const & clusters, ALocalValues & retValues, const GeomDetUnit& gd, const LocalTrajectoryParameters &ltp) const override;

  StripClusterParameterEstimator::LocalValues
  localParameters( const SiStripCluster& cl, AlgoParam const & ap) const override;

  
  StripClusterParameterEstimator::LocalValues
  localParameters( const SiStripCluster&, const GeomDetUnit&, const LocalTrajectoryParameters&) const override;
  
  float stripErrorSquared(const unsigned N, const float uProj, const SiStripDetId::SubDetector loc ) const ;
  float legacyStripErrorSquared(const unsigned N, const float uProj) const;


  StripCPEfromTrackAngle( edm::ParameterSet & conf,
                          const MagneticField& mag,
                          const TrackerGeometry& geom,
                          const SiStripLorentzAngle& lorentz,
                          const SiStripBackPlaneCorrection& backPlaneCorrection,
                          const SiStripConfObject& confObj,
                          const SiStripLatency& latency);

};
#endif
