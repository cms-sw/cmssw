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

public:  
  StripClusterParameterEstimator::LocalValues
  localParameters( const SiStripCluster&, const GeomDetUnit&, const LocalTrajectoryParameters&) const;
  
  float stripErrorSquared(const unsigned N, const float uProj, const SiStripDetId::SubDetector loc ) const ;
  float legacyStripErrorSquared(const unsigned N, const float uProj) const;

  StripCPEfromTrackAngle( edm::ParameterSet & conf, 
			  const MagneticField& mag, 
			  const TrackerGeometry& geom, 
			  const SiStripLorentzAngle& lorentz,
                          const SiStripBackPlaneCorrection& backPlaneCorrection,
			  const SiStripConfObject& confObj,
			  const SiStripLatency& latency) 
  : StripCPE(conf, mag, geom, lorentz, backPlaneCorrection, confObj, latency )
  , useLegacyError(conf.existsAs<bool>("useLegacyError") ? conf.getParameter<bool>("useLegacyError") : true)
  , maxChgOneMIP(conf.existsAs<float>("maxChgOneMIP") ? conf.getParameter<double>("maxChgOneMIP") : -6000.)
  {
    mLC_P[0] = conf.existsAs<double>("mLC_P0") ? conf.getParameter<double>("mLC_P0") : -.326;
    mLC_P[1] = conf.existsAs<double>("mLC_P1") ? conf.getParameter<double>("mLC_P1") :  .618;
    mLC_P[2] = conf.existsAs<double>("mLC_P2") ? conf.getParameter<double>("mLC_P2") :  .300;

    mHC_P[SiStripDetId::TIB - 3][0] = conf.existsAs<double>("mTIB_P0") ? conf.getParameter<double>("mTIB_P0") : -.742  ;
    mHC_P[SiStripDetId::TIB - 3][1] = conf.existsAs<double>("mTIB_P1") ? conf.getParameter<double>("mTIB_P1") :  .202  ;
    mHC_P[SiStripDetId::TID - 3][0] = conf.existsAs<double>("mTID_P0") ? conf.getParameter<double>("mTID_P0") : -1.026 ;
    mHC_P[SiStripDetId::TID - 3][1] = conf.existsAs<double>("mTID_P1") ? conf.getParameter<double>("mTID_P1") :  .253  ;
    mHC_P[SiStripDetId::TOB - 3][0] = conf.existsAs<double>("mTOB_P0") ? conf.getParameter<double>("mTOB_P0") : -1.427 ;
    mHC_P[SiStripDetId::TOB - 3][1] = conf.existsAs<double>("mTOB_P1") ? conf.getParameter<double>("mTOB_P1") :  .433  ;
    mHC_P[SiStripDetId::TEC - 3][0] = conf.existsAs<double>("mTEC_P0") ? conf.getParameter<double>("mTEC_P0") : -1.885 ;
    mHC_P[SiStripDetId::TEC - 3][1] = conf.existsAs<double>("mTEC_P1") ? conf.getParameter<double>("mTEC_P1") :  .471  ;  
  }
};
#endif
