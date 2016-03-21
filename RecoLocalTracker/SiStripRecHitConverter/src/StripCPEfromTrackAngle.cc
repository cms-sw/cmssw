#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPEfromTrackAngle.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"                                                           
#include "DataFormats/SiStripCluster/interface/SiStripClusterTools.h"

#include "vdt/vdtMath.h"

StripCPEfromTrackAngle::StripCPEfromTrackAngle( edm::ParameterSet & conf,
                          const MagneticField& mag,
                          const TrackerGeometry& geom,
                          const SiStripLorentzAngle& lorentz,
                          const SiStripBackPlaneCorrection& backPlaneCorrection,
                          const SiStripConfObject& confObj,
                          const SiStripLatency& latency)
  : StripCPE(conf, mag, geom, lorentz, backPlaneCorrection, confObj, latency )
  , useLegacyError(conf.existsAs<bool>("useLegacyError") ? conf.getParameter<bool>("useLegacyError") : true)
  , maxChgOneMIP(conf.existsAs<float>("maxChgOneMIP") ? conf.getParameter<double>("maxChgOneMIP") : -6000.),
    m_algo(useLegacyError ? Algo::legacy : ( maxChgOneMIP<0 ? Algo::mergeCK : Algo::chargeCK))
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

float StripCPEfromTrackAngle::stripErrorSquared(const unsigned N, const float uProj, const SiStripDetId::SubDetector loc ) const {
  auto fun = [&] (float x)  -> float { return mLC_P[0]*x*vdt::fast_expf(-x*mLC_P[1])+mLC_P[2];};
  auto uerr = (N <= 4) ?  fun(uProj) :  mHC_P[loc-3][0]+float(N)*mHC_P[loc-3][1];
  return uerr*uerr;
}

float StripCPEfromTrackAngle::legacyStripErrorSquared(const unsigned N, const float uProj) const {
  if unlikely( (float(N)-uProj) > 3.5f )
    return float(N*N)/12.f;
  else {
    static constexpr float P1=-0.339;
    static constexpr float P2=0.90;
    static constexpr float P3=0.279;
    const float uerr = P1*uProj*vdt::fast_expf(-uProj*P2)+P3;
    return uerr*uerr;
  }
}

StripClusterParameterEstimator::LocalValues 
StripCPEfromTrackAngle::localParameters( const SiStripCluster& cluster, const GeomDetUnit& det, const LocalTrajectoryParameters& ltp) const {
  
  StripCPE::Param const & p = param(det);
  SiStripDetId ssdid = SiStripDetId( det.geographicalId() );  
 
  LocalVector track = ltp.momentum();
  track *= -p.thickness/track.z();

  const unsigned N = cluster.amplitudes().size();
  const float fullProjection = p.coveredStrips( track+p.drift, ltp.position());
  float uerr2=0;

  switch (m_algo) {
    case Algo::chargeCK :
       {
       auto dQdx = siStripClusterTools::chargePerCM(cluster, ltp, p.invThickness);
       uerr2 = dQdx > maxChgOneMIP ? legacyStripErrorSquared(N,std::abs(fullProjection)) : stripErrorSquared( N, std::abs(fullProjection),ssdid.subDetector() );
       }
       break;
    case Algo::legacy :
       uerr2 = legacyStripErrorSquared(N,std::abs(fullProjection));
       break;
    case Algo::mergeCK :
      uerr2 = cluster.isMerged() ? legacyStripErrorSquared(N,std::abs(fullProjection)) : stripErrorSquared( N, std::abs(fullProjection),ssdid.subDetector() );
      break;
  }

  const float strip = cluster.barycenter() -  0.5f*(1.f-p.backplanecorrection) * fullProjection
    + 0.5f*p.coveredStrips(track, ltp.position());

  return std::make_pair( p.topology->localPosition(strip, ltp.vector()),
			 p.topology->localError(strip, uerr2, ltp.vector()) );
}

