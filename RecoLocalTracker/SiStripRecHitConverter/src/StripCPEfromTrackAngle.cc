#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPEfromTrackAngle.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"                                                           
#include "DataFormats/SiStripCluster/interface/SiStripClusterTools.h"

#include "vdt/vdtMath.h"

float StripCPEfromTrackAngle::stripErrorSquared(const unsigned N, const float uProj, const SiStripDetId::SubDetector loc ) const {
  if( loc == SiStripDetId::UNKNOWN)
    throw cms::Exception("StripCPEfromTrackAngle::stripErrorSquared", "Incompatible sub-detector.");

  auto fun = [&] (float x)  -> float { return mLC_P[0]*x*vdt::fast_expf(-x*mLC_P[1])+mLC_P[2];};
  auto uerr = (N <= 4) ?  fun(uProj) :  mHC_P[loc-3][0]+float(N)*mHC_P[loc-3][1];
  return uerr*uerr;
}

float StripCPEfromTrackAngle::legacyStripErrorSquared(const unsigned N, const float uProj) const {
  if( (float(N)-uProj) > 3.5f )
    return float(N*N)/12.f;
  else {
    static constexpr float P1=-0.339;
    static constexpr float P2=0.90;
    static constexpr float P3=0.279;
    const float uerr = P1*uProj*vdt::fast_expf(-uProj*P2)+P3;
    return uerr*uerr;
  }
}

StripClusterParameterEstimator::LocalValues StripCPEfromTrackAngle::
localParameters( const SiStripCluster& cluster, const GeomDetUnit& det, const LocalTrajectoryParameters& ltp) const {
  
  StripCPE::Param const & p = param(det);
  SiStripDetId ssdid = SiStripDetId( det.geographicalId() );  
 
  LocalVector track = ltp.momentum();
  track *= 
    (track.z()<0) ?  std::abs(p.thickness/track.z()) : 
    (track.z()>0) ? -std::abs(p.thickness/track.z()) :  
                         p.maxLength/track.mag() ;

  const unsigned N = cluster.amplitudes().size();
  const float fullProjection = p.coveredStrips( track+p.drift, ltp.position());
  float uerr2;
  if (useLegacyError) {
    uerr2 = legacyStripErrorSquared(N,std::abs(fullProjection));
  } else if (maxChgOneMIP < 0.0) {
    uerr2 = cluster.isMerged() ? legacyStripErrorSquared(N,std::abs(fullProjection)) : stripErrorSquared( N, std::abs(fullProjection),ssdid.subDetector() );
  } else {
    float dQdx = siStripClusterTools::chargePerCM(ssdid, cluster, ltp);
    uerr2 = dQdx > maxChgOneMIP ? legacyStripErrorSquared(N,std::abs(fullProjection)) : stripErrorSquared( N, std::abs(fullProjection),ssdid.subDetector() );
  }
  const float strip = cluster.barycenter() -  0.5f*(1.f-p.backplanecorrection) * fullProjection
    + 0.5f*p.coveredStrips(track, ltp.position());

  return std::make_pair( p.topology->localPosition(strip, ltp.vector()),
			 p.topology->localError(strip, uerr2, ltp.vector()) );
}

