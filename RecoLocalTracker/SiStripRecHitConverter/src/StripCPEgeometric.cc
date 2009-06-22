#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPEgeometric.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/CrosstalkInversion.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include <numeric>

StripCPEgeometric::StripCPEgeometric( edm::ParameterSet& conf, 
				      const MagneticField* mag, 
				      const TrackerGeometry* geom, 
				      const SiStripLorentzAngle* LorentzAngle)
  : StripCPE(conf, mag, geom, LorentzAngle ),
    invsqrt12(1/sqrt(12)),
    tandriftangle(conf.getParameter<double>("TanDriftAngle")),    
    thickness_RelErr2(pow(conf.getParameter<double>("ThicknessRelativeUncertainty"), 2)),
    noise_threshold(conf.getParameter<double>("NoiseThreshold")),
    crossoverRate(15)
{
  std::string mode = conf.getParameter<bool>("APVpeakmode") ? "Peak" : "Dec";
  crosstalk.resize(7,0);
  crosstalk[SiStripDetId::TIB] = conf.getParameter<double>("CouplingConstant"+mode+"TIB");
  crosstalk[SiStripDetId::TID] = conf.getParameter<double>("CouplingConstant"+mode+"TID");
  crosstalk[SiStripDetId::TOB] = conf.getParameter<double>("CouplingConstant"+mode+"TOB");
  crosstalk[SiStripDetId::TEC] = conf.getParameter<double>("CouplingConstant"+mode+"TEC");
}

StripClusterParameterEstimator::LocalValues StripCPEgeometric::
localParameters( const SiStripCluster& cluster, const GeomDetUnit& det, const LocalTrajectoryParameters& ltp) const {
  return localParameters(cluster,ltp);
}

StripClusterParameterEstimator::LocalValues StripCPEgeometric::
localParameters( const SiStripCluster& cluster, const LocalTrajectoryParameters& ltp) const {
  StripCPE::Param const& p = param(DetId(cluster.geographicalId()));

  LocalVector track = ltp.momentum();
  track *=   (track.z()<0) ?  fabs(p.thickness/track.z()) : 
             (track.z()>0) ? -fabs(p.thickness/track.z()) :  
                              p.maxLength/track.mag() ;

  const float projection = fabs( p.coveredStrips( track+p.drift, ltp.position() ));
  const float minProjection = 2*p.thickness*tandriftangle/p.topology->localPitch(ltp.position());
  const float projection_RelErr2 = thickness_RelErr2 + p.pitch_RelErr2;

  const std::pair<float,float> s_se2 = strip_stripErrorSquared( cluster, std::max(projection,minProjection), projection_RelErr2 );
  const float strip = p.driftCorrected( s_se2.first, ltp.position() );

  return std::make_pair( p.topology->localPosition( strip ),
			 p.topology->localError( strip, s_se2.second ) );
}


std::pair<float,float> StripCPEgeometric::
strip_stripErrorSquared( const SiStripCluster& cluster, const float& projection, const float& projection_RelErr2) const {
  WrappedCluster wc(cluster, crosstalk);
  if( isMultiPeaked( cluster, projection ) ) return std::make_pair( wc.middle(), wc.N*wc.N/12. ) ;
  while( useNMinusOne( wc, projection) )  wc.dropSmallerEdgeStrip();
  if( wc.deformed() ) return std::make_pair( wc.middle()+wc.centroid(), 1/12.);

  const float eta = wc.eta();
  float sigma;
  switch( wc.N ) {
  case 1: sigma = invsqrt12*( 1-0.82*projection );                                                                break;
  default: sigma = 0.5*projection* sqrt(eta*eta*projection_RelErr2+ wc.etaErr2() );                               break;
  }
  const float crossoverPoint = projection - wc.maxProjection();
  const float offset = mix(   0.5*projection*eta,   wc.centroid(),   crossoverPoint);
  const float sigma2 = mix(          sigma*sigma,           1/12.,   crossoverPoint-0.22);                     

  return std::make_pair( wc.middle() + offset,  sigma2 );
}

inline
bool StripCPEgeometric::
isMultiPeaked(const SiStripCluster& cluster, const float& projection) const {
  uint16_t N = cluster.amplitudes().size();
  if(projection > N-2) return false;
  if(projection < N-4) return true;

  return false;
}

inline
bool StripCPEgeometric::
useNMinusOne(const WrappedCluster& wc, const float& projection) const {
  WrappedCluster wcTest(wc); wcTest.dropSmallerEdgeStrip();
  if( wc.N == 1 ) return false;
  if( wc.smallerEdgeCharge() < 0) return true;
  if( projection < wc.N-2) return true;
  if( projection >= wcTest.maxProjection() ) return false;
  if( wc.eta() > 1./(wc.N-1) ) return true;
  if( wc.N==2 || wc.N==3)  
    return wc.smallerEdgeCharge() < noise_threshold;
  return fabs(  wcTest.dedxRatio(projection)-1 )   <   fabs(  wc.dedxRatio(projection)-1 ); 
}

inline
float StripCPEgeometric::
mix(const float& left, const float& right, const float& crossoverPoint ) const {
  const float e = exp(crossoverRate*crossoverPoint);
  return left/(1+e) + right/(1+1/e);
}

inline
StripCPEgeometric::WrappedCluster::
WrappedCluster(const SiStripCluster& cluster, const std::vector<float>& xtalk) 
  : N(cluster.amplitudes().size()),
    type(SiStripDetId(cluster.geographicalId()).subDetector()),
    firstStrip(cluster.firstStrip())
{ 
  Q = InverseCrosstalkMatrix::unfold( cluster.amplitudes(), xtalk[type]);
  first = Q.begin();
  last = Q.end()-1;
  sumQ = accumulate(first, last+1, float(0));
}

inline
float StripCPEgeometric::WrappedCluster::
eta() const 
{ return (*last-*first) / sumQ; }

inline
float StripCPEgeometric::WrappedCluster::
etaErr2() const 
{ return ( pow( *last-*first, 2) / sumQ + *first + *last ) / (sumQ*sumQ);}

inline
float StripCPEgeometric::WrappedCluster::
middle() const 
{ return firstStrip + N/2.;}

inline
float StripCPEgeometric::WrappedCluster::
maxProjection() const
{ return N/(1+fabs(eta())); }

inline
float StripCPEgeometric::WrappedCluster::
dedxRatio(const float& projection) const 
{ return ( sumQ/(*first+*last) - 1 ) * ( projection/(N-2) - 1 ); }

inline
float StripCPEgeometric::WrappedCluster::
smallerEdgeCharge() const 
{ return (*first<*last)?  *first  : *last; }

float StripCPEgeometric::WrappedCluster::
centroid() const { 
  float sumXQ(0);
  for(std::vector<float>::const_iterator i = first; i<last+1; i++) sumXQ += (i-first)*(*i);
  return sumXQ/sumQ - (N-1)/2.;
}

inline
void StripCPEgeometric::WrappedCluster::
dropSmallerEdgeStrip() {
  if(*first == *last)   { sumQ-= *first; first++; firstStrip++; 
                          sumQ-=  *last;  last--;                N-=2; } 
  else if(*first<*last) { sumQ-= *first; first++; firstStrip++;  N-=1; }
  else                  { sumQ-=  *last;  last--;                N-=1; }
}

inline
bool StripCPEgeometric::WrappedCluster::
deformed() const {  
  return   N>2  &&  std::max(*first,*last) > (sumQ-*first-*last)/(N-2);
}
