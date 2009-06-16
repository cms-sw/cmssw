#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPEgeometric.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include <numeric>

StripCPEgeometric::StripCPEgeometric( edm::ParameterSet& conf, 
				      const MagneticField* mag, 
				      const TrackerGeometry* geom, 
				      const SiStripLorentzAngle* LorentzAngle)
  : StripCPE(conf, mag, geom, LorentzAngle ),
    invsqrt12(1/sqrt(12)),
    crosstalksigma(conf.getParameter<double>("CouplingConstantSpread")),
    tandriftangle(conf.getParameter<double>("TanDriftAngle")),    
    crossoverRate(15)
{
  std::string mode = conf.getParameter<bool>("APVpeakmode") ? "Peak" : "Dec";
  crosstalk.resize(7,0);
  crosstalk[SiStripDetId::TIB] = conf.getParameter<double>("CouplingConstant"+mode+"TIB");
  crosstalk[SiStripDetId::TID] = conf.getParameter<double>("CouplingConstant"+mode+"TID");
  crosstalk[SiStripDetId::TOB] = conf.getParameter<double>("CouplingConstant"+mode+"TOB");
  crosstalk[SiStripDetId::TEC] = conf.getParameter<double>("CouplingConstant"+mode+"TEC");
  transform(crosstalk.begin(), crosstalk.end(), back_inserter(edgeRatioCut), edgeRatioFromCrosstalk(crosstalksigma));
  edgeRatioCut[SiStripDetId::TID] *= 1.1;
  edgeRatioCut[SiStripDetId::TEC] *= 1.1;
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
  const float projection = std::max( 2*p.thickness*tandriftangle/p.topology->localPitch(ltp.position()),
				     fabs( p.coveredStrips( track+p.drift, ltp.position() )) );

  const std::pair<float,float> s_se2 = strip_stripErrorSquared( cluster, projection);
  const float strip = p.driftCorrected( s_se2.first, ltp.position() );

  return std::make_pair( p.topology->localPosition( strip ),
			 p.topology->localError( strip, s_se2.second ) );
}


std::pair<float,float> StripCPEgeometric::
strip_stripErrorSquared( const SiStripCluster& cluster, const float& projection) const {
  WrappedCluster wc(cluster);
  if( isMultiPeaked( cluster, projection ) ) return std::make_pair( wc.middle(), wc.N*wc.N/12. ) ;
  while( useNMinusOne( wc, projection) )  wc.dropSmallerEdgeStrip();
  if( wc.deformed() ) return std::make_pair( wc.centroid(), 1/12.);

  const float eta = wc.eta(crosstalk);
  float sigma;
  switch( wc.N ) {
  case 1: sigma = invsqrt12*( 1-0.82*projection );                                                                break;
  case 2: sigma =  0.013 + 0.060*projection*projection*(1-eta*eta)*sqrt(0.5*(fabs(projection-2)-(projection-2))); break;
  case 3: sigma = -0.040 + 0.060*projection + invsqrt12/(1+exp(10*(projection-2*fabs(eta)-0.8)));                 break;
  case 4: sigma = -0.078 + 0.060*projection + invsqrt12/(1+exp(15*(projection-2.1)));                             break;
  case 5: sigma = -0.190 + 0.075*projection + invsqrt12/(1+exp(15*(projection-3.2)));                             break;
  case 6: sigma = -0.350 + 0.095*projection + invsqrt12/(1+exp(15*(projection-4.3)));                             break;
  default: sigma = 0.250 + 0.100*(projection-wc.N);                                                               break;
  }
  const float crossoverPoint = projection - wc.maxProjection(crosstalk);
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
  if( projection < wc.N-2) return true;
  if( projection >= wcTest.maxProjection(crosstalk) ) return false;
  if( wc.eta(crosstalk) > 1./(wc.N-1) ) return true;
  if( wc.N==2 || wc.N==3)  
    return wc.smallEdgeRatio() < edgeRatioCut[wc.type];
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
WrappedCluster(const SiStripCluster& cluster) 
  : N(cluster.amplitudes().size()),
    type(SiStripDetId(cluster.geographicalId()).subDetector()),
    first(cluster.amplitudes().begin()),
    last(cluster.amplitudes().end()-1),
    firstStrip(cluster.firstStrip()),
    sumQ(0)
{ for(std::vector<uint8_t>::const_iterator i = first; i<first+N; i++)  sumQ+=(*i);}

inline
float StripCPEgeometric::WrappedCluster::
eta(const std::vector<float>& xtalk) const { 
  const float x = xtalk[type];
  switch(N) {   /*   (Q_r-Q_l)/sumQ   */
  case  1: return 0;
  case  2: return (1-x)/(1-3*x) * (*last-*first)/sumQ; 
  case  3: return (1-2*x-2*x*x/(1-2*x)) * (*last-*first) / ((1-3*x)*sumQ - x* *(first+1));
  default: return ((1-2*x)*(*last-*first)-x*(*(last-1)-*(first+1)+x*x*(*(last-2)-*(first+2)))) / 
	     ((pow(1-2*x,2)-x*x)*(sumQ-x/(1-2*x) * (*last+*first)));
  }
}

inline
float StripCPEgeometric::WrappedCluster::
middle() const 
{ return firstStrip + N/2.;}

inline
float StripCPEgeometric::WrappedCluster::
maxProjection(const std::vector<float>& xtalk) const
{ return N/(1+fabs(eta(xtalk))); }

inline
float StripCPEgeometric::WrappedCluster::
dedxRatio(const float& projection) const 
{ return ( sumQ/(*first+*last) - 1 ) * ( projection/(N-2) - 1 ); }

inline
float StripCPEgeometric::WrappedCluster::
smallEdgeRatio() const 
{ return (*first<*last)? ( *first / float(*(first+1)) ) : (*last / float(*(last-1))); }

float StripCPEgeometric::WrappedCluster::
centroid() const { 
  float sumXQ(0);
  for(std::vector<uint8_t>::const_iterator i = first; i<last+1; i++) sumXQ += (i-first)*(*i);
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
