#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPEgeometric.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include <numeric>

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
  const float projection = std::max( 2*p.thickness*tandriftangle[p.subdet],
				     fabs( p.coveredStrips( track+p.drift, ltp.position() )) );

  const std::pair<float,float> s_se2 = strip_stripErrorSquared( cluster, projection);
  const float strip = p.driftCorrected( s_se2.first, ltp.position() );

  return std::make_pair( p.topology->localPosition( strip ),
			 p.topology->localError( strip, s_se2.second ) );
}


std::pair<float,float> StripCPEgeometric::
strip_stripErrorSquared( const SiStripCluster& cluster, const float& projection) const {
  WrappedCluster wc(cluster);
  if( isMultiPeaked( cluster, projection ) )
    return std::make_pair( wc.middle(), wc.N*invsqrt12 ) ;
  
  while( useNMinusOne( wc, projection) ) 
    wc.dropSmallerEdgeStrip();

  float sigma;
  switch( wc.N ) {
    /*  case 1: sigma = invsqrt12*( 1-projection );                break;
	case 2: sigma = (0.007 - 0.01*wc.eta() + 0.05*projection); break;
	case 3: sigma = invsqrt12;                                 break;
	case 4: sigma = invsqrt12;                                 break;
	case 5: sigma = invsqrt12;                                 break;
	case 6: sigma = invsqrt12;                                 break; */
  default: sigma = invsqrt12;                                 break;
  }
  const float crossoverPoint = projection - wc.N/(1+fabs(wc.eta()));
  const float offset = mix(   0.5*wc.eta()*projection,   wc.centroid(),   crossoverPoint);
  const float sigma2 = mix(               sigma*sigma,           1/12.,   crossoverPoint);                     

  return std::make_pair( wc.middle() + offset,  sigma2 );
}

inline
bool StripCPEgeometric::
isMultiPeaked(const SiStripCluster& cluster, const float& projection) const {
  uint16_t N = cluster.amplitudes().size();
  if(projection > N-2) return false;

  std::vector<uint8_t>::const_iterator first,maxL,maxR;
  first = cluster.amplitudes().begin();
  maxL = std::max_element(first,first+N/2);
  maxR = std::max_element(first+N/2,first+N);

  const float Qbetween = accumulate(maxL+1,maxR,float(0));
  if(Qbetween>0) {
    float ratio = (*maxL<*maxR)? *maxL/(*maxR) : *maxR/(*maxL);
    if( ratio>0.5 && Qbetween/((maxR-maxL)-1) < 0.5*(*maxL+*maxR)/2. )
      return true;
  }
  return false;
}

inline
bool StripCPEgeometric::
useNMinusOne(const WrappedCluster& wc, const float& projection) const {
  if( projection < wc.N-2) return true;
  if( wc.N-1 < projection) return false;
  if( wc.N==2 || wc.N==3)  return wc.smallEdgeRatio() < edgeRatioCut[wc.type];

  WrappedCluster wcTest(wc);  
  wcTest.dropSmallerEdgeStrip();
  return   fabs(  wcTest.dedxRatio(projection)-1 )   <   fabs(  wc.dedxRatio(projection)-1 ); 
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
eta() const 
{ return (*last-*first)/sumQ; }

inline
float StripCPEgeometric::WrappedCluster::
middle() const 
{ return firstStrip + N/2.;}

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
  if(*first<*last) {
    firstStrip++;
    sumQ-= *first;
    first++;
  }  else {
    sumQ-= *last;
    last--;
  }
  N--;
  return;
}
