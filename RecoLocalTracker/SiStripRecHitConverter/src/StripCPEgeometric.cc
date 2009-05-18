#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPEgeometric.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include <numeric>

StripClusterParameterEstimator::LocalValues StripCPEgeometric::
localParameters( const SiStripCluster & cluster, const LocalTrajectoryParameters & ltp) const{

  SiStripDetId::SubDetector subdet = SiStripDetId(cluster.geographicalId()).subDetector();
  StripCPE::Param const& p = param(DetId(cluster.geographicalId()));

  float localPitch = p.topology->localPitch(ltp.position());
  LocalVector track = ltp.momentum();
  track *= 
    (track.z()<0) ?  fabs(p.thickness/track.z()) : 
    (track.z()>0) ? -fabs(p.thickness/track.z()) :  
                         p.maxLength/track.mag() ;

  std::pair<float,float> position_sigma = position_sigma_inStrips( cluster.firstStrip(),
								   cluster.amplitudes().begin(),
								   cluster.amplitudes().end()-1,
								   std::max( 2*tandriftangle*p.thickness, fabs( (track+p.drift).x() ) ) / localPitch,
								   subdet);
  position_sigma.first -= 0.5* p.drift.x() / localPitch;
  return std::make_pair( p.topology->localPosition(position_sigma.first),
			 p.topology->localError(position_sigma.first, pow(position_sigma.second,2)) );
}

//treats strips as parallel
std::pair<float,float> StripCPEgeometric::
position_sigma_inStrips( uint16_t firstStrip, chargeIt_t first, chargeIt_t last, float projection, SiStripDetId::SubDetector type) const
{
  const unsigned N = 1+(last-first);
  const float middle = firstStrip + N/2.;

  if( N==1 ){
    return std::make_pair( middle,
			  invsqrt12*( 1 - projection/(1+exp(10*(projection-1.1))) )   );
  }
  else if( projection < N-2 ) {
    return (  hasMultiPeak(first,last) 
	      ? std::make_pair( middle, invsqrt12*N )
	      : ( ( *first<*last ) 
		  ? position_sigma_inStrips( firstStrip+1, first+1, last,   projection, type)
		  : position_sigma_inStrips( firstStrip,   first,   last-1, projection, type) ));
  }
  else if( projection < N-1  &&  useNMinusOne( first, last, projection, type) ) {
    return ( ( *first<*last )
	     ? position_sigma_inStrips( firstStrip+1, first+1, last,   projection, type)
	     : position_sigma_inStrips( firstStrip,   first,   last-1, projection, type) );
  }
  else {
    float sumQ(0), sumXQ(0); for(chargeIt_t i=first; i<last+1; i++) { sumQ += *i; sumXQ += (*i)*(i-first);}

    const float centroid = sumXQ/sumQ  - 0.5*(N-1);
    const float eta = (*last-*first)/sumQ;
    const float crossover = exp( crossoverRate *(projection - N/(1+fabs(eta))));

    const float offset = ( 0.5*eta*projection   /(1+crossover)
			   + centroid           /(1+1/crossover) );

    const float sigma = ( (0.007 - 0.01*eta + 0.05*projection) / (1+crossover)
			  + invsqrt12                          / (1+1/crossover) );

    return std::make_pair( middle + offset,  sigma);
  }
  throw cms::Exception("Illegal state");
  return std::make_pair(0,0);
}

bool StripCPEgeometric::
useNMinusOne(chargeIt_t first, chargeIt_t last, float projection, SiStripDetId::SubDetector type) const 
{
  unsigned N = (last-first)+1;
  switch(N) {
  case 2: return (fabs(*last-*first)/float(*last+*first) > (type==SiStripDetId::TIB? TIBeta: TOBeta));
  case 3: 
    {
      chargeIt_t middle = first + 1;
      return ( (*last<*first)
	       ? ((*middle-*last)/(*middle+*last)   > (type==SiStripDetId::TIB? TIBeta: TOBeta))
	       : ((*middle-*first)/(*middle+*first) > (type==SiStripDetId::TIB? TIBeta: TOBeta)));
    }
  default:
    {
      float dEdX_N_int = accumulate(first+1,last,0)/(N-2);
      float dEdX_N_ext = (*first+*last)/(projection-(N-2));
      float dEdX_Minus1_int = (*first<*last) 	
	? accumulate(first+2,last,float(0))/(N-3) 
	: accumulate(first+1,last-1,float(0))/(N-3);
      float dEdX_Minus1_ext = (*first<*last)   
	? (*(first+1)+*last)/(projection-(N-3))    
	:  (*first+*(last-1))/(projection-(N-3));
      
      return 
	fabs( dEdX_Minus1_ext/dEdX_Minus1_int - 1) 
	< fabs(    dEdX_N_ext/dEdX_N_int      - 1);
    }
  }
  return false;
}

bool StripCPEgeometric::
hasMultiPeak(chargeIt_t first, chargeIt_t last) const
{//only called for N>=3
  unsigned N = 1+(last-first);
  chargeIt_t max1 = std::max_element(first,first+N/2);
  chargeIt_t max2 = std::max_element(first+N/2,last+1);
  float Qbetween = accumulate(max1+1,max2,float(0));
  if(Qbetween>0) {
    float ratio = (*max1<*max2)? *max1/(*max2) : *max2/(*max1);
    if( ratio>0.5 && Qbetween/(max2-max1-1) < 0.5*(*max1+*max2)/2. )
      return true;
  }
  return false;
}
