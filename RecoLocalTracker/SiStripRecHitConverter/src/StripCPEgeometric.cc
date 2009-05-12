#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPEgeometric.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <numeric>

StripClusterParameterEstimator::LocalValues StripCPEgeometric::localParameters( const SiStripCluster & cl,const LocalTrajectoryParameters & ltp) const{

  StripCPE::Param const & p = param(DetId(cl.geographicalId()));
  
  LocalVector track = ltp.momentum();
  track *= 
    (track.z()<0) ?  fabs(p.thickness/track.z()) : 
    (track.z()>0) ? -fabs(p.thickness/track.z()) :  
                         p.maxLength/track.mag() ;
  float projectedWidthInPitches = std::max(minProj, fabs((track+p.drift).x()/p.topology->localPitch(ltp.position()))); //neglects edge conditions


  std::pair<float,float> position_sigma = position_sigma_inStrips( cl.firstStrip(),
								   cl.amplitudes().begin(),
								   cl.amplitudes().end()-1,
								   projectedWidthInPitches);

  LocalPoint result = p.topology->localPosition( position_sigma.first) - 0.5*p.drift;
  LocalError eresult= p.topology->localError( p.topology->measurementPosition(result),
					      MeasurementError( pow(position_sigma.second,2), 
								0., 
								1./12.)    );
  return std::make_pair(result,eresult);
}

//treats strips as parallel
std::pair<float,float> StripCPEgeometric::
position_sigma_inStrips( uint16_t firstStrip, chargeIt_t first, chargeIt_t last, float projection) const
{
  const unsigned N = 1+(last-first);
  const float middle = firstStrip + N/2.;

  if( N==1 ){
    return std::make_pair( middle,
			  invsqrt12*( 1 - projection )+0.01   );
  }
  else if( projection < N-2 ) {
    return (  hasMultiPeak(first,last) 
	      ? std::make_pair( middle, invsqrt12*N )
	      : ( ( *first<*last ) 
		  ? position_sigma_inStrips( firstStrip+1, first+1, last,   projection)
		  : position_sigma_inStrips( firstStrip,   first,   last-1, projection) ));
  }
  else if( projection < N-1  &&  useNMinusOne( first, last, projection) ) {
    return ( ( *first<*last )
	     ? position_sigma_inStrips( firstStrip+1, first+1, last,   projection)
	     : position_sigma_inStrips( firstStrip,   first,   last-1, projection) );
  }
  else {
    float sumQ(0), sumXQ(0); for(chargeIt_t i=first; i<last+1; i++) { sumQ += *i; sumXQ += (*i)*(i-first);}

    const float centroid = sumXQ/sumQ  - 0.5*(N-1);
    const float eta = (*last-*first)/sumQ;
    const float crossover = exp( crossoverRate *(projection - 1/(1-fabs(eta))));

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
useNMinusOne(chargeIt_t first, chargeIt_t last, float projection) const 
{
  unsigned N = (last-first)+1;
  switch(N) {
  case 2: return (fabs(*last-*first)/float(*last+*first) > 0.8);
  case 3:
    {
      float dEdX_int = *(first+1);
      float dEdX_ext = (*first+*last)/(projection-1);
      return fabs(dEdX_ext/dEdX_int -1) > width3threshold;
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
{
  return false;
}
