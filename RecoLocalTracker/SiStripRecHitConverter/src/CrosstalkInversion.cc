#include "RecoLocalTracker/SiStripRecHitConverter/interface/CrosstalkInversion.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "FWCore/Utilities/interface/isFinite.h"

namespace reco {

std::vector<stats_t<float> > InverseCrosstalkMatrix::
unfold(const SiStripCluster& clus, const float x) {
  auto const & q = clus.amplitudes();
  const stats_t<float> suppressed(-5,100);
  const stats_t<float> saturated(254,400);
  #define STATS(value) ( (value<254) ? stats_t<float>(value) : saturated )

  const unsigned N=q.size();
  std::vector<stats_t<float> > Q(N+2,stats_t<float>(0));
  Q[0] = Q[N+1] = suppressed;

  if(N==1)                                 //optimize N==1
    Q[1] = STATS(q[0])/(1-2*x);  
  else if(N==2) {                          //optimize N==2
    const double A=1-2*x; 
    const stats_t<float> q0 = STATS(q[0]);
    const stats_t<float> q1 = STATS(q[1]);
    Q[1] = ( A*q0 -x*q1 ) / (A*A-x*x);
    Q[2] = ( A*q1 -x*q0 ) / (A*A-x*x);
  } 
  else {                                   //general case
    const InverseCrosstalkMatrix inverse(N,x);  
    for(unsigned i=0; i<(N+1)/2; i++) {
      for(unsigned j=i; j<N-i; j++) {
	const float Cij = inverse(i+1,j+1);
	Q[i+1] += Cij * STATS(q[  j  ]) ;  if( i!=j)   
	Q[j+1] += Cij * STATS(q[  i  ]) ;  if( N!=i+j+1) {
	Q[N-i] += Cij * STATS(q[N-j-1]) ;  if( i!=j)
	Q[N-j] += Cij * STATS(q[N-i-1]) ;
	}
      }
    }
  }
  #undef STATS
  return Q;
}

InverseCrosstalkMatrix::
InverseCrosstalkMatrix(const unsigned N, const float x)
  : N( x>0 ? N : 0 ),
    sq( sqrt(-x*4+1)),
    lambdaP( 1+(1+sq)/(-x*2) ),
    lambdaM( 1+(1-sq)/(-x*2) ),
    denominator( sq * ( pow(lambdaP,N+1) - pow(lambdaM,N+1) ) )
{}

float InverseCrosstalkMatrix::
operator()(const unsigned i, const unsigned j) const
{ return N==0 || edm::isNotFinite(denominator) ? i==j : i>=j ? element(i,j) : element(j,i) ; }

inline
float InverseCrosstalkMatrix::
element(const unsigned i, const unsigned j) const 
{ return ( pow(lambdaM,N+1-i) - pow(lambdaP,N+1-i) ) * ( pow(lambdaM,j) - pow(lambdaP,j) ) / denominator; }

}
