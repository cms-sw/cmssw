#include "RecoLocalTracker/SiStripRecHitConverter/interface/CrosstalkInversion.h"

namespace reco {

std::vector<stats_t<float> > InverseCrosstalkMatrix::
unfold(const std::vector<uint8_t>& q, const float x) {
  const unsigned N=q.size();
  std::vector<stats_t<float> > Q(N,stats_t<float>(0));

  if(N==1)                                 //optimize N==1
    Q[0] = stats_t<float>( q[0] )/(1-2*x);  
  else if(N==2) {                          //optimize N==2
    const double A=1-2*x; 
    Q[0] = ( A*stats_t<float>(q[0]) -x*stats_t<float>(q[1]) ) / (A*A-x*x);
    Q[1] = ( A*stats_t<float>(q[1]) -x*stats_t<float>(q[0]) ) / (A*A-x*x);
  } 
  else {                                   //general case
    InverseCrosstalkMatrix inverse(N,x);  
    for(unsigned i=0; i<(N+1)/2; i++) {
      for(unsigned j=i; j<N-i; j++) {
	const float Cij = inverse(i+1,j+1);
	Q[  i  ] += Cij * stats_t<float>( q[  j  ] ) ;  if( i!=j)   
	Q[  j  ] += Cij * stats_t<float>( q[  i  ] ) ;  if( N!=i+j+1) {
	Q[N-i-1] += Cij * stats_t<float>( q[N-j-1] ) ;  if( i!=j)
	Q[N-j-1] += Cij * stats_t<float>( q[N-i-1] ) ;
	}
      }
    }
  }
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
{ return N==0 || std::isinf(denominator) ? i==j : i>=j ? element(i,j) : element(j,i) ; }

inline
float InverseCrosstalkMatrix::
element(const unsigned i, const unsigned j) const 
{ return ( pow(lambdaM,N+1-i) - pow(lambdaP,N+1-i) ) * ( pow(lambdaM,j) - pow(lambdaP,j) ) / denominator; }

}
