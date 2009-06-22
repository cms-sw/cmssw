#include "RecoLocalTracker/SiStripRecHitConverter/interface/CrosstalkInversion.h"

std::vector<float> InverseCrosstalkMatrix::
unfold(const std::vector<uint8_t>& q, const float xtalk) {
  const unsigned N=q.size();
  InverseCrosstalkMatrix inverse(N,xtalk);
  
  std::vector<float> Q(N,0);
  
  for(unsigned i=0; i<(N+1)/2; i++) {
    for(unsigned j=i; j<N-i; j++) {
      const float Cij = inverse(i+1,j+1);
      Q[  i  ] += q[  j  ] *Cij;  if( i!=j)   
      Q[  j  ] += q[  i  ] *Cij;  if( N!=i+j+1) {
      Q[N-i-1] += q[N-j-1] *Cij;  
      Q[N-j-1] += q[N-i-1] *Cij;
      }
    }
  }
  return Q;
}

InverseCrosstalkMatrix::
InverseCrosstalkMatrix(const unsigned N, const float x)
  : r(-x), N(N)
{
  if(r!=0 && N>1) {
    const float sq = sqrt(4*r+1);
    lambda1 = 1+(1+sq)/(2*r);
    lambda2 = 1+(1-sq)/(2*r);
    rmu0 = r*mu(0);
  }
}

inline
float InverseCrosstalkMatrix::
operator()(const unsigned i, const unsigned j) const { 
  return 
    r==0 ? i==j :
    N==1 ? 1/(1+2*r) :
    i>=j ? element(i,j) : element(j,i);
}

inline
float InverseCrosstalkMatrix::
element(const unsigned i, const unsigned j) const 
{ return mu(i)*mu(N+1-j)/rmu0; }

inline
float InverseCrosstalkMatrix::
mu(const unsigned i) const 
{ return ( pow(lambda1, N-i+1) - pow(lambda2, N-i+1) ) / (lambda1 - lambda2); }
