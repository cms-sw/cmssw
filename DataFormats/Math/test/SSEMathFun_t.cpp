#include "DataFormats/Math/interface/sse_mathfun.h"
#include "DataFormats/Math/interface/SSEVec.h"
#include<iostream>

#ifdef  CMS_USE_SSE

// vertical vector (for floats and SSE)
const size_t SIZE = 10;

template<size_t S>
struct VVECSIZE {
  static const size_t size = S;
  static const size_t ssesize = (S+3)/4;
  static const size_t arrsize = 4*ssesize;
};

template<size_t S>
union VVEC {
  typedef VVECSIZE<S> SIZE;
  __m128 vec[SIZE::ssesize];
  float __attribute__ ((aligned(16))) arr[SIZE::arrsize];
};


typedef VVEC<10> V10;
static const size_t ssesize = V10::SIZE::ssesize;
static const size_t arrsize = V10::SIZE::arrsize;


void compChi2Scalar(V10 const & ampl, V10 const & err2, float t, float sumAA, float& chi2, float& amp) {
  typedef float Scalar;

  Scalar sumAf = 0;
  Scalar sumff = 0;
  Scalar const eps = Scalar(1e-6);
  Scalar const denom =  Scalar(1)/Scalar(SIZE);

  Scalar alpha = 2.;
  Scalar overab = 0.2;

  for(unsigned int it = 0; it < SIZE; it++){
    Scalar offset = (Scalar(it) - t)*overab;
    Scalar term1 = Scalar(1) + offset;
    if(term1>eps){
      Scalar f = std::exp( alpha*(std::log(term1) - offset) );
      sumAf += ampl.arr[it]*(f*err2.arr[it]);
      sumff += f*(f*err2.arr[it]);
    }
  }
 
  chi2 = sumAA;
  amp = 0;
  if( sumff > 0 ){
    amp = sumAf/sumff;
    chi2 = sumAA - sumAf*amp;
  }
  chi2 *=denom;
}

void compChi2(V10 const & ampl, V10 const & err2, float t, float sumAA, float& chi2, float& amp) {
  typedef float Scalar;
  typedef  __m128 Vec;
  Scalar const denom =  Scalar(1)/Scalar(SIZE);

  Vec one = _mm_set1_ps(1);
  Vec eps = _mm_set1_ps(1e-6);

  Vec tv = _mm_set1_ps(t);
  Vec alpha = _mm_set1_ps(2);
  Vec overab = _mm_set1_ps(0.2);

  V10 index;
  for(unsigned int it = 0; it < arrsize; it++){
    index.arr[it]=it;
  }

  Vec sumAf =  _mm_setzero_ps();
  Vec sumff =  _mm_setzero_ps();


  for(unsigned int it = 0; it < ssesize; it++){
    Vec offset =  _mm_mul_ps(_mm_sub_ps(index.vec[it],tv),overab);
    Vec term1 =  _mm_add_ps(one,offset);
    Vec cmp = _mm_cmpgt_ps(term1,eps);
    
    Vec f = exp_ps( _mm_sub_ps(_mm_mul_ps(alpha,log_ps(term1)),offset) );
    //Vec f = _mm_or_ps(_mm_andnot_ps(cmp, _mm_setzero_ps()), _mm_and_ps(cmp, f));
    f = _mm_and_ps(cmp, f);
    Vec fe = _mm_mul_ps(f, err2.vec[it]);
    sumAf = _mm_add_ps(sumAf, _mm_mul_ps(ampl.vec[it],fe));
    sumff = _mm_add_ps(sumff, _mm_mul_ps(f,fe));
  }
  
  sumAf = _mm_hadd_ps(sumAf,sumAf);
  sumAf = _mm_hadd_ps(sumAf,sumAf);
  sumff = _mm_hadd_ps(sumff,sumff);
  sumff = _mm_hadd_ps(sumff,sumff);

  float af; _mm_store_ss(&af,sumAf);
  float ff; _mm_store_ss(&ff,sumff);
  
  chi2 = sumAA;
  amp = 0;
  if( ff > 0 ){
    amp = af/ff;
    chi2 = sumAA - af*amp;
  }
  chi2 *=denom;
}



int main() {  
  using mathSSE::Vec4F;
  typedef  __m128 Vec;

  Vec4F x(0.,-1.,1.,1000.);
  std::cout << x << std::endl;
  
  Vec4F y; y.vec = exp_ps(x.vec);
  std::cout << y << std::endl;

  Vec4F z; z.vec = log_ps(x.vec);
  std::cout << z << std::endl;

  // some of z are nan... check if I can find out..
  Vec4F m; m.vec = _mm_cmpeq_ps(_mm_andnot_ps(z.vec, *(Vec*)_ps_mant_mask),_mm_setzero_ps());
  std::cout << m << std::endl;

  y.vec = log_ps(y.vec);
  std::cout << y << std::endl;

  z.vec = exp_ps(z.vec);
  std::cout << z << std::endl;

  Vec4F k(0.1,-1.,1.1e-3,1.);
  Vec4F eps(1.e-3);
  Vec4F cmp; cmp.vec = _mm_cmpgt_ps(k.vec,eps.vec);
  std::cout << cmp  << std::endl;
  k.vec = _mm_and_ps(cmp.vec, k.vec  ); 
  std::cout << k  << std::endl;


  V10 ampl;
  V10 err2;
  float sumAA=0;
  for(unsigned int it = 0; it < SIZE; it++){
    ampl.arr[it] = abs(SIZE/2-it)*10;
    err2.arr[it] = std::pow(1./(0.05*ampl.arr[it]),2);
    sumAA+=ampl.arr[it]*ampl.arr[it]*err2.arr[it];
  }

  
  float chi2=0;
  float amp=0;
  compChi2(ampl, err2, 4.7, sumAA, chi2, amp);
  std::cout << chi2 << " " << amp << std::endl;
  compChi2Scalar(ampl, err2, 4.7, sumAA, chi2, amp);
   std::cout << chi2 << " " << amp << std::endl;
   
  return 0;

}

#else
int main() {
  return 0;
}
#endif


