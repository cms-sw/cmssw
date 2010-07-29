#include "DataFormats/Math/interface/sse_mathfun.h"

#ifdef  CMS_USE_SSE

// vertical vector (for floats and SSE)
const size_t SIZE = 10;

template<size_t S>
union VVEC {
  const size_t SSSESIZE = (S+3)/4;
  const size_t AllignSIZE = 4*SSSESIZE;
  __m128 vec[SSSESIZE];
  float __attribute__ ((aligned(16))) arr[AllignSIZE];
};



void chi2(VVEC const & ampl, VVEC const & err2, float t, float sumAA, float&chi2, float&amp) {
  typedef float Scalar;
  typedev  __m128 Vec;
  Scalar const eps = Scalar(1e-6);
  Scalar const denom =  Scalar(1)/Scalar(SIZE);

  Vec one = _mm_set1_ps(1);
  Vec eps = _mm_set1_ps(1e-6);

  Vec tv = _mm_set1_ps(t);
  Vec alpha = _mm_set1_ps(2);
  Vec overab = _mm_set1_ps(0.2);

  VVEC index;
  for(unsigned int it = 0; it < VVEC::AllignSIZE; it++){
    index.arr[it]=it;
  }

  Vec sumAf =  _mm_setzero_ps();
  Vec sumff =  _mm_setzero_ps();


  for(unsigned int it = 0; it < VVEC::SSESIZE; it++){
    Vec offset =  _mm_mul_ps(_mm_sub_ps(index.vec[it],tv),overab);
    Vec term1 =  _mm_add_ps(one,offset);
    Vec cmp = _mm_cmp_gt(term1,eps);
    
    Vec f = exp_pf( _mm_sub_ps(_mm_mul_ps(alpha,log_pf(term1)),offset) );
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


  chi2 = sumAA;
  amp = 0;
  if( sumff > 0 ){
    amp = sumAf/sumff;
    chi2 = sumAA - sumAf*amp;
  }
  chi2 *=denom;
}



int main() {  

  Vec4F x(0.,-1.,1.,1000.);
  std::cout << x << std::endl;
  
  Vec4F y.vec = exp_ps(x.vec);
  std::cout << y << std::endl;

  Vec4F z.vec = log_ps(x.vec);
  std::cout << z << std::endl;

  Vec4F y.vec = log_ps(y.vec);
  std::cout << y << std::endl;

  Vec4F z.vec = exp_ps(z.vec);
  std::cout << z << std::endl;

  Vec4F k(0.,-1.,1.,1.1e-3);
  Vec4F eps(1.e-3);
  Vec4F cmp; cmp.vec = _mm_cmp_gt(k,eps.vec);
  std::cout << k << std::endl;

  return 0;

}

#else
int main() {
  return 0;
}
#endif



#endif

