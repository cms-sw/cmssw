// this test documents the derivation of the fast deltaphi used in gpu doublet code..
//
//
//
#include<cmath>
#include<algorithm>
#include<numeric>
#include<cassert>

/**
| 1) circle is parameterized as:                                              |
|    C*[(X-Xp)**2+(Y-Yp)**2] - 2*alpha*(X-Xp) - 2*beta*(Y-Yp) = 0             |
|    Xp,Yp is a point on the track (Yp is at the center of the chamber);      |
|    C = 1/r0 is the curvature  ( sign of C is charge of particle );          |
|    alpha & beta are the direction cosines of the radial vector at Xp,Yp     |
|    i.e.  alpha = C*(X0-Xp),                                                 |
|          beta  = C*(Y0-Yp),                                                 |
|    where center of circle is at X0,Y0.                                      |
|    Alpha > 0                                                                |
|    Slope dy/dx of tangent at Xp,Yp is -alpha/beta.                          |
| 2) the z dimension of the helix is parameterized by gamma = dZ/dSperp       |
|    this is also the tangent of the pitch angle of the helix.                |
|    with this parameterization, (alpha,beta,gamma) rotate like a vector.     |
| 3) For tracks going inward at (Xp,Yp), C, alpha, beta, and gamma change sign|
|
*/

template<typename T>
class FastCircle {

public:

  FastCircle(){}
  FastCircle(T x1, T y1,
	     T x2, T y2,
	     T x3, T y3) { 
    compute(x1,y1,x2,y2,x3,y3);
  }

  void compute(T x1, T y1,
	       T x2, T y2,
	       T x3, T y3);
  

  T m_xp;
  T m_yp;
  T m_c;
  T m_alpha;
  T m_beta;

};


template<typename T>
void FastCircle<T>::compute(T x1, T y1,
			    T x2, T y2,
			    T x3, T y3) {
  bool flip = std::abs(x3-x1) > std::abs(y3-y1);
   
  auto x1p = x1-x2;
  auto y1p = y1-y2;
  auto d12 = x1p*x1p + y1p*y1p;
  auto x3p = x3-x2;
  auto y3p = y3-y2;
  auto d32 = x3p*x3p + y3p*y3p;

  if (flip) {
    std::swap(x1p,y1p);
    std::swap(x3p,y3p);
  }

  auto num = x1p*y3p-y1p*x3p;  // num also gives correct sign for CT
  auto det = d12*y3p-d32*y1p;
  if( std::abs(det)==0 ) {
    // and why we flip????
  }
  auto ct  = num/det;
  auto sn  = det>0 ? T(1.) : T(-1.);  
  auto st2 = (d12*x3p-d32*x1p)/det;
  auto seq = T(1.) +st2*st2;
  auto al2 = sn/std::sqrt(seq);
  auto be2 = -st2*al2;
  ct *= T(2.)*al2;
  
  if (flip) {
    std::swap(x1p,y1p);
    std::swap(al2,be2);
    al2 = -al2;
    be2 = -be2;
    ct = -ct;
  }
  
  m_xp = x1;
  m_yp = y1;
  m_c= ct;
  m_alpha = al2 - ct*x1p;
  m_beta = be2 - ct*y1p;
  
}



// compute curvature given two points (and origin)
float fastDPHI(float ri, float ro, float dphi) {

  /*
  x3=0 y1=0 x1=0;
  y3=ro
  */

  // auto x2 = ri*dphi;
  // auto y2 = ri*(1.f-0.5f*dphi*dphi);


  /*
  auto x1p = x1-x2;
  auto y1p = y1-y2;
  auto d12 = x1p*x1p + y1p*y1p;
  auto x3p = x3-x2;
  auto y3p = y3-y2;
  auto d32 = x3p*x3p + y3p*y3p;
  */
   
  /*
  auto x1p = -x2;
  auto y1p = -y2;
  auto d12 = ri*ri;
  auto x3p = -x2;
  auto y3p = ro-y2;
  auto d32 = ri*ri + ro*ro - 2.f*ro*y2;
  */
  

  // auto rat = (ro -2.f*y2);
  // auto det =  ro - ri - (ro - 2.f*ri -0.5f*ro)*dphi*dphi;

  //auto det2 = (ro-ri)*(ro-ri) -2.*(ro-ri)*(ro - 2.f*ri -0.5f*ro)*dphi*dphi;
  // auto seq = det2 +  dphi*dphi*(ro-2.f*ri)*(ro-2.f*ri);    // *rat2;
  // auto seq = (ro-ri)*(ro-ri) +  dphi*dphi*ri*ro;

  // and little by little simplifing and removing higher over terms 
  // we get
  auto r2 = (ro-ri)*(ro-ri)/(dphi*dphi) + ri*ro;


  // d2 = (ro-ri)*(ro-ri)/(4.f*r2 -ri*ro);  
  // return -2.f*dphi/std::sqrt(seq);

  return -1.f/std::sqrt(r2/4.f);
  
}



#include<iostream>

template<typename T>
bool equal(T a, T b) {
  //  return float(a-b)==0;
  return std::abs(float(a-b)) < std::abs(0.01f*a);
}



int n=0;
void go(float ri, float ro, float dphi, bool print=false) {
  ++n;
  float x3 = 0.f, y3 = ro;
  float x2 = ri*sin(dphi);
  float y2 = ri*cos(dphi);

  
  FastCircle<float> c(0,0,
		  x2,y2,
                  x3,y3);

  auto cc = fastDPHI(ri,ro,dphi);
  if (print) std::cout << c.m_c << ' ' << cc << std::endl;
  assert(equal(c.m_c,cc));

  
}

int main() {


  go(4.,7.,0.1, true);

  for (float r1=2; r1<15; r1+=1)
    for (float dr=0.5; dr<10; dr+=0.5)
      for (float dphi=0.02; dphi<0.2; dphi+=0.2)
	go(r1,r1+dr,dphi);

  std::cout << "done " << n << std::endl;
  return 0;
};

