//
//  VVIObjF.cc  Version 2.0 
//
//  Port of CERNLIB G116 Functions vviden/vvidis
//
// Created by Morris Swartz on 1/14/2010.
// Copyright 2010 __TheJohnsHopkinsUniversity__. All rights reserved.
//
// V1.1 - make dzero call both fcns with a switch
// V1.2 - remove inappriate initializers and add methods to return non-zero/normalized region
// V2.0 - restructuring and speed improvements by V. Innocente
//

#ifndef SI_PIXEL_TEMPLATE_STANDALONE
// put CMSSW location of SimpleHelix.h here
#include "RecoLocalTracker/SiPixelRecHits/interface/VVIObjF.h"
#else
#include "VVIObjF.h"
#endif


#include <cmath>
#include <algorithm>
#include "vdt/vdtMath.h"

namespace VVIObjFDetails {
  void sincosint(float x, float & sint, float & cint);  //! Private version of the cosine and sine integral
  float expint(float x);    //! Private version of the exponential integral
  
  template<typename F>
  int dzero(float a, float b, float& x0, 
	    float& rv, float eps, int mxf, F func);
}



// ***************************************************************************************************************************************
//! Constructor
//! Set Vavilov parameters kappa and beta2 and define whether to calculate density fcn or distribution fcn
//! \param kappa - (input) Vavilov kappa parameter [0.01 (Landau-like) < kappa < 10. (Gaussian-like)]
//! \param beta2 - (input) Vavilov beta2 parameter (square of particle speed in v/c units)
//! \param mode  - (input) set to 0 to calculate the density function and to 1 to calculate the distribution function
// *************************************************************************************************************************************** 

VVIObjF::VVIObjF(float kappa, float beta2, int mode) : mode_(mode) {
  
  const float xp[9] = { 9.29,2.47,.89,.36,.15,.07,.03,.02,0.0 };
  const float xq[7] = { .012,.03,.08,.26,.87,3.83,11.0 };
  float h_[7];
  float  q, u, x, c1, c2, c3, c4, d1, h4, h5, h6, q2, x1, d, ll, ul, xf1, xf2, rv;
  int lp, lq, k, l, n;
  
  // Make sure that the inputs are reasonable
  
  if(kappa < 0.01f) kappa = 0.01f;
  if(kappa > 10.f) kappa = 10.f;
  if(beta2 < 0.f) beta2 = 0.f;
  if(beta2 > 1.f) beta2 = 1.f;
  
  float invKappa = 1.f/kappa;
  h_[4] = 1.f - beta2*0.42278433999999998f + (7.6f*invKappa);
  h_[5] = beta2;
  h_[6] = 1.f - beta2;
  h4 = - (7.6f*invKappa) - (beta2 * .57721566f + 1.f);
  h5 = vdt::fast_logf(kappa);
  h6 = invKappa;
  t0_ = (h4 - h_[4]*h5 - (h_[4] + beta2)*(vdt::fast_logf(h_[4]) + VVIObjFDetails::expint(h_[4])) + vdt::fast_expf(-h_[4]))/h_[4];
  
  // Set up limits for the root search
  
  for (lp = 0; lp < 9; ++lp) {
    if (kappa >= xp[lp]) break;
  }
  ll = -float(lp) - 1.5f;
  for (lq = 0; lq < 7; ++lq) {
    if (kappa <= xq[lq]) break;
  }
  ul = lq - 6.5f;
  auto f2 = [h_](float x) { return h_[4]-x+h_[5]*(vdt::fast_logf(std::abs(x))+VVIObjFDetails::expint(x))-h_[6]*vdt::fast_expf(-x);};
  VVIObjFDetails::dzero(ll, ul, u, rv, 1.e-3f, 100, f2);
  q = 1./u;
  t1_ = h4 * q - h5 - (beta2 * q + 1.f) * (vdt::fast_logf((fabs(u))) + VVIObjFDetails::expint(u)) + vdt::fast_expf(-u) * q;
  t_ = t1_ - t0_;
  omega_ = 6.2831853000000004f/t_;
  h_[0] = kappa * (beta2 * .57721566f + 2.f) + 9.9166128600000008f;
  if (kappa >= .07) {h_[0] += 6.90775527f;}
  h_[1] = beta2 * kappa;
  h_[2] = h6 * omega_;
  h_[3] = omega_ * 1.5707963250000001f;
  auto f1 = [h_](float x){ return h_[0]+h_[1]*vdt::fast_logf(h_[2]*x)-h_[3]*x;};
  VVIObjFDetails::dzero(5.f, 155.f, x0_, rv, 1.e-3f, 100, f1);
  n = x0_ + 1.;
  d = vdt::fast_expf(kappa * (beta2 * (.57721566f - h5) + 1.f)) * .31830988654751274f;
  a_[n - 1] = 0.f;
  if (mode_ == 0) {
    a_[n - 1] = omega_ * .31830988654751274f;
  }
  q = -1.;
  q2 = 2.;
  for (k = 1; k < n; ++k) {
    l = n - k;
    x = omega_ * k;
    x1 = h6 * x;
    VVIObjFDetails::sincosint(x1,c2,c1);
    c1 = vdt::fast_logf(x) - c1;
    vdt::fast_sincosf(x1,c3,c4);
    xf1 = kappa * (beta2 * c1 - c4) - x * c2;
    xf2 = x * c1 + kappa * (c3 + beta2 * c2) + t0_ * x;
    float s,c; vdt::fast_sincosf(xf2,s,c);
    if (mode_ == 0) {
      d1 = q * d * omega_ * vdt::fast_expf(xf1);
      a_[l - 1] = d1 * c;
      b_[l - 1] = -d1 * s;
    } else {
      d1 = q * d * vdt::fast_expf(xf1)/k;
      a_[l - 1] = d1 * s;
      b_[l - 1] = d1 * c;
      a_[n - 1] += q2 * a_[l - 1];
    }
    q = -q;
    q2 = -q2;
  }
  
} // VVIObjF

// *************************************************************************************************************************************
//! Vavilov function method
//! Returns density fcn (mode=0) or distribution fcn (mode=1)
//! \param x  - (input) Argument of function [typically defined as (Q-mpv)/sigma]
// ************************************************************************************************************************************* 


float VVIObjF::fcn(float x) const {
	
	// Local variables
	
	float f, u, y, a0, a1;
	float a2 = 0.;
	float b1, b0, b2, cof;
	int k, n, n1;
	
	n = x0_;
	if (x < t0_) {
		f = 0.f;
	} else if (x <= t1_) {
	  y = x - t0_;
	  u = omega_ * y - 3.141592653589793f;
	  float su,cu; vdt::fast_sincosf(u,su,cu);
	  cof = cu * 2.f;
	  a1 = 0.;
	  a0 = a_[0];
	  n1=n+1;
	  for (k = 2; k <= n1; ++k) {
	    a2 = a1;
	    a1 = a0;
	    a0 = a_[k - 1] + cof * a1 - a2;
	  }
	  b1 = 0.;
	  b0 = b_[0];
	  for (k = 2; k <= n; ++k) {
	    b2 = b1;
	    b1 = b0;
	    b0 = b_[k - 1] + cof * b1 - b2;
	  }
	  f = (a0 - a2) * .5f + b0 * su;
	  if (mode_ != 0) {f += y / t_;}
	} else {
	  f = 0.f;
	  if (mode_ != 0) {f = 1.f;}
	}
	return f;
} // fcn



// *************************************************************************************************************************************
//! Vavilov limits method
//! \param xl - (output) Smallest value of the argument for the density and the beginning of the normalized region for the distribution
//! \param xu - (output) Largest value of the argument for the density and the end of the normalized region for the distribution
// ************************************************************************************************************************************* 


void VVIObjF::limits(float& xl, float& xu) const {
	
   xl = t0_;
   xu = t1_;
	return;
} // limits


#include "sicif.h"
namespace VVIObjFDetails {
  void sincosint(float x, float & sint, float & cint) {
    sicif(x,sint,cint);
  }


float expint(float x) {
  
  // Initialized data
  
  const float zero = 0.;
  const float q2[7] = { .10340013040487,3.319092135933,
			 20.449478501379,41.280784189142,32.426421069514,10.041164382905,
			 1. };
  const float p3[6] = { -2.3909964453136,-147.98219500504,
			 -254.3763397689,-119.55761038372,-19.630408535939,-.9999999999036 
  };
  const float q3[6] = { 177.60070940351,530.68509610812,
			 462.23027156148,156.81843364539,21.630408494238,1. };
  const float p4[8] = { -8.6693733995107,-549.14226552109,
			 -4210.0161535707,-249301.39345865,-119623.66934925,
			 -22174462.775885,3892804.213112,-391546073.8091 };
  const float q4[8] = { 34.171875,-1607.0892658722,35730.029805851,
			 -483547.43616216,4285596.2461175,-24903337.574054,89192576.757561,
			 -165254299.72521 };
  const float a1[8] = { -2.1808638152072,-21.901023385488,
			 9.3081638566217,25.076281129356,-33.184253199722,60.121799083008,
			 -43.253113287813,1.0044310922808 };
  const float b1[8] = { 0.,3.9370770185272,300.89264837292,
			 -6.2504116167188,1003.6743951673,14.325673812194,2736.2411988933,
			 .52746885196291 };
  const float a2[8] = { -3.4833465360285,-18.65454548834,
			 -8.2856199414064,-32.34673303054,17.960168876925,1.7565631546961,
			 -1.9502232128966,.99999429607471 };
  const float b2[8] = { 0.,69.500065588743,57.283719383732,
			 25.777638423844,760.76114800773,28.951672792514,-3.4394226689987,
			 1.0008386740264 };
  const float a3[6] = { -27.780928934438,-10.10479081576,
			 -9.1483008216736,-5.0223317461851,-3.0000077799358,
			 1.0000000000704 };
  const float one = 1.;
  const float b3[6] = { 0.,122.39993926823,2.7276100778779,
			 -7.1897518395045,-2.9990118065262,1.999999942826 };
  const float two = 2.;
  const float three = 3.;
  const float x0 = .37250741078137;
  const float xl[6] = { -24.,-12.,-6.,0.,1.,4. };
  const float p1[5] = { 4.293125234321,39.894153870321,
			 292.52518866921,425.69682638592,-434.98143832952 };
  const float q1[5] = { 1.,18.899288395003,150.95038744251,
			 568.05252718987,753.58564359843 };
  const float p2[7] = { .43096783946939,6.9052252278444,
			 23.019255939133,24.378408879132,9.0416155694633,.99997957705159,
			 4.656271079751e-7 };
  
  // Local variables 
   float v, y, ap, bp, aq, dp, bq, dq;
  
  if (x <= xl[0]) {
    ap = a3[0] - x;
    for ( int i__ = 2; i__ <= 5; ++i__) {
      ap = a3[i__ - 1] - x + b3[i__ - 1] / ap;
    }
    y = vdt::fast_expf(-x) / x * (one - (a3[5] + b3[5] / ap) / x);
  } else if (x <= xl[1]) {
    ap = a2[0] - x;
    for ( int i__ = 2; i__ <= 7; ++i__) {
      ap = a2[i__ - 1] - x + b2[i__ - 1] / ap;
    }
    y = vdt::fast_expf(-x) / x * (a2[7] + b2[7] / ap);
  } else if (x <= xl[2]) {
    ap = a1[0] - x;
    for ( int i__ = 2; i__ <= 7; ++i__) {
      ap = a1[i__ - 1] - x + b1[i__ - 1] / ap;
    }
    y = vdt::fast_expf(-x) / x * (a1[7] + b1[7] / ap);
  } else if (x < xl[3]) {
    v = -two * (x / three + one);
    bp = zero;
    dp = p4[0];
    for ( int i__ = 2; i__ <= 8; ++i__) {
      ap = bp;
      bp = dp;
      dp = p4[i__ - 1] - ap + v * bp;
    }
    bq = zero;
    dq = q4[0];
    for ( int i__ = 2; i__ <= 8; ++i__) {
      aq = bq;
      bq = dq;
      dq = q4[i__ - 1] - aq + v * bq;
    }
    y = -vdt::fast_logf(-x / x0) + (x + x0) * (dp - ap) / (dq - aq);
  } else if (x == xl[3]) {
    return zero;
  } else if (x < xl[4]) {
    ap = p1[0];
    aq = q1[0];
    for ( int i__ = 2; i__ <= 5; ++i__) {
      ap = p1[i__ - 1] + x * ap;
      aq = q1[i__ - 1] + x * aq;
    }
    y = -vdt::fast_logf(x) + ap / aq;
  } else if (x <= xl[5]) {
    y = one / x;
    ap = p2[0];
    aq = q2[0];
    for ( int i__ = 2; i__ <= 7; ++i__) {
      ap = p2[i__ - 1] + y * ap;
      aq = q2[i__ - 1] + y * aq;
    }
    y = vdt::fast_expf(-x) * ap / aq;
  } else {
    y = one / x;
    ap = p3[0];
    aq = q3[0];
    for ( int i__ = 2; i__ <= 6; ++i__) {
      ap = p3[i__ - 1] + y * ap;
      aq = q3[i__ - 1] + y * aq;
    }
    y = vdt::fast_expf(-x) * y * (one + y * ap / aq);
  }
  return y;
} // expint
  



  template<typename F>
  int dzero(float a, float b, float& x0, 
	    float& rv, float eps, int mxf, F func) {
    /* System generated locals */
    float d__1, d__2, d__3, d__4;
    
    // Local variables
    float f1, f2, f3, u1, u2, x1, x2, u3, u4, x3, ca, cb, cc, fa, fb, ee, ff;
    int mc;
    float xa, xb, fx, xx, su4;
    
    xa = std::min(a,b);
    xb = std::max(a,b);
    fa = func(xa);
    fb = func(xb);
    if (fa * fb > 0.f) {
      rv = (xb - xa) * -2.f;
      x0 = 0.f;
      return 1;
    }
    mc = 0;
  L1:
    x0 = (xa + xb) * 0.5f;
    rv = x0 - xa;
    ee = eps * (std::abs(x0) + 1.f);
    if (rv <= ee) {
      rv = ee;
      ff = func(x0);
      return 0;
    }
    f1 = fa;
    x1 = xa;
    f2 = fb;
    x2 = xb;
  L2:
    fx = func(x0);
    ++mc;
    if (mc > mxf) {
      rv = (d__1 = xb - xa, fabs(d__1)) * -0.5f;
      x0 = 0.;
      return 0;
    }
    if (fx * fa > 0.f) {
      xa = x0;
      fa = fx;
    } else {
      xb = x0;
      fb = fx;
    }
  L3:
    u1 = f1 - f2;
    u2 = x1 - x2;
    u3 = f2 - fx;
    u4 = x2 - x0;
    if (u2 == 0.f || u4 == 0.f) {goto L1;}
    f3 = fx;
    x3 = x0;
    u1 /= u2;
    u2 = u3 / u4;
    ca = u1 - u2;
    cb = (x1 + x2) * u2 - (x2 + x0) * u1;
    cc = (x1 - x0) * f1 - x1 * (ca * x1 + cb);
    if (ca == 0.f) {
      if (cb == 0.f) {goto L1;}
      x0 = -cc / cb;
    } else {
      u3 = cb / (ca * 2.f);
      u4 = u3 * u3 - cc / ca;
      if (u4 < 0.f) {goto L1;}
      su4 = std::abs(u4);
      if (x0 + u3 < 0.f) {su4 = -su4;}
      x0 = -u3 + su4;
    }
    if (x0 < xa || x0 > xb) {goto L1;}
    // Computing MIN
    d__3 = (d__1 = x0 - x3, std::abs(d__1));
    d__4 = (d__2 = x0 - x2, std::abs(d__2));
    rv = std::min(d__3,d__4);
    ee = eps * (std::abs(x0) + 1);
    if (rv > ee) {
      f1 = f2;
      x1 = x2;
      f2 = f3;
      x2 = x3;
      goto L2;
    }
    fx = func(x0);
    if (fx == 0.f) {
      rv = ee;
      ff = func(x0);
      return 0;
    }
    if (fx * fa < 0.f) {
      xx = x0 - ee;
      if (xx <= xa) {
	rv = ee;
	ff = func(x0);
	return 0;
      }
      ff = func(xx);
      fb = ff;
      xb = xx;
    } else {
      xx = x0 + ee;
      if (xx >= xb) {
	rv = ee;
	ff = func(x0);
	return 0;
      }
      ff = func(xx);
      fa = ff;
      xa = xx;
    }
    if (fx * ff > 0.f) {
      mc += 2;
      if (mc > mxf) {
	rv = (d__1 = xb - xa, std::abs(d__1)) * -0.5f;
	x0 = 0.f;
	return 0;
      }
      f1 = f3;
      x1 = x3;
      f2 = fx;
      x2 = x0;
      x0 = xx;
      fx = ff;
      goto L3;
    }
    /* L4: */
    rv = ee;
    ff = func(x0);
    return 0;
  } // dzero
  
}
