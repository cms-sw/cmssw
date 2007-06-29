#include "DQM/SiStripCommissioningAnalysis/interface/Utilities.h"
#include <iostream>
#include <math.h>

using namespace sistrip;

// ----------------------------------------------------------------------------
// 
sistrip::LinearFit::LinearFit() 
  : x_(),
    y_(),
    e_(),
    ss_(0.),
    sx_(0.),
    sy_(0.) 
{ 
}

// ----------------------------------------------------------------------------
// 
void sistrip::LinearFit::add( const float& x,
			      const float& y ) {
  x_.push_back(x);
  y_.push_back(y);
  float wt = 1.;
  ss_ += wt;
  sx_ += x*wt;
  sy_ += y*wt;
}

// ----------------------------------------------------------------------------
// 
void sistrip::LinearFit::add( const float& x,
			      const float& y,
			      const float& e ) {
  if ( e > 0. ) { 
    x_.push_back(x);
    y_.push_back(y);
    e_.push_back(e);
    float wt = 1. / sqrt(e); 
    ss_ += wt;
    sx_ += x*wt;
    sy_ += y*wt;
  } 
}

// ----------------------------------------------------------------------------
// 
void sistrip::LinearFit::fit( Params& params ) {

  float s2 = 0.;
  float b = 0;
  for ( uint16_t i = 0; i < x_.size(); i++ ) {
    float t = ( x_[i] - sx_/ss_ ) / e_[i]; 
    s2 += t*t;
    b += t * y_[i] / e_[i];
  }
  
  // Set parameters
  params.n_ = x_.size();
  params.b_ = b / s2;
  params.a_ = ( sy_ - sx_ * params.b_ ) / ss_;
  params.erra_ = sqrt( ( 1. + (sx_*sx_) / (ss_*s2) ) / ss_ );
  params.errb_ = sqrt( 1. / s2 );
  
  /*
    params.chi2_ = 0.;
    *q=1.0;
    if (mwt == 0) {
    for (i=1;i<=ndata;i++)
    *chi2 += SQR(y[i]-(*a)-(*b)*x[i]);
    sigdat=sqrt((*chi2)/(ndata-2));
    *sigb *= sigdat;
    */    
  
}

// ----------------------------------------------------------------------------
// 
sistrip::MeanAndStdDev::MeanAndStdDev() 
  : s_(0.),
    x_(0.),
    xx_(0.),
    vec_()
{;}

// ----------------------------------------------------------------------------
// 
void sistrip::MeanAndStdDev::add( const float& x,
			 const float& e ) {
  if ( e > 0. ) { 
    float wt = 1. / sqrt(e); 
    s_ += wt;
    x_ += x*wt;
    xx_ += x*x*wt;
  } else {
    s_++;
    x_ += x;
    xx_ += x*x;
  }
  vec_.push_back(x);
}

// ----------------------------------------------------------------------------
// 
void sistrip::MeanAndStdDev::fit( Params& params ) {
  if ( s_ > 0. ) { 
    float m = x_/s_;
    float t = xx_/s_ - m*m;
    if ( t > 0. ) { t = sqrt(t); } 
    else { t = 0.; }
    params.mean_ = m;
    params.rms_  = t;
  }
  if ( !vec_.empty() ) {
    sort( vec_.begin(), vec_.end() );
    uint16_t index = vec_.size()%2 ? vec_.size()/2 : vec_.size()/2-1;
    params.median_ = vec_[index];
  }      
}


















/* 
// ----------------------------------------------------------------------------
// 
uint16_t LinearFit::fit( float& intercept, float& gradient ) {
  static float n  = static_cast<float>(n_);
  static float xx = (1./n)*xx_ - ( (x_*x_) / (n/n) );
  static float xy = (1./n)*xy_ - ( (y_*x_) / (n/n) );
  gradient = xy / xx;
  intercept = y_/n - gradient*x_/n;

  LogTrace(mlCommissioning_) << " here "
       << (1./n)*xx_ - ( (x_*x_) / (n/n) ) << " " 
       << 1./(n*xx_) - ( (x_*x_) / (n/n) );
  return n_; 
}

*/








/*
// ----------------------------------------------------------------------------
// 
void LinearFit::add( const float& value_x,
		     const float& value_y,
		     const float& error_y ) {
  n_++;
  x_ += value_x;
  y_ += value_y;
  xx_ += value_x*value_x;
  yy_ += value_y*value_y;
  xy_ += value_x*value_y;
}

*/
  
/* 
   data points x[1..ndata], y[1..ndata] 
   with individual standard deviations sig[1..ndata], 
   fit them to a straight line y = a + bx by minimizing q2. 
   Returned are a,b and their respective probable uncertainties siga and sigb
   the chi-square chi2, and the goodness-of-fit probability q 
   (that the fit would have £q2 this large or larger). 

   If mwt=0 on input, 
   then the standard deviations are assumed to be unavailable: 
   q is returned as 1.0 and the 
   normalization of chi2 is to unit standard deviation on all points.

   
   void fit(float x[], float y[], int ndata, float sig[], int mwt, float *a,
   float *b, float *siga, float *sigb, float *chi2, float *q) {
    
	   {

    if (mwt) { Accumulate sums ...
	ss=0.0;
      for (i=1;i<=ndata;i++) { ...with weights
	  wt=1.0/SQR(sig[i]);
	ss += wt;
	sx += x[i]*wt;
	sy += y[i]*wt;
      }
    } else {
      for (i=1;i<=ndata;i++) { ...or without weights.
			       sx += x[i];
			       sy += y[i];
      }
      ss=ndata;
    }

    int i;
    float wt,t,sxoss,sx=0.0,sy=0.0,st2=0.0,ss,sigdat;
    *b=0.0;

    sxoss=sx/ss;
    if (mwt) {
      for (i=1;i<=ndata;i++) {
	t=(x[i]-sxoss)/sig[i];
	st2 += t*t;
	*b += t*y[i]/sig[i];
      }
    } else {
      for (i=1;i<=ndata;i++) {
	t=x[i]-sxoss;
	st2 += t*t;
	*b += t*y[i];
      }
    }
    *b /= st2; Solve for a, b, £ma, and £mb.
    *a=(sy-sx*(*b))/ss;
    *siga=sqrt((1.0+sx*sx/(ss*st2))/ss);
    *sigb=sqrt(1.0/st2);
  
  
  }

  *chi2=0.0; // Calculate £q2.
  *q=1.0;
  if (mwt == 0) {
    for (i=1;i<=ndata;i++)
      *chi2 += SQR(y[i]-(*a)-(*b)*x[i]);
    sigdat=sqrt((*chi2)/(ndata-2)); For unweighted data evaluate typical
				      sig using chi2, and adjust
				      the standard deviations.
				      *siga *= sigdat;
    *sigb *= sigdat;
  } else {
    for (i=1;i<=ndata;i++)
      *chi2 += SQR((y[i]-(*a)-(*b)*x[i])/sig[i]);
    if (ndata>2) *q=gammq(0.5*(ndata-2),0.5*(*chi2)); Equation (15.2.12).
							}
  }



*/
