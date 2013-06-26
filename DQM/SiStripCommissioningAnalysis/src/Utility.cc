#include "DQM/SiStripCommissioningAnalysis/src/Utility.h"
#include <algorithm>

// ----------------------------------------------------------------------------
// 
sistrip::LinearFit::LinearFit() 
  : x_(),
    y_(),
    e_(),
    ss_(0.),
    sx_(0.),
    sy_(0.) 
{;}

// ----------------------------------------------------------------------------
// 
void sistrip::LinearFit::add( const float& x,
			      const float& y ) {
  float e = 1.; // default
  x_.push_back(x);
  y_.push_back(y);
  e_.push_back(e);
  float wt = 1. / sqrt(e); 
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
