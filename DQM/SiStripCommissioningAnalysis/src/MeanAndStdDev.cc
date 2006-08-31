#include "DQM/SiStripCommissioningAnalysis/interface/MeanAndStdDev.h"
#include <iostream>

using namespace std;

// ----------------------------------------------------------------------------
// 
MeanAndStdDev::MeanAndStdDev() 
  : s_(0.),
    x_(0.),
    xx_(0.),
    vec_()
{;}

// ----------------------------------------------------------------------------
// 
void MeanAndStdDev::add( const float& x,
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
void MeanAndStdDev::fit( Params& params ) {
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
