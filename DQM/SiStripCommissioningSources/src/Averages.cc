#include "DQM/SiStripCommissioningSources/interface/Averages.h"
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace std;

// ----------------------------------------------------------------------------
// 
Averages::Averages() 
  : n_(0),
    s_(0.),
    x_(0.),
    xx_(0.),
    median_(),
    mode_()
{;}

// ----------------------------------------------------------------------------
// 
void Averages::add( const uint32_t& x,
		    const uint32_t& e ) {
  mode_[x]++;
  add( static_cast<float>(x),
       static_cast<float>(e) );
}
// ----------------------------------------------------------------------------
// 
void Averages::add( const uint32_t& x ) {
  mode_[x]++;
  add( static_cast<float>(x), -1. );
}

// ----------------------------------------------------------------------------
// 
void Averages::add( const float& x,
		    const float& e ) {
  n_++;
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
  median_.push_back(x);
}

// ----------------------------------------------------------------------------
// 
void Averages::add( const float& x ) { 
  add( x, -1. );
}

// ----------------------------------------------------------------------------
// 
void Averages::calc( Params& params ) {
  params.num_ = n_;
  if ( s_ > 0. ) { 
    float m = x_/s_;
    float t = xx_/s_ - m*m;
    if ( t > 0. ) { t = sqrt(t); } 
    else { t = 0.; }
    params.mean_ = m;
    params.rms_ = t;
    params.weight_ = s_;
  }
  if ( !median_.empty() ) {
    sort( median_.begin(), median_.end() );
    uint16_t index = median_.size()%2 ? median_.size()/2 : median_.size()/2-1;
    params.median_ = median_[index];
    params.max_ = median_.back();
    params.min_ = median_.front();
  }
  if ( !mode_.empty() ) {
    uint32_t max = 0;
    std::map<uint32_t,uint32_t>::const_iterator imap = mode_.begin();
    for ( ; imap != mode_.end(); imap++ ) {
      if ( imap->second > max ) { 
	max = imap->second;
	params.mode_ = imap->first;
      }
    }    
  }
  
}
