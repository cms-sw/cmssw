// This is CSCStripData.cc

#include "RecoLocalMuon/CSCRecHitD/src/CSCStripData.h"

#include <iostream>
// required for ostream_iterator...
#include <iterator>

std::ostream & operator<<(std::ostream & os, const CSCStripData & data) {
  os << "CSCStripData " << std::endl
     << "------------ " << std::endl
     << "no. of time bins = " << data.ntbins_ << std::endl
     << "strip = " << data.istrip_ 
     << ", phmax = " << data.phmax_ 
     << ", tmax = " << data.tmax_ << std::endl
     << "phraw: " << std::endl;
  std::copy( data.phRaw_.begin(), data.phRaw_.end(), std::ostream_iterator<int>(os,"\n") );
  os << "ph: " << std::endl;
  std::copy( data.ph_.begin(), data.ph_.end(), std::ostream_iterator<float>(os,"\n") );
  return os;     
}

// Define space for static 
const int CSCStripData::ntbins_;
