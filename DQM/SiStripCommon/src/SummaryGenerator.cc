#include "DQM/SiStripCommon/interface/SummaryGenerator.h"

using namespace std;

// -----------------------------------------------------------------------------
// 
pair<float,float> SummaryGenerator::range() {

  if ( map_.empty() ) { return pair<float,float>(0.,0.); }
  
  pair<float,float> range = pair<float,float>(0.,0.);
  range.first = -1.e6;
  range.second = 1.e6;
  
  map< string, pair<float,float> >::const_iterator iter = map_.begin();
  for ( ; iter != map_.end(); iter++ ) {
    if ( iter->second.first > range.first ) { range.first = iter->second.first; }
    if ( iter->second.first < range.second ) { range.second = iter->second.first; }
  }
  return range;

}

