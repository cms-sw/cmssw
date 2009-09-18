#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"

#include <algorithm>
#include <iterator>
#include <iostream>

using namespace std;

bool SiStripLatency::put( const uint32_t detId, const uint16_t apv, const float & latency )
{
  // Store all the values in the vectors
  uint32_t detIdAndApv = (detId << 2) | apv;
  latIt pos = lower_bound(latencies_.begin(), latencies_.end(), detIdAndApv, OrderByDetIdAndApv());

  if( pos != latencies_.end() && pos->detIdAndApv == detIdAndApv ) {
    cout << "Value already inserted, skipping insertion" << endl;
    return false;
  }
  latencies_.insert(pos, Latency(detIdAndApv, latency));

  return true;
}

void SiStripLatency::compress()
{
  cout << "Starting compression" << endl;
  cout << "Total number of elements before compression = " << latencies_.size() << endl;

  int i = 0;
  for( latIt it = latencies_.begin(); it != latencies_.end(); ++it, ++i ) {
    cout << "latency["<<i<<"] = " << it->latency << ", for detIdAndApv = " << it->detIdAndApv << endl;;
  }
  // Remove latency duplicates. Note that unique is stable.
  // CANNOT USE THIS: it will leave one element, but you do not know which one.
  // unique(latencies_.begin(), latencies_.end(), EqualByLatency());
  // Cannot use lower_bound or upper_bound with latencies because the vector is sorted in detIdAndApvs and not in latencies
  // For the same reason cannot use equal_range.

  // Go through the elements one by one and remove the current one if it has the same latency as the next one
  // for( latIt lat = latencies_.begin(); lat != latencies_.end(); ++lat ) {
  latIt lat = latencies_.begin();
  while( lat != latencies_.end() ) {
    // If it is not the the last and it has the same latency as the next one
    if( ((lat+1) != latencies_.end()) && ((lat+1)->latency == lat->latency) ) {
      // Remove the current one
      lat = latencies_.erase(lat);
    }
    else {
      ++lat;
    }
  }
  cout << "Total number of elements after compression = " << latencies_.size() << endl;
  i = 0;
  for( latIt it = latencies_.begin(); it != latencies_.end(); ++it, ++i ) {
    cout << "latency["<<i<<"] = " << it->latency << ", for detIdAndApv = " << it->detIdAndApv << endl;;
  }
}

float SiStripLatency::get(const uint32_t detId, const uint16_t apv)
{
  if( latencies_.empty() ) {
    cout << "SiStripLatency: Error, range is empty" << endl;
    return -1;
  }
  uint32_t detIdAndApv = (detId << 2) | apv;
  latIt pos = lower_bound(latencies_.begin(), latencies_.end(), detIdAndApv, OrderByDetIdAndApv());

  // When the position is after the last one return -1
  if( pos == latencies_.end() ) {
    return -1.;
  }
  return pos->latency;
}

float SiStripLatency::getSingleLatency()
{
  if( latencies_.size() == 1 ) {
    return latencies_[0].latency;
  }
  return -1;
}
