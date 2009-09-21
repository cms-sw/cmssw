#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"

#include <algorithm>
#include <iterator>
#include <iostream>

using namespace std;

bool SiStripLatency::put( const uint32_t detId, const uint16_t apv, const float & latency, const uint16_t mode )
{
  // Store all the values in the vectors
  uint32_t detIdAndApv = (detId << 2) | apv;
  latIt pos = lower_bound(latencies_.begin(), latencies_.end(), detIdAndApv, OrderByDetIdAndApv());

  if( pos != latencies_.end() && pos->detIdAndApv == detIdAndApv ) {
    cout << "Value already inserted, skipping insertion" << endl;
    return false;
  }
  // cout << "Filling with: latency = " << latency << ", mode = " << mode << endl;
  latencies_.insert(pos, Latency(detIdAndApv, latency, mode));

  return true;
}

void SiStripLatency::compress()
{
  // cout << "Starting compression" << endl;
  // cout << "Total number of elements before compression = " << latencies_.size() << endl;

  // int i = 0;
  // for( latIt it = latencies_.begin(); it != latencies_.end(); ++it, ++i ) {
  //   cout << "latency["<<i<<"] = " << it->latency << ", mode["<<i<<"] = " << (int)it->mode << " for detIdAndApv = " << it->detIdAndApv << endl;;
  // }
  // Remove latency duplicates. Note that unique is stable.
  // CANNOT USE THIS: it will leave one element, but you do not know which one.
  // unique(latencies_.begin(), latencies_.end(), EqualByLatency());
  // Cannot use lower_bound or upper_bound with latencies because the vector is sorted in detIdAndApvs and not in latencies
  // For the same reason cannot use equal_range.

  // Go through the elements one by one and remove the current one if it has the same latency as the next one
  // for( latIt lat = latencies_.begin(); lat != latencies_.end(); ++lat ) {
  latIt lat = latencies_.begin();
  while( lat != latencies_.end() ) {
    // If it is not the last and it has the same latency and mode as the next one remove it
    if( ((lat+1) != latencies_.end()) && ((lat+1)->mode == lat->mode) && ((lat+1)->latency == lat->latency) ) {
      lat = latencies_.erase(lat);
    }
    else {
      ++lat;
    }
  }
  // cout << "Total number of elements after compression = " << latencies_.size() << endl;
  // i = 0;
  // for( latIt it = latencies_.begin(); it != latencies_.end(); ++it, ++i ) {
  //   cout << "latency["<<i<<"] = " << it->latency << ", mode["<<i<<"] = " << (int)it->mode  << ", for detIdAndApv = " << it->detIdAndApv << endl;;
  // }
}

// const latConstIt SiStripLatency::position(const uint32_t detId, const uint16_t apv) const
// {
//   if( latencies_.empty() ) {
//     // cout << "SiStripLatency: Error, range is empty" << endl;
//     return latencies_.end();
//   }
//   uint32_t detIdAndApv = (detId << 2) | apv;
//   latConstIt pos = lower_bound(latencies_.begin(), latencies_.end(), detIdAndApv, OrderByDetIdAndApv());
//   return pos;
// }

float SiStripLatency::latency(const uint32_t detId, const uint16_t apv) const
{
  const latConstIt & pos = position(detId, apv);
  if( pos == latencies_.end() ) {
    return -1.;
  }
  return pos->latency;
}

uint16_t SiStripLatency::mode(const uint32_t detId, const uint16_t apv) const
{
  const latConstIt & pos = position(detId, apv);
  if( pos == latencies_.end() ) {
    return 0;
  }
  return pos->mode;
}

pair<float, uint16_t> SiStripLatency::latencyAndMode(const uint32_t detId, const uint16_t apv) const
{
  const latConstIt & pos = position(detId, apv);
  if( pos == latencies_.end() ) {
    return make_pair(-1., 0);
  }
  return make_pair(pos->latency, pos->mode);
}

float SiStripLatency::singleLatency() const
{
  if( latencies_.size() == 1 ) {
    return latencies_[0].latency;
  }
  int differentLatenciesNum = 0;
  // Count the number of different latencies
  for( latConstIt it = latencies_.begin(); it != latencies_.end()-1; ++it ) {
    if( it->latency != (it+1)->latency ) {
      ++differentLatenciesNum;
    }
  }
  if( differentLatenciesNum == 0 ) {
    return latencies_[0].latency;
  }
  return -1;
}

uint16_t SiStripLatency::singleMode() const
{
  if( latencies_.size() == 1 ) {
    return latencies_[0].mode;
  }
  int differentModesNum = 0;
  // Count the number of different modes
  for( latConstIt it = latencies_.begin(); it != latencies_.end()-1; ++it ) {
    if( it->mode != (it+1)->mode ) {
      ++differentModesNum;
    }
  }
  if( differentModesNum == 0 ) {
    return latencies_[0].mode;
  }
  return 0;
}

// pair<float, uint16_t> SiStripLatency::singleLatencyAndMode() const
// {
//   if( latencies_.size() == 1 ) {
//     return make_pair(latencies_[0].latency, latencies_[0].mode);
//   }
//   return make_pair(-1, 0);
// }
