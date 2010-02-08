#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>
#include <iterator>
#include <iostream>
#include <sstream>

using namespace std;

bool SiStripLatency::put( const uint32_t detId, const uint16_t apv, const uint16_t latency, const uint16_t mode )
{
  if( detId > 536870911 ) {
    stringstream error;
    error << "ERROR: the detId = " << detId << " is bigger than the maximum acceptable value = 2^(29) - 1 = " << 536870911 << endl;
    error << "Since we are using 29 bits for the detId and 3 bits for the apv value. The maximum tracker detId at the moment" << endl;
    error << "of the writing of this class was 47017836 as defined in CalibTracker/SiStripCommon/data/SiStripDetInfo.dat." << endl;
    error << "If the maximum value has changed a revision of this calss is needed, possibly changing the detIdAndApv value from" << endl;
    error << "from uint32_t to uint64_t." << endl;
    edm::LogError("SiStripLatency::put") << error;
    throw cms::Exception("InsertFailure");
  }

  // Store all the values in the vectors
  uint32_t detIdAndApv = (detId << 3) | apv;
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

uint16_t SiStripLatency::latency(const uint32_t detId, const uint16_t apv) const
{
  const latConstIt & pos = position(detId, apv);
  if( pos == latencies_.end() ) {
    return 255;
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

pair<uint16_t, uint16_t> SiStripLatency::latencyAndMode(const uint32_t detId, const uint16_t apv) const
{
  const latConstIt & pos = position(detId, apv);
  if( pos == latencies_.end() ) {
    return make_pair(255, 0);
  }
  return make_pair(pos->latency, pos->mode);
}

uint16_t SiStripLatency::singleLatency() const
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
  return 255;
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

void SiStripLatency::allModes(vector<uint16_t> & allModesVector) const
{
  for( latConstIt it = latencies_.begin(); it != latencies_.end(); ++it ) {
    allModesVector.push_back(it->mode);
  }
  // The Latencies are sorted by DetIdAndApv, we need to sort the modes again and then remove duplicates
  sort( allModesVector.begin(), allModesVector.end() );
  allModesVector.erase( unique( allModesVector.begin(), allModesVector.end() ), allModesVector.end() );
}

void SiStripLatency::allLatencies(vector<uint16_t> & allLatenciesVector) const
{
//   if( !(latencies_.empty()) ) {
//     allLatenciesVector.push_back(latencies_[0].latency);
//     if( latencies_.size() > 1 ) {
//       for( latConstIt it = latencies_.begin()+1; it != latencies_.end(); ++it ) {
//         if( it->latency != (it-1)->latency) {
//           allLatenciesVector.push_back(it->latency);
//           cout << "Saved latency = " << short(it->latency) << endl;
//         }
//       }
//       // The Latencies are sorted by DetIdAndApv, we need to sort the latencies again
//       sort( allLatenciesVector.begin(), allLatenciesVector.end() );
//       allLatenciesVector.erase( unique( allLatenciesVector.begin(), allLatenciesVector.end() ) );
//     }
//   }

  for( latConstIt it = latencies_.begin(); it != latencies_.end(); ++it ) {
    allLatenciesVector.push_back(it->latency);
  }
  // The Latencies are sorted by DetIdAndApv, we need to sort the latencies again and then remove duplicates
  sort( allLatenciesVector.begin(), allLatenciesVector.end() );
  allLatenciesVector.erase( unique( allLatenciesVector.begin(), allLatenciesVector.end() ), allLatenciesVector.end() );
}

// pair<uint16_t, uint16_t> SiStripLatency::singleLatencyAndMode() const
// {
//   if( latencies_.size() == 1 ) {
//     return make_pair(latencies_[0].latency, latencies_[0].mode);
//   }
//   return make_pair(-1, 0);
// }

void SiStripLatency::printSummary(std::stringstream & ss) const
{
  uint16_t lat = singleLatency();
  uint16_t mode = singleMode();
  if( lat != 255 ) {
    ss << "All the Tracker has the same latency = " << lat << endl;
  }
  else {
    vector<uint16_t> allLatenciesVector;
    allLatencies(allLatenciesVector);
    if( allLatenciesVector.size() > 1 ) {
      ss << "There is more than one latency value in the Tracker" << endl;
    }
    else {
      ss << "Latency value is " << lat << " that means invalid" << endl;
    }
  }

  if( mode != 0 ) {
    ss << "All the Tracker has the same mode = " << mode << endl;
  }
  else {
    vector<uint16_t> allModesVector;
    allModes(allModesVector);
    if( allModesVector.size() > 1 ) {
      ss << "There is more than one mode in the Tracker" << endl;
    }
    else {
      ss << "Mode value is " << mode << " that means invalid" << endl;
    }
  }

  ss << "Total number of ranges = " << latencies_.size() << endl;
}

void SiStripLatency::printDebug(std::stringstream & ss) const
{
  ss << "List of all the latencies and modes for the " << latencies_.size() << " ranges in the object:" << endl;
  for( latConstIt it = latencies_.begin(); it != latencies_.end(); ++it ) {
    int detId = it->detIdAndApv >> 3;
    int apv = it->detIdAndApv & 7; // 7 is 0...0111
    ss << "for detId = " << detId << " and apv pair = " << apv << " latency = " << int(it->latency) << " and mode = " << int(it->mode) << endl;
  }
}
