#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>
#include <iterator>
#include <iostream>
#include <sstream>

bool SiStripLatency::put( const uint32_t detId, const uint16_t apv, const uint16_t latency, const uint16_t mode )
{
  if( detId > 536870911 ) {
    std::stringstream error;
    error << "ERROR: the detId = " << detId << " is bigger than the maximum acceptable value = 2^(29) - 1 = " << 536870911 << std::endl;
    error << "Since we are using 29 bits for the detId and 3 bits for the apv value. The maximum tracker detId at the moment" << std::endl;
    error << "of the writing of this class was 47017836 as defined in CalibTracker/SiStripCommon/data/SiStripDetInfo.dat." << std::endl;
    error << "If the maximum value has changed a revision of this calss is needed, possibly changing the detIdAndApv value from" << std::endl;
    error << "from uint32_t to uint64_t." << std::endl;
    edm::LogError("SiStripLatency::put") << error.str();
    throw cms::Exception("InsertFailure");
  }

  // Store all the values in the vectors
  uint32_t detIdAndApv = (detId << 3) | apv;
  latIt pos = lower_bound(latencies_.begin(), latencies_.end(), detIdAndApv, OrderByDetIdAndApv());

  if( pos != latencies_.end() && pos->detIdAndApv == detIdAndApv ) {
    std::cout << "Value already inserted, skipping insertion" << std::endl;
    return false;
  }
  // std::cout << "Filling with: latency = " << latency << ", mode = " << mode << std::endl;
  latencies_.insert(pos, Latency(detIdAndApv, latency, mode));

  return true;
}

void SiStripLatency::compress()
{
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
}

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

std::pair<uint16_t, uint16_t> SiStripLatency::latencyAndMode(const uint32_t detId, const uint16_t apv) const
{
  const latConstIt & pos = position(detId, apv);
  if( pos == latencies_.end() ) {
    return std::make_pair(255, 0);
  }
  return std::make_pair(pos->latency, pos->mode);
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

void SiStripLatency::allModes(std::vector<uint16_t> & allModesVector) const
{
  for( latConstIt it = latencies_.begin(); it != latencies_.end(); ++it ) {
    allModesVector.push_back(it->mode);
  }
  // The Latencies are sorted by DetIdAndApv, we need to sort the modes again and then remove duplicates
  sort( allModesVector.begin(), allModesVector.end() );
  allModesVector.erase( unique( allModesVector.begin(), allModesVector.end() ), allModesVector.end() );
}

int16_t SiStripLatency::singleReadOutMode() const
{
  uint16_t mode = singleMode();
  if(mode != 0 ) {
    if( (mode & READMODEMASK) == READMODEMASK ) return 1;
    if( (mode & READMODEMASK) == 0 ) return 0;
  }
  else {
    // If we are here the Tracker is not in single mode. Check if it is in single Read-out mode.
    bool allInPeakMode = true;
    bool allInDecoMode = true;
    std::vector<uint16_t> allModesVector;
    allModes(allModesVector);
    std::vector<uint16_t>::const_iterator it = allModesVector.begin();
    if (allModesVector.size() == 1 && allModesVector[0] == 0) allInPeakMode = false;
    else{
      for( ; it != allModesVector.end(); ++it ) {
	if( (*it) % 2 == 0 ) continue;
	if( ((*it) & READMODEMASK) == READMODEMASK ) allInDecoMode = false;
	if( ((*it) & READMODEMASK) == 0 ) allInPeakMode = false;
      }
    }
    if( allInPeakMode ) return 1;
    if( allInDecoMode ) return 0;
  }
  return -1;
}


void SiStripLatency::allLatencies(std::vector<uint16_t> & allLatenciesVector) const
{

  for( latConstIt it = latencies_.begin(); it != latencies_.end(); ++it ) {
    allLatenciesVector.push_back(it->latency);
  }
  // The Latencies are sorted by DetIdAndApv, we need to sort the latencies again and then remove duplicates
  sort( allLatenciesVector.begin(), allLatenciesVector.end() );
  allLatenciesVector.erase( unique( allLatenciesVector.begin(), allLatenciesVector.end() ), allLatenciesVector.end() );
}


std::vector<SiStripLatency::Latency> SiStripLatency::allUniqueLatencyAndModes()
{
  std::vector<Latency> latencyCopy(latencies_);
  sort( latencyCopy.begin(), latencyCopy.end(), OrderByLatencyAndMode() );
  latencyCopy.erase( unique( latencyCopy.begin(), latencyCopy.end(), SiStripLatency::EqualByLatencyAndMode() ), latencyCopy.end() );
  return latencyCopy;
}

void SiStripLatency::printSummary(std::stringstream & ss, const TrackerTopology* trackerTopo) const
{
  ss << std::endl;
  if(singleReadOutMode()==1){
     ss << "SingleReadOut = PEAK" << std::endl;
  }else if(singleReadOutMode()==0){
    ss << "SingleReadOut = DECO" << std::endl;
  }else{
    ss << "SingleReadOut = MIXED" << std::endl;
  }
  uint16_t lat = singleLatency();
  if( lat != 255 ) {
    ss << "All the Tracker has the same latency = " << lat << std::endl;
  }
  else {
    std::vector<uint16_t> allLatenciesVector;
    allLatencies(allLatenciesVector);
    if( allLatenciesVector.size() > 1 ) {
      ss << "There is more than one latency value in the Tracker" << std::endl;
    }
    else {
      ss << "Latency value is " << lat << " that means invalid" << std::endl;
    }
  }
  ss << "Total number of ranges = " << latencies_.size() << std::endl;
  printDebug(ss, trackerTopo);
}


void SiStripLatency::printDebug(std::stringstream & ss, const TrackerTopology* /*trackerTopo*/) const
{
  ss << "List of all the latencies and modes for the " << latencies_.size() << " ranges in the object:" << std::endl;
  for( latConstIt it = latencies_.begin(); it != latencies_.end(); ++it ) {
    int detId = it->detIdAndApv >> 3;
    int apv = it->detIdAndApv & 7; // 7 is 0...0111
    ss << "for detId = " << detId << " and apv pair = " << apv << " latency = " << int(it->latency) << " and mode = " << int(it->mode) << std::endl;
  }
}

#undef READMODEMASK
