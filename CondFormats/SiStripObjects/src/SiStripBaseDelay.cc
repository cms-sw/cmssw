#include "CondFormats/SiStripObjects/interface/SiStripBaseDelay.h"

#include <algorithm>
#include <iterator>
#include <iostream>
#include <sstream>

#include <boost/bind.hpp>

bool SiStripBaseDelay::put( const uint32_t detId, const uint16_t coarseDelay, const uint16_t fineDelay )
{
  delays_.push_back(Delay(detId, coarseDelay, fineDelay));
  return true;
}

uint16_t SiStripBaseDelay::coarseDelay(const uint32_t detId)
{
  delayConstIt it = std::find_if(delays_.begin(), delays_.end(), boost::bind(&Delay::detId, _1) == detId);
  if( it != delays_.end() ) {
    return it->coarseDelay;
  }
  return 0;
}

uint16_t SiStripBaseDelay::fineDelay(const uint32_t detId) const
{
  delayConstIt it = std::find_if(delays_.begin(), delays_.end(), boost::bind(&Delay::detId, _1) == detId);
  if( it != delays_.end() ) {
    return it->fineDelay;
  }
  return 0;
}

double SiStripBaseDelay::delay(const uint32_t detId) const
{
  delayConstIt it = std::find_if(delays_.begin(), delays_.end(), boost::bind(&Delay::detId, _1) == detId);
  if( it != delays_.end() ) {
    return makeDelay(it->coarseDelay, it->fineDelay);
  }
  return 0;
}

void SiStripBaseDelay::detIds(std::vector<uint32_t> & detIdVector) const
{
  std::vector<Delay>::const_iterator it = delays_.begin();
  for( ; it != delays_.end(); ++it ) {
    detIdVector.push_back(it->detId);
  }
}

void SiStripBaseDelay::printSummary(std::stringstream & ss, const TrackerTopology* trackerTopo) const
{
  ss << "Total number of delays = " << delays_.size() << std::endl;
  SiStripDetSummary summaryDelays{trackerTopo};
  delayConstIt it = delays_.begin();
  for( ; it != delays_.end(); ++it ) {
    summaryDelays.add(it->detId, makeDelay(it->coarseDelay, it->fineDelay));
  }
  ss << std::endl << "Summary:" << std::endl;
  summaryDelays.print(ss);
}

void SiStripBaseDelay::printDebug(std::stringstream & ss, const TrackerTopology* trackerTopo) const
{
  printSummary(ss, trackerTopo);
  delayConstIt it = delays_.begin();
  ss << std::endl << "All pedestal values:" << std::endl;
  for( ; it != delays_.end(); ++it ) {
    ss << "detId = " << it->detId << " delay = " << makeDelay(it->coarseDelay, it->fineDelay) << std::endl;
  }
}
