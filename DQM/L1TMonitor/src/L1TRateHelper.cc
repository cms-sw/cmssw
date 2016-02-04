#include "DQM/L1TMonitor/interface/L1TRateHelper.h"


namespace L1TRateHelper {


std::pair< int, int> L1TRateHelper::removeAndGetRateForEarliestTime(){ 
  if (m_rateMap.begin() == m_rateMap.end() ) 
    return std::make_pair(-1,-1);
  
  if (m_timeStart==-1) {
    m_timeStart = m_rateMap.begin()->second.getTime()-1; // so time will start from 1
  }
  int r1 = m_rateMap.begin()->second.getTime()-m_timeStart; 
  int r2 = m_rateMap.begin()->second.m_events;
  m_lastRemovedOrbit = m_rateMap.begin()->second.m_orbitHigh;
  m_rateMap.erase(m_rateMap.begin());
  return std::make_pair(r1,r2);
}


}


