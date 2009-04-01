#include "CondFormats/SiStripObjects/interface/SiStripDetVOff.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/SiStripObjects/interface/SiStripDetSummary.h"

bool SiStripDetVOff::put(const uint32_t DetId, const bool HVoff, const bool LVoff)
{
  // Shift the DetId number of 2 bits to the left to have it in the final format with
  // the two additional bits used for HV and LV.
  uint32_t enDetId = (DetId << bitShift) & eightBitMask;

  // Binary search to determine if the element is already in the vector
  vOffIterator p = std::lower_bound(v_Voff.begin(), v_Voff.end(), enDetId);
  if( p != v_Voff.end() && (*p >> bitShift) == DetId) {
    // Found a matching entry, insert the HV and LV information
    if( HVoff ) *p |= HVmask;
    if( LVoff ) *p |= LVmask;
  }
  else {
    // Not found, insert a new entry
    if( HVoff ) enDetId |= HVmask;
    if( LVoff ) enDetId |= LVmask;
    v_Voff.insert(p, enDetId);
    // The vector is already sorted
    // // Sort the vector (necessary for the next binary search to work)
    // // std::sort(v_Voff.begin(), v_Voff.end());
  }
  return true;
}

bool SiStripDetVOff::put(std::vector<uint32_t>& DetId, std::vector<bool>& HVoff, std::vector<bool>& LVoff)
{
  if( DetId.size() == HVoff.size() && DetId.size() == LVoff.size() ) {
    constVoffIterator detIdIt = DetId.begin();
    constVoffIterator detIdItEnd = DetId.end();
    constVboolIterator HVoffIt = HVoff.begin();
    constVboolIterator LVoffIt = LVoff.begin();
    for( ; detIdIt != detIdItEnd; ++detIdIt, ++HVoffIt, ++LVoffIt ) {
      put( *detIdIt, *HVoffIt, *LVoffIt );
    }
  }
  else {
    std::cout << "Error: inconsistent sizes of vectors:" << std::endl;
    std::cout << "DetId size = " << DetId.size() << ", HVoff size = " << HVoff.size() << ", LVoff size = " << LVoff.size() << std::endl;
    return false;
  }
  return true;
}

void SiStripDetVOff::getDetIds(std::vector<uint32_t>& DetIds_) const
{
  // returns vector of DetIds in map
  DetIds_.clear();
  // Extract the detId from the bitSet and fill the vector
  constVoffIterator bitSetIt = v_Voff.begin();
  constVoffIterator bitSetItEnd = v_Voff.end();
  for( ; bitSetIt != bitSetItEnd; ++bitSetIt ) {
    std::cout << "DetId = " << (*bitSetIt >> bitShift) << std::endl;
    DetIds_.push_back( (*bitSetIt) >> bitShift );
  }
}

bool SiStripDetVOff::IsModuleVOff(const uint32_t DetId) const
{
  uint32_t enDetId = (DetId << bitShift) & eightBitMask;
  constVoffIterator p = std::lower_bound(v_Voff.begin(), v_Voff.end(), enDetId);
  if( p != v_Voff.end() && (*p >> bitShift) == DetId) return true;
  return false;
}

bool SiStripDetVOff::IsModuleLVOff(const uint32_t DetId) const
{
  uint32_t enDetId = (DetId << bitShift) & eightBitMask;
  constVoffIterator p = std::lower_bound(v_Voff.begin(), v_Voff.end(), enDetId);
  if( p != v_Voff.end() && (*p >> bitShift) == DetId && (*p & LVmask) ) return true;
  return false;
}

bool SiStripDetVOff::IsModuleHVOff(const uint32_t DetId) const
{
  uint32_t enDetId = (DetId << bitShift) & eightBitMask;
  constVoffIterator p = std::lower_bound(v_Voff.begin(), v_Voff.end(), enDetId);
  if( p != v_Voff.end() && (*p >> bitShift) == DetId && (*p & HVmask) ) return true;
  return false;
}

void SiStripDetVOff::printDebug(std::stringstream & ss) const
{
  std::vector<uint32_t> detIds;
  getDetIds(detIds);
  constVoffIterator it = detIds.begin();
  ss << "DetId    \t HV \t LV" << endl;
  for( ; it!=detIds.end(); ++it ) {
    ss << *it << "\t";
    if( IsModuleHVOff(*it)) ss << "OFF\t";
    else ss << "ON \t";
    if( IsModuleLVOff(*it)) ss << "OFF" << endl;
    else ss << "ON" << endl;
  }
}

void SiStripDetVOff::printSummary(std::stringstream & ss) const
{
  SiStripDetSummary summaryHV;
  SiStripDetSummary summaryLV;
  std::vector<uint32_t> detIds;
  getDetIds(detIds);
  constVoffIterator it = detIds.begin();
  for( ; it!=detIds.end(); ++it ) {
    if( IsModuleHVOff(*it)) summaryHV.add(*it);
    if( IsModuleLVOff(*it)) summaryLV.add(*it);
  }
  ss << "Summary of detectors with HV off:" << endl;
  summaryHV.print(ss, false);
  ss << "Summary of detectors with LV off:" << endl;
  summaryLV.print(ss, false);
}
