#include "CondFormats/SiStripObjects/interface/SiStripDetVOff.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

bool SiStripDetVOff::put(const uint32_t DetId, const bool HVoff, const bool LVoff)
{
  // Shift the DetId number of 2 bits to the left to have it in the final format with
  // the two additional bits used for LV and HV.
  uint32_t enDetId = (DetId << bitShift) & eightBitMask;

  std::cout << "DetId = " << DetId << ", LVmask = " << LVmask << ", enDetId = " << enDetId << std::endl;

  // Binary search to determine if the element is already in the vector
  vOffIterator p = std::lower_bound(v_Voff.begin(), v_Voff.end(), enDetId);
  if( p != v_Voff.end() && (*p >> bitShift) == DetId) {
    // Found a matching entry, insert the HV and LV information
    if( LVoff ) *p |= LVmask;
    if( HVoff ) *p |= HVmask;
     std::cout << "Inside if: *p = " << *p << std::endl;
  }
  else {
    // Not found, insert a new entry
    if( LVoff ) enDetId |= LVmask;
    if( HVoff ) enDetId |= HVmask;
    std::cout << "Inside else: enDetId = " << enDetId << std::endl;
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
    constVboolIterator LVoffIt = LVoff.begin();
    constVboolIterator HVoffIt = HVoff.begin();
    for( ; detIdIt != detIdItEnd; ++detIdIt, ++LVoffIt, ++HVoffIt ) {
      put( *detIdIt, *LVoffIt, *HVoffIt );
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

bool SiStripDetVOff::IsModuleHVOff(const uint32_t DetId) const
{
  uint32_t enDetId = (DetId << bitShift) & eightBitMask;
  constVoffIterator p = std::lower_bound(v_Voff.begin(), v_Voff.end(), enDetId);
  if( p != v_Voff.end() && (*p >> bitShift) == DetId && (*p & LVmask) ) return true;
  return false;
}

bool SiStripDetVOff::IsModuleLVOff(const uint32_t DetId) const
{
  uint32_t enDetId = (DetId << bitShift) & eightBitMask;
  constVoffIterator p = std::lower_bound(v_Voff.begin(), v_Voff.end(), enDetId);
  if( p != v_Voff.end() && (*p >> bitShift) == DetId && (*p & HVmask) ) return true;
  return false;
}
