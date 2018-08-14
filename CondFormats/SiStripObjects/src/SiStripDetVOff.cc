#include "CondFormats/SiStripObjects/interface/SiStripDetVOff.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/SiStripObjects/interface/SiStripDetSummary.h"

#include <algorithm>

void SiStripDetVOff::setBits( uint32_t & enDetId, const int HVoff, const int LVoff )
{
  if( LVoff != -1 ) {
    // LVonMask has all bits equal to 1 apart from the last one.
    if( LVoff == 0 ) enDetId &= LVonMask;
    if( LVoff == 1 ) enDetId |= LVmask;
  }
  if( HVoff != -1 ) {
    // HVonMask has all bits equal to 1 apart from the next to last one.
    if( HVoff == 0 ) enDetId &= HVonMask;
    if( HVoff == 1 ) enDetId |= HVmask;
  }
}

bool SiStripDetVOff::put(const uint32_t DetId, const int HVoff, const int LVoff)
{
  // Shift the DetId number of 2 bits to the left to have it in the final format with
  // the two additional bits used for HV and LV.
  uint32_t enDetId = (DetId << bitShift) & eightBitMask;

  // Binary search to determine if the element is already in the vector
  vOffIterator p = std::lower_bound(v_Voff.begin(), v_Voff.end(), enDetId);
  if( p != v_Voff.end() && (*p >> bitShift) == DetId) {
    // Found a matching entry, insert the HV and LV information.
    setBits(*p, HVoff, LVoff);
    // Check if the detector has all on, in that case remove it from the list.
    if( (~(*p) & allOnMask) == allOnMask ) v_Voff.erase(p);
  }
  else {
    // Not found, insert a new entry only if it is not all on
    setBits(enDetId, HVoff, LVoff);
    if( (~enDetId & allOnMask) != allOnMask ) v_Voff.insert(p, enDetId);
  }
  return true;
}

bool SiStripDetVOff::put(std::vector<uint32_t>& DetId, std::vector<int>& HVoff, std::vector<int>& LVoff)
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

void SiStripDetVOff::printDebug(std::stringstream & ss, const TrackerTopology* /*trackerTopo*/) const
{
  std::vector<uint32_t> detIds;
  getDetIds(detIds);
  constVoffIterator it = detIds.begin();
  ss << "DetId    \t HV \t LV" << std::endl;
  for( ; it!=detIds.end(); ++it ) {
    ss << *it << "\t";
    if( IsModuleHVOff(*it)) ss << "OFF\t";
    else ss << "ON \t";
    if( IsModuleLVOff(*it)) ss << "OFF" << std::endl;
    else ss << "ON" << std::endl;
  }
}

int SiStripDetVOff::getLVoffCounts() const
{
  std::vector<uint32_t> detIds;
  getDetIds(detIds);
  return std::count_if(std::begin(detIds), std::end(detIds),
      [this] ( uint32_t id ) -> bool { return IsModuleLVOff(id); });
}

int SiStripDetVOff::getHVoffCounts() const
{
  std::vector<uint32_t> detIds;
  getDetIds(detIds);
  return std::count_if(std::begin(detIds), std::end(detIds),
      [this] ( uint32_t id ) -> bool { return IsModuleHVOff(id); });
}

void SiStripDetVOff::printSummary(std::stringstream & ss, const TrackerTopology* trackerTopo) const
{
  SiStripDetSummary summaryHV{trackerTopo};
  SiStripDetSummary summaryLV{trackerTopo};
  std::vector<uint32_t> detIds;
  getDetIds(detIds);
  constVoffIterator it = detIds.begin();
  for( ; it!=detIds.end(); ++it ) {
    if( IsModuleHVOff(*it)) summaryHV.add(*it);
    if( IsModuleLVOff(*it)) summaryLV.add(*it);
  }
  ss << "Summary of detectors with HV off:" << std::endl;
  summaryHV.print(ss, false);
  ss << "Summary of detectors with LV off:" << std::endl;
  summaryLV.print(ss, false);
}
