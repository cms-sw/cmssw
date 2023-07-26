/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*   Maciej Wróbel (wroblisko@gmail.com)
*
****************************************************************************/

#include "FWCore/Utilities/interface/typelookup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/PPSObjects/interface/TotemDAQMapping.h"

using namespace std;

//----------------------------------------------------------------------------------------------------

std::ostream &operator<<(std::ostream &s, const TotemVFATInfo &vi) {
  s << vi.symbolicID << ", hw id=0x" << hex << vi.hwID << dec;

  return s;
}

//----------------------------------------------------------------------------------------------------

void TotemDAQMapping::insert(const TotemFramePosition &fp, const TotemVFATInfo &vi) {
  auto it = VFATMapping.find(fp);
  if (it != VFATMapping.end()) {
    edm::LogWarning("Totem") << "WARNING in DAQMapping::insert > Overwriting entry at " << fp << ". Previous: "
                             << "    " << VFATMapping[fp] << ","
                             << "  new: "
                             << "    " << vi << ". " << endl;
  }

  VFATMapping[fp] = vi;
}

//----------------------------------------------------------------------------------------------------
void TotemDAQMapping::insert(const TotemT2FramePosition &fp2, const TotemVFATInfo &vi) {
  const TotemFramePosition fp1(fp2.getRawPosition());
  auto it = VFATMapping.find(fp1);
  if (it != VFATMapping.end()) {
    edm::LogWarning("Totem") << "WARNING in DAQMapping::insert > Overwriting T2 entry at " << fp2 << ". Previous: "
                             << "    " << VFATMapping[fp1] << ","
                             << "  new: "
                             << "    " << vi << ". " << endl;
  }

  VFATMapping[fp1] = vi;
}

//----------------------------------------------------------------------------------------------------

const TotemDAQMapping::TotemTimingPlaneChannelPair TotemDAQMapping::getTimingChannel(const uint8_t hwId) const {
  TotemTimingPlaneChannelPair pair;
  auto iterator = totemTimingChannelMap.find(hwId);
  if (iterator != totemTimingChannelMap.end())
    pair = iterator->second;
  return pair;
}

//----------------------------------------------------------------------------------------------------

void TotemDAQMapping::print(std::ostream &os) const {
  os << "TotemDQMMapping VFAT mapping" << endl;
  for (const auto &p : VFATMapping) {
    os << "    " << p.first << " -> " << p.second << endl;
  }

  os << "TotemDQMMapping channel mapping" << endl;
  for (const auto &p : totemTimingChannelMap) {
    os << "    " << p.first << " plane " << p.second.plane << " channel " << p.second.channel << endl;
  }
}

std::ostream &operator<<(std::ostream &os, TotemDAQMapping mapping) {
  mapping.print(os);
  return os;
}

//----------------------------------------------------------------------------------------------------

TYPELOOKUP_DATA_REG(TotemDAQMapping);
