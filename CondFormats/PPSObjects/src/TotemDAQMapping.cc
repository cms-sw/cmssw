/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*   Maciej Wróbel (wroblisko@gmail.com)
*
****************************************************************************/

#include "CondFormats/PPSObjects/interface/TotemDAQMapping.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/typelookup.h"

using namespace std;

void TotemVFATInfo::print(ostream &os, string subSystemName) const {
  symbolicID.print(os, subSystemName);
  if (subSystemName != "TotemTiming") {
    os << ", hw id=0x" << hex << hwID << dec;
  }
  os << endl;
}

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

void TotemDAQMapping::print(std::ostream &os, std::string subSystemName) const {
  os << "TotemDAQMapping VFAT mapping" << endl;
  for (auto &p : VFATMapping) {
    os << "    " << p.first << " -> ";
    p.second.print(os, subSystemName);
  }

  if (subSystemName == "TotemTiming" || subSystemName.empty()) {
    os << "TotemDAQMapping channel mapping" << endl;
    for (const auto &p : totemTimingChannelMap) {
      os << "    "
         << " hw id=0x" << hex << (int)p.first << dec << " plane=" << p.second.plane << " channel=" << p.second.channel
         << endl;
    }
  }
}

std::ostream &operator<<(std::ostream &os, TotemDAQMapping mapping) {
  mapping.print(os, "");
  return os;
}

//----------------------------------------------------------------------------------------------------

TYPELOOKUP_DATA_REG(TotemDAQMapping);
