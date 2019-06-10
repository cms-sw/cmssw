#include "CondFormats/SiStripObjects/interface/Phase2TrackerModule.h"
#include <iostream>
#include <sstream>

std::string Phase2TrackerModule::description(bool compact) const {
  std::stringstream ss;
  if (compact) {
    ss << "0x" << std::hex << getDetid() << " (" << (getModuleType() == Phase2TrackerModule::SS ? "2S" : "PS") << ") ";
    ss << "GBT 0x" << std::hex << getGbtid() << " FED " << std::dec << getCh().first << "." << getCh().second;
    ss << " C " << getCoolingLoop() << " P " << getPowerGroup() << std::endl;
  } else {
    ss << "Module of type " << (getModuleType() == Phase2TrackerModule::SS ? "2S" : "PS") << ":" << std::endl;
    ss << "  Detid: 0x" << std::hex << getDetid() << " GBTid: 0x" << getGbtid() << std::endl;
    ss << "  FED connection: " << std::dec << getCh().first << "." << getCh().second << std::endl;
    ss << "  Cooling loop: " << getCoolingLoop() << std::endl;
    ss << "  Power group: " << getPowerGroup() << std::endl;
  }
  return ss.str();
}
