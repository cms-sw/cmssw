#include "DataFormats/ForwardDetId/interface/MTDDetId.h"
#include <iomanip>
#include <bitset>

std::ostream& operator<<(std::ostream& os, const MTDDetId& id) {
  return os << "[MTDDetId::print] "
            << " " << std::bitset<4>((id.rawId() >> 28) & 0xF).to_string() << " "
            << std::bitset<4>((id.rawId() >> 24) & 0xF).to_string() << " "
            << std::bitset<4>((id.rawId() >> 20) & 0xF).to_string() << " "
            << std::bitset<4>((id.rawId() >> 16) & 0xF).to_string() << " "
            << std::bitset<4>((id.rawId() >> 12) & 0xF).to_string() << " "
            << std::bitset<4>((id.rawId() >> 8) & 0xF).to_string() << " "
            << std::bitset<4>((id.rawId() >> 4) & 0xF).to_string() << " "
            << std::bitset<4>(id.rawId() & 0xF).to_string() << std::endl
            << " rawId       : 0x" << std::hex << std::setfill('0') << std::setw(8) << id.rawId() << std::dec
            << std::endl
            << " bits[0:24]  : " << std::hex << std::setfill('0') << std::setw(8) << (0x01FFFFFF & id.rawId())
            << std::dec << std::endl
            << " Detector        : " << id.det() << std::endl
            << " SubDetector     : " << id.subdetId() << std::endl
            << " MTD subdetector : " << id.mtdSubDetector() << std::endl;
}
