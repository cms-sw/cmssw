#include "DataFormats/ForwardDetId/interface/MTDDetId.h"
#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"
#include "DataFormats/DetId/interface/DetId.h"

#include <iostream>
#include <iomanip>
#include <bitset>

int main() {
  DetId id;
  BTLDetId btlid;
  ETLDetId etlid;

  auto printBits = [&](const uint32_t& id) {
    std::stringstream ss;
    ss << std::bitset<4>((id >> 28) & 0xF).to_string() << " " << std::bitset<4>((id >> 24) & 0xF).to_string() << " "
       << std::bitset<4>((id >> 20) & 0xF).to_string() << " " << std::bitset<4>((id >> 16) & 0xF).to_string() << " "
       << std::bitset<4>((id >> 12) & 0xF).to_string() << " " << std::bitset<4>((id >> 8) & 0xF).to_string() << " "
       << std::bitset<4>((id >> 4) & 0xF).to_string() << " " << std::bitset<4>(id & 0xF).to_string();
    return ss.str();
  };

  std::cout << "DetId    " << std::setw(20) << id.rawId() << std::setw(40) << printBits(id.rawId()) << " mask "
            << std::setw(40) << printBits(MTDDetId::kMTDMask) << " isMTD " << MTDDetId::testForMTD(id) << " isBTL "
            << MTDDetId::testForBTL(id) << " isETL " << MTDDetId::testForETL(id) << std::endl;
  std::cout << "BTLDetId " << std::setw(20) << btlid.rawId() << std::setw(40) << printBits(btlid.rawId()) << " mask "
            << std::setw(40) << printBits(MTDDetId::kBTLMask) << " isMTD " << MTDDetId::testForMTD(btlid) << " isBTL "
            << MTDDetId::testForBTL(btlid) << " isETL " << MTDDetId::testForETL(btlid) << std::endl;
  std::cout << "ETLDetId " << std::setw(20) << etlid.rawId() << std::setw(40) << printBits(etlid.rawId()) << " mask "
            << std::setw(40) << printBits(MTDDetId::kETLMask) << " isMTD " << MTDDetId::testForMTD(etlid) << " isBTL "
            << MTDDetId::testForBTL(etlid) << " isETL " << MTDDetId::testForETL(etlid) << std::endl;

  return 0;
}
