#include "DataFormats/HGCalDigi/interface/HGCalElectronicsId.h"
#include "EventFilter/HGCalRawToDigi/interface/HGCalUnpacker.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

uint16_t enabledERXMapping(uint16_t sLink, uint8_t captureBlock, uint8_t econd) {
  if (sLink == 0 && captureBlock == 0 && econd == 3) {
    return 0b1;
  }
  return 0b11;
}

HGCalElectronicsId logicalMapping(HGCalElectronicsId elecID) { return elecID; }

int main() {
  uint32_t testInput[80] = {
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0xffffffff, 0xfffff000, 0xaa0250bf, 0x00000000, 0xfdfffffc,
      0x00000007, 0x0fffff1f, 0xfc2fffff, 0x3fffff7f, 0xffffffff, 0xffffff00, 0xffffffff, 0x00000000, 0x00000000,
      0xaa0010ff, 0x00000000, 0xaa001f7f, 0x00000000, 0xaa0a30bf, 0x00000000, 0xfdffffff, 0xffffffff, 0xffffffff,
      0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
      0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
      0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
      0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
      0x00000000, 0xffffffff, 0xffffffd3, 0xaa0050bf, 0x00000000, 0xffffffff, 0x00000000, 0xaa0090bf, 0x00000000,
      0xffffffff, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};

  HGCalUnpackerConfig config;
  config.sLinkCaptureBlockMax = 2;
  HGCalUnpacker<HGCalElectronicsId> unpacker(config);
  unpacker.parseSLink(testInput, 80, enabledERXMapping, logicalMapping);

  auto channeldata = unpacker.getChannelData();
  auto cms = unpacker.getCommonModeIndex();
  for (unsigned int i = 0; i < channeldata.size(); i++) {
    auto data = channeldata.at(i);
    auto cm = cms.at(i);
    auto id = data.id();
    auto idraw = id.raw();
    auto raw = data.raw();
    std::cout << "id=" << idraw << ", raw=" << raw << ", common mode index=" << cm << std::endl;
  }
  auto badECONDs = unpacker.getBadECOND();
  for (auto badECOND : badECONDs) {
    std::cout << std::dec << badECOND << std::endl;
  }
  return 0;
}
