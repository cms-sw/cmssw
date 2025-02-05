#include "EventFilter/HGCalRawToDigi/interface/Unpackertools.h"
#include <boost/crc.hpp>
#include <vector>
#include <iostream>

//
//

bool hgcal::econdCRCAnalysis(const uint64_t *header, uint32_t pos, uint32_t payloadLength, uint64_t crcpol) {
  int index = 0;
  std::vector<uint32_t> data32b;  //  reading 32-bit words, input is 64b

  while (data32b.size() < payloadLength + 2) {
    uint64_t word = *(header + pos + index);  // move to ECOND header
    data32b.push_back(word >> 32);            // Reading first 32b

    if (data32b.size() == payloadLength + 2)
      break;
    data32b.push_back(word & 0xFFFFFFFF);  //read second 32b
    ++index;
  }

  auto target = data32b.back();  // CRC in ECOND trailer

  //Compute CRC using all eRx subpackets but not the event paket header (two first words)
  std::vector<uint32_t> crcvec(data32b.begin() + 2, data32b.end() - 1);
  std::transform(crcvec.begin(), crcvec.end(), crcvec.begin(), [](uint32_t w) {
    return ((w << 24) & 0xFF000000) | ((w << 8) & 0x00FF0000) | ((w >> 8) & 0x0000FF00) | ((w >> 24) & 0x000000FF);
  });

  auto array = &(crcvec[0]);
  auto bytes = reinterpret_cast<const unsigned char *>(array);
  auto crc32 = boost::crc<32, 0x4c11db7, 0x0, 0x0, false, false>(bytes, (payloadLength - 1) * 4);
  //LogDebug("[HGCalUnpacker]") << "crc32 = " << crc32 << " target = " << target;
  return crc32 == target;
}
