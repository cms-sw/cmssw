#include "EventFilter/HGCalRawToDigi/interface/UnpackerTools.h"
#include "DataFormats/HGCalDigi/interface/HGCalRawDataDefinitions.h"
#include <boost/crc.hpp>
#include <vector>
#include <algorithm>
#include <iostream>
//
//

bool hgcal::econdCRCAnalysis(const uint64_t *header, const uint32_t pos, const uint32_t payloadLength) {
  //there needs to be at least the CRC word otherwise it can't be checked
  if (payloadLength == 0)
    return false;

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
    return ((w << 24) & 0xFF000000) | ((w << 8) & 0x00FF0000) | ((w >> 8) & 0x0000FF00) |
           ((w >> 24) & 0x000000FF);  //swapping endianness
  });

  auto array = &(crcvec[0]);
  auto bytes = reinterpret_cast<const unsigned char *>(array);
  auto crc32 = boost::crc<32,
                          hgcal::ECOND_FRAME::CRC_POL,
                          hgcal::ECOND_FRAME::CRC_INITREM,
                          hgcal::ECOND_FRAME::CRC_FINALXOR,
                          false,
                          false>(bytes, (payloadLength - 1) * 4);  //32-bit words, hence need to parse 4 bytes

  return crc32 == target;
}
