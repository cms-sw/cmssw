#ifndef IOPool_Streamer_FRDFileHeader_h
#define IOPool_Streamer_FRDFileHeader_h

#include <array>
#include <cstddef>
#include <cstdint>

/*
 * FRD File Header optionally found at the beginning of the FRD RAW file
 *
 * Version 1 Format:
 *   uint8_t [4] - id contanining 4 characters: 0x52, 0x41, 0x57, 0x5f  "RAW_"
 *   uint8_t [4] - version string 4 characters: 0x30, 0x30, 0x30, 0x31  "0001"
 *   uint16_t - header size: 24
 *   uint16_t - number of events in the RAW file
 *   uint32_t - lumisection
 *   uint64_t - total size of the raw file (including header)
 *
 * */

constexpr std::array<unsigned char, 4> FRDFileHeader_id{{0x52, 0x41, 0x57, 0x5f}};
constexpr std::array<unsigned char, 4> FRDFileVersion_1{{0x30, 0x30, 0x30, 0x31}};

struct FRDFileHeader_v1 {
  FRDFileHeader_v1() = default;

  FRDFileHeader_v1(uint16_t eventCount, uint32_t lumiSection, uint64_t fileSize)
      : id_{FRDFileHeader_id},
        version_{FRDFileVersion_1},
        headerSize_(sizeof(FRDFileHeader_v1)),
        eventCount_(eventCount),
        lumiSection_(lumiSection),
        fileSize_(fileSize) {}

  std::array<uint8_t, 4> id_;
  std::array<uint8_t, 4> version_;
  uint16_t headerSize_;
  uint16_t eventCount_;
  uint32_t lumiSection_;
  uint64_t fileSize_;
};

inline uint16_t getFRDFileHeaderVersion(const std::array<uint8_t, 4>& id, const std::array<uint8_t, 4>& version) {
  size_t i;
  for (i = 0; i < 4; i++)
    if (id[i] != FRDFileHeader_id[i])
      return 0;  //not FRD file header
  uint16_t ret = 0;
  for (i = 0; i < 4; i++) {
    if (version[i] > '9' || version[i] < '0')
      return 0;  //NaN sequence
    ret = ret * 10 + (uint16_t)(version[i] - '0');
  }
  return ret;
}

#endif
