/****************************************************************************
*
* This is a part of the TOTEM testbeam/monitoring software.
* This is a part of the TOTEM offline software.
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
*  Leszek Grzanka
*
****************************************************************************/

#include "EventFilter/CTPPSRawToDigi/interface/VFATFrame.h"

#include <cstdio>
#include <cstring>

VFATFrame::VFATFrame(const VFATFrame::word *_data)
    : presenceFlags(15),   // by default BC, EC, ID and CRC are present
      daqErrorFlags(0),    // by default, no DAQ error
      numberOfClusters(0)  // no clusters by default
{
  if (_data)
    setData(_data);
  else
    memset(data, 0, 12 * sizeof(word));
}

void VFATFrame::setData(const VFATFrame::word *_data) { memcpy(data, _data, 24); }

std::vector<unsigned char> VFATFrame::getActiveChannels() const {
  std::vector<unsigned char> channels;

  for (int i = 0; i < 8; i++) {
    // quick check
    if (!data[1 + i])
      continue;

    // go throug bits
    word mask;
    char offset;
    for (mask = 1 << 15, offset = 15; mask; mask >>= 1, offset--) {
      if (data[1 + i] & mask)
        channels.push_back(i * 16 + offset);
    }
  }

  return channels;
}

bool VFATFrame::checkFootprint() const {
  if (isIDPresent() && (data[9] & 0xF000) != 0xE000)
    return false;

  if (isECPresent() && (data[10] & 0xF000) != 0xC000)
    return false;

  if (isBCPresent() && (data[11] & 0xF000) != 0xA000)
    return false;

  return true;
}

bool VFATFrame::checkFootprintT2() const {
  if (isIDPresent() && (data[2] & 0xF000) != 0xE000)
    return false;

  if (isECPresent() && (data[1] & 0xF000) != 0xC000)
    return false;

  if (isBCPresent() && (data[0] & 0xF000) != 0xA000)
    return false;

  return true;
}

bool VFATFrame::checkCRC() const {
  // check DAQ error flags
  if (daqErrorFlags != 0)
    return false;

  // return true if CRC not present
  if (!isCRCPresent())
    return true;

  // compare CRC
  word crc_fin = 0xffff;

  for (int i = 11; i >= 1; i--)
    crc_fin = calculateCRC(crc_fin, data[i]);

  return (crc_fin == data[0]);
}

bool VFATFrame::checkCRCT2() const {
  // return true if CRC not present
  if (!isCRCPresent())
    return true;

  // compare CRC
  word crc_fin = 0xffff;

  for (int i = 0; i < 11; i++)
    crc_fin = calculateCRC(crc_fin, data[i]);

  return (crc_fin == data[11]);
}

VFATFrame::word VFATFrame::calculateCRC(VFATFrame::word crc_in, VFATFrame::word dato) {
  word v = 0x0001;
  word mask = 0x0001;
  bool d = false;
  word crc_temp = crc_in;
  unsigned char datalen = 16;

  for (int i = 0; i < datalen; i++) {
    if (dato & v)
      d = true;
    else
      d = false;

    if ((crc_temp & mask) ^ d)
      crc_temp = crc_temp >> 1 ^ 0x8408;
    else
      crc_temp = crc_temp >> 1;

    v <<= 1;
  }

  return crc_temp;
}

void VFATFrame::Print(bool binary) const {
  if (binary) {
    for (int i = 0; i < 12; i++) {
      const word &w = data[11 - i];
      word mask = (1 << 15);
      for (int j = 0; j < 16; j++) {
        if (w & mask)
          printf("1");
        else
          printf("0");
        mask = (mask >> 1);
        if ((j + 1) % 4 == 0)
          printf("|");
      }
      printf("\n");
    }
  } else {
    printf("ID = %03x, BC = %04u, EC = %03u, flags = %2u, CRC = %04x ",
           getChipID(),
           getBC(),
           getEC(),
           getFlags(),
           getCRC());

    if (checkCRC())
      printf("(  OK), footprint ");
    else
      printf("(FAIL), footprint ");

    if (checkFootprint())
      printf("  OK");
    else
      printf("FAIL");

    printf(", frame = %04x|%04x|%04x|", data[11], data[10], data[9]);
    for (int i = 8; i > 0; i--)
      printf("%04x", data[i]);
    printf("|%04x", data[0]);

    printf(", presFl=%x", presenceFlags);
    printf(", daqErrFl=%x", daqErrorFlags);

    printf("\n");
  }
}

void VFATFrame::PrintT2(bool binary) const {
  if (binary) {
    for (int i = 0; i < 12; i++) {
      const word &w = data[i];
      word mask = (1 << 15);
      for (int j = 0; j < 16; j++) {
        if (w & mask)
          printf("1");
        else
          printf("0");
        mask = (mask >> 1);
        if ((j + 1) % 4 == 0)
          printf("|");
      }
      printf("\n");
    }
  } else {
    // print right CRC
    word crc_fin = 0xffff;

    for (int i = 0; i < 11; i++)
      crc_fin = calculateCRC(crc_fin, data[i]);

    printf("CRC = %04x ", getCRCT2());

    if (checkCRCT2())
      printf("(  OK), footprint ");
    else
      printf("(FAIL, right = %04x), footprint ", crc_fin);

    if (checkFootprintT2())
      printf("  OK");
    else
      printf("FAIL");

    printf("Frame = %04x|%04x|%04x|", data[0], data[1], data[2]);
    for (int i = 3; i < 11; i++)
      printf("%04x", data[i]);
    printf("|%04x", data[11]);

    printf("\n");
  }
}
