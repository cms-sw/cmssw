#ifndef DataFormats_GEMDigi_GEMVfatStatusDigi_H
#define DataFormats_GEMDigi_GEMVfatStatusDigi_H

#include <cstdint>

class GEMVfatStatusDigi {
public:
  explicit GEMVfatStatusDigi(uint64_t lsData,
                             uint64_t msData,
                             uint16_t crc,
                             uint16_t crc_calc,
                             uint8_t b1010,
                             uint8_t b1100,
                             uint8_t b1110,
                             uint8_t flag,
                             bool isBlockGood);

  GEMVfatStatusDigi() {}

  uint64_t getLsData() const { return lsData_; }
  uint64_t getMsData() const { return msData_; }
  uint16_t getCrc() const { return crc_; }
  uint16_t getCrc_calc() const { return crc_calc_; }
  uint16_t getIsBlocGood() const { return isBlockGood_; }
  uint8_t getB1010() const { return b1010_; }
  uint8_t getB1100() const { return b1100_; }
  uint8_t getB1110() const { return b1110_; }
  uint8_t getFlag() const { return flag_; }

private:
  uint64_t lsData_;    ///<channels from 1to64
  uint64_t msData_;    ///<channels from 65to128
  uint16_t crc_;       ///<Check Sum value, 16 bits
  uint16_t crc_calc_;  ///<Check Sum value recalculated, 16 bits
  uint8_t b1010_;      ///<1010:4 Control bits, shoud be 1010
  uint8_t b1100_;      ///<1100:4, Control bits, shoud be 1100
  uint8_t b1110_;      ///<1110:4 Control bits, shoud be 1110
  uint8_t flag_;       ///<Control Flags: 4 bits, Hamming Error/AFULL/SEUlogic/SUEI2C
  bool isBlockGood_;   ///<Shows if block is good (control bits, chip ID and CRC checks)
};
#endif
