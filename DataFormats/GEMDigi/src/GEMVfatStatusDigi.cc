#include "DataFormats/GEMDigi/interface/GEMVfatStatusDigi.h"

GEMVfatStatusDigi::GEMVfatStatusDigi(uint64_t lsData,
                                     uint64_t msData,
                                     uint16_t crc,
                                     uint16_t crc_calc,
                                     uint8_t b1010,
                                     uint8_t b1100,
                                     uint8_t b1110,
                                     uint8_t flag,
                                     bool isBlockGood)
    : lsData_(lsData),
      msData_(msData),
      crc_(crc),
      crc_calc_(crc_calc),
      b1010_(b1010),
      b1100_(b1100),
      b1110_(b1110),
      flag_(flag),
      isBlockGood_(isBlockGood){};
