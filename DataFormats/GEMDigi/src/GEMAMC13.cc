#include <cstdint>
#include "DataFormats/GEMDigi/interface/GEMAMC13.h"

void GEMAMC13::setCDFHeader(uint8_t Evt_ty, uint32_t LV1_id, uint16_t BX_id, uint16_t Source_id) {
  CDFHeader u{0};
  u.cb5 = 0x5;
  u.eventType = Evt_ty;
  u.lv1Id = LV1_id;
  u.bxId = BX_id;
  u.sourceId = Source_id;
  cdfh_ = u.word;
}

void GEMAMC13::setAMC13Header(uint8_t CalTyp, uint8_t nAMC, uint32_t OrN) {
  AMC13Header u{0};
  u.cb0 = 0x0;
  u.calType = CalTyp;
  u.nAMC = nAMC;
  u.orbitN = OrN;
  amc13h_ = u.word;
}

void GEMAMC13::setAMC13Trailer(uint8_t Blk_NoT, uint8_t LV1_idT, uint16_t BX_idT) {
  AMC13Trailer u{0};
  u.blkN = Blk_NoT;
  u.lv1IdT = LV1_idT;
  u.bxIdT = BX_idT;
  amc13t_ = u.word;
}

void GEMAMC13::setCDFTrailer(uint32_t EvtLength) {
  CDFTrailer u{0};
  u.cbA = 0xA;
  u.eventType = CDFHeader{cdfh_}.eventType;
  u.evtLength = EvtLength;
  cdft_ = u.word;
}

void GEMAMC13::addAMCheader(uint64_t word) { amcHeaders_.push_back(word); }

void GEMAMC13::addAMCheader(uint32_t AMC_size, uint8_t Blk_No, uint8_t AMC_No, uint16_t BoardID) {
  // AMC Header word
  // 55 - 32  | 27 - 20 | 19 - 16 | 15 - 0  |
  // AMC_size | Blk_No  | AMC_No  | BoardID |
  uint64_t word = (static_cast<uint64_t>(AMC_size & 0x00ffffff) << 32) | (static_cast<uint64_t>(Blk_No & 0xff) << 20) |
                  (static_cast<uint64_t>(AMC_No & 0x0f) << 16) | (static_cast<uint64_t>(BoardID & 0xffff));
  amcHeaders_.push_back(word);
}
