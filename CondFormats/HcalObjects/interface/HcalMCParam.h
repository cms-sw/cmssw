#ifndef HcalMCParam_h
#define HcalMCParam_h

/** 
\class HcalMCParam
\author Radek Ofierzynski
POOL object to store MC information
*/

#include "CondFormats/Serialization/interface/Serializable.h"
#include <cstdint>

// definition 8.Feb.2011
// MC signal shape integer variable assigned to each readout this way:
// 0 - regular HPD  HB/HE/HO shape
// 1 - "special" HB shape
// 2 - SiPMs shape (HO, possibly also in HB/HE)
// 3 - HF Shape
// 4 - ZDC shape
//
// change in definition  28.Oct.2011  sk
// mParam1 is now packed word.
//   pulseShapeID                              [0,500]          9 bits  (use this as phot0 detetor ID as well)
//   syncPhase = cms.bool(True),               bool             1 bit   (use this for QPLL unlocked channel)
//   binOfMaximum = cms.int32(5)               [1-10]           4 bits
//   timePhase = cms.double(5.0),              [-30.0,30.0]     8 bits  (0.25ns step)
//   timeSmearing = cms.bool(False)            bool             1 bit
//   packingScheme                                              4 bits
class HcalMCParam {
public:
  HcalMCParam() : mId(0), mParam1(0) {}

  HcalMCParam(unsigned long fId, unsigned int fParam1) : mId(fId), mParam1(fParam1) {}

  uint32_t rawId() const { return mId; }

  unsigned int param1() const { return mParam1; }
  unsigned int signalShape() const { return mParam1 & 0x1FF; }
  bool syncPhase() const { return (mParam1 >> 9) & 0x1; }
  unsigned int binOfMaximum() const { return (mParam1 >> 10) & 0xF; }
  float timePhase() const { return ((mParam1 >> 14) & 0xFF) / 4.0 - 32.0; }
  bool timeSmearing() const { return (mParam1 >> 22) & 0x1; }
  unsigned int packingScheme() const { return (mParam1 >> 27) & 0xF; }

private:
  uint32_t mId;
  uint32_t mParam1;

  COND_SERIALIZABLE;
};

#endif
