#ifndef SiPixelVCal_h
#define SiPixelVCal_h
#include <cstdint>
#include <iostream>
#include <map>
#include <vector>
#include "CondFormats/Serialization/interface/Serializable.h"

class SiPixelVCal {
public:
  SiPixelVCal(){};
  ~SiPixelVCal(){};

  using mapToDetId = std::map<uint32_t, float>;

  struct VCal {
    float slope = 47.;
    float offset = -60.;
    COND_SERIALIZABLE;
  };

  inline void putSlopeAndOffset(std::map<unsigned int, VCal>& vcal) { m_vcal = vcal; }
  inline const std::map<unsigned int, VCal>& getSlopeAndOffset() const { return m_vcal; }
  bool putSlopeAndOffset(const uint32_t&, float&, float&);
  VCal getSlopeAndOffset(const uint32_t&) const;
  float getSlope(const uint32_t&) const;
  float getOffset(const uint32_t&) const;
  mapToDetId getAllSlopes() const;
  mapToDetId getAllOffsets() const;
  // uint32_t is pixid, see CondTools/SiPixel/test/SiPixelVCalDB.h

private:
  // Convert VCal to #electrons, which changes with irradiation and varies
  // between pixel layers & disks
  //     #electrons = slope * VCal + offset
  // with
  //   slope  ~  47 (50 for L1)
  //   offset ~ -60 (-670 for L1)
  std::map<unsigned int, VCal> m_vcal;

  COND_SERIALIZABLE;
};

#endif
