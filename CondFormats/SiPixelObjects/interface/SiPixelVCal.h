#ifndef SiPixelVCal_h
#define SiPixelVCal_h
#include "CondFormats/Serialization/interface/Serializable.h"
#include <vector>
#include <map>
#include <iostream>
#include <cstdint>

class SiPixelVCal {
public:
  SiPixelVCal(){};
  ~SiPixelVCal(){};

  inline void putSlope(std::map<unsigned int, float>& slope) { m_slope = slope; }
  inline void putOffset(std::map<unsigned int, float>& offset) { m_offset = offset; }
  inline const std::map<unsigned int, float>& getSlope() const { return m_slope; }
  inline const std::map<unsigned int, float>& getOffset() const { return m_offset; }

  // integer is pixid, see CondTools/SiPixel/test/SiPixelVCalDB.h
  bool putSlope(const uint32_t&, float&);
  bool putOffset(const uint32_t&, float&);
  bool putVCal(const uint32_t&, float&, float&);
  float getSlope(const uint32_t&) const;
  float getOffset(const uint32_t&) const;

private:
  // Convert VCal to #electrons, which changes with irradiation and varies between pixel layers & disks
  //     VCal = (slope) * (#electrons) + offset
  // with
  //   slope  ~  47 (50 for L1)
  //   offset ~ -60 (-670 for L1)
  std::map<unsigned int, float> m_slope;
  std::map<unsigned int, float> m_offset;

  COND_SERIALIZABLE;
};

#endif
