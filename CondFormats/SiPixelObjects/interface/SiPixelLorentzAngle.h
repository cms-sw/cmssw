#ifndef SiPixelLorentzAngle_h
#define SiPixelLorentzAngle_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include <map>
#include <iostream>
#include <cstdint>

class SiPixelLorentzAngle {
public:
  SiPixelLorentzAngle(){};
  ~SiPixelLorentzAngle(){};

  inline void putLorentsAngles(std::map<unsigned int, float>& LA) { m_LA = LA; }
  inline const std::map<unsigned int, float>& getLorentzAngles() const { return m_LA; }

  bool putLorentzAngle(const uint32_t&, float&);
  float getLorentzAngle(const uint32_t&) const;

private:
  std::map<unsigned int, float> m_LA;

  COND_SERIALIZABLE;
};

#endif
