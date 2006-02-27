#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"

SiStripNoises::SiStripNoises(){}
SiStripNoises::~SiStripNoises(){}

const std::vector<SiStripNoises::SiStripData> & SiStripNoises::getSiStripNoiseVector(const uint32_t & DetId) const {
  SiStripNoiseMapIterator mapiter=m_noises.find(DetId);
  if (mapiter!=m_noises.end())
    return mapiter->second;
  return SiStripNoiseVector();
};


