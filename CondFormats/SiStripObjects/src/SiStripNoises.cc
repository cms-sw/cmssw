#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "FWCore/Utilities/interface/Exception.h"

SiStripNoises::SiStripNoises(){}
SiStripNoises::~SiStripNoises(){}

const std::vector<SiStripNoises::SiStripData> & SiStripNoises::getSiStripNoiseVector(const uint32_t & DetId) const {
  SiStripNoiseMapIterator mapiter=m_noises.find(DetId);
  if (mapiter!=m_noises.end())
    return mapiter->second;

  throw cms::Exception("CorruptData")
    << "[SiStripNoises::getSiStripNoiseVector] looking for SiStripNoise for a detid not existing in the DB... detid = " << DetId;
};


