#include "CondFormats/SiPixelObjects/interface/SiPixelVCal.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

bool SiPixelVCal::putSlopeAndOffset(const uint32_t& pixid, float& slopeValue, float& offsetValue) {
  std::map<unsigned int, VCal>::const_iterator id = m_vcal.find(pixid);
  if (id != m_vcal.end()) {
    edm::LogError("SiPixelVCal") << "SiPixelVCal for pixid " << pixid << " is already stored. Skippig this put"
                                 << std::endl;
    return false;
  } else {
    m_vcal[pixid] = {slopeValue, offsetValue};
  }
  return true;
}

SiPixelVCal::VCal SiPixelVCal::getSlopeAndOffset(const uint32_t& pixid) const {
  std::map<unsigned int, VCal>::const_iterator id = m_vcal.find(pixid);
  if (id != m_vcal.end())
    return id->second;
  else
    edm::LogError("SiPixelVCal") << "SiPixelVCal for pixid " << pixid << " is not stored" << std::endl;
  VCal vcal_default;
  return vcal_default;
}

float SiPixelVCal::getSlope(const uint32_t& pixid) const {
  std::map<unsigned int, VCal>::const_iterator id = m_vcal.find(pixid);
  if (id != m_vcal.end())
    return id->second.slope;
  else
    edm::LogError("SiPixelVCal") << "SiPixelVCal slope for pixid " << pixid << " is not stored" << std::endl;
  return 47.;
}

float SiPixelVCal::getOffset(const uint32_t& pixid) const {
  std::map<unsigned int, VCal>::const_iterator id = m_vcal.find(pixid);
  if (id != m_vcal.end())
    return id->second.offset;
  else
    edm::LogError("SiPixelVCal") << "SiPixelVCal offset for pixid " << pixid << " is not stored" << std::endl;
  return -60.;
}
