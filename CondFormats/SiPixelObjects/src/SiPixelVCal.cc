#include "CondFormats/SiPixelObjects/interface/SiPixelVCal.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

bool SiPixelVCal::putSlope(const uint32_t& pixid, float& value) {
  std::map<unsigned int, float>::const_iterator id = m_slope.find(pixid);
  if(id!=m_slope.end()){
    edm::LogError("SiPixelVCal") << "SiPixelVCal for pixid " << pixid
                                 << " is already stored. Skippig this put" << std::endl;
    return false;
  }else{
    m_slope[pixid] = value;
  }
  return true;
}

bool SiPixelVCal::putOffset(const uint32_t& pixid, float& value) {
  std::map<unsigned int, float>::const_iterator id = m_offset.find(pixid);
  if(id!=m_offset.end()){
    edm::LogError("SiPixelVCal") << "SiPixelVCal for pixid " << pixid
                                 << " is already stored. Skippig this put" << std::endl;
    return false;
  }else{
    m_offset[pixid] = value;
  }
  return true;
}

bool SiPixelVCal::putVCal(const uint32_t& pixid, float& slopeValue, float& offsetValue) {
  return putSlope(pixid,slopeValue) and putOffset(pixid,offsetValue);
}

float SiPixelVCal::getSlope(const uint32_t& pixid) const {
  std::map<unsigned int, float>::const_iterator id = m_slope.find(pixid);
  if(id!=m_slope.end())
    return id->second;
  else
    edm::LogError("SiPixelVCal") << "SiPixelVCal for pixid " << pixid << " is not stored" << std::endl;
  return 47.;
}

float SiPixelVCal::getOffset(const uint32_t& pixid) const {
  std::map<unsigned int, float>::const_iterator id = m_offset.find(pixid);
  if(id!=m_offset.end())
    return id->second;
  else
    edm::LogError("SiPixelVCal") << "SiPixelVCal for pixid " << pixid << " is not stored" << std::endl;
  return -60.;
}
