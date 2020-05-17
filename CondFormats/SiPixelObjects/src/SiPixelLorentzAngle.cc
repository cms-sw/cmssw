#include "CondFormats/SiPixelObjects/interface/SiPixelLorentzAngle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

bool SiPixelLorentzAngle::putLorentzAngle(const uint32_t& detid, float& value) {
  auto id = m_LA.find(detid);
  if (id != m_LA.end()) {
    edm::LogError("SiPixelLorentzAngle") << "SiPixelLorentzAngle for DetID " << detid
                                         << " is already stored. Skippig this put" << std::endl;
    return false;
  } else
    m_LA[detid] = value;
  return true;
}
float SiPixelLorentzAngle::getLorentzAngle(const uint32_t& detid) const {
  auto id = m_LA.find(detid);
  if (id != m_LA.end())
    return id->second;
  else {
    edm::LogError("SiPixelLorentzAngle") << "SiPixelLorentzAngle for DetID " << detid << " is not stored" << std::endl;
  }
  return 0;
}
