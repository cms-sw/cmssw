#include "CondFormats/SiPhase2TrackerObjects/interface/SiPhase2OuterTrackerLorentzAngle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

bool SiPhase2OuterTrackerLorentzAngle::putLorentzAngle(const uint32_t& detid, float value) {
  std::unordered_map<unsigned int, float>::const_iterator id = m_LA.find(detid);
  if (id != m_LA.end()) {
    edm::LogError("SiPhase2OuterTrackerLorentzAngle") << "SiPhase2OuterTrackerLorentzAngle for DetID " << detid
                                                      << " is already stored. Skipping this put" << std::endl;
    return false;
  } else
    m_LA[detid] = value;
  return true;
}

float SiPhase2OuterTrackerLorentzAngle::getLorentzAngle(const uint32_t& detid) const {
  std::unordered_map<unsigned int, float>::const_iterator id = m_LA.find(detid);
  if (id != m_LA.end())
    return id->second;
  else {
    throw cms::Exception("SiPhase2OuterTrackerLorentzAngle")
        << "SiPhase2OuterTrackerLorentzAngle for DetID " << detid << " is not stored" << std::endl;
  }
}

void SiPhase2OuterTrackerLorentzAngle::getLorentzAnglesByModuleType(const TrackerGeometry* trackerGeometry,
                                                                    const TrackerGeometry::ModuleType& theType,
                                                                    std::unordered_map<unsigned int, float>& out) const {
  for (const auto& [det, LA] : m_LA) {
    if (trackerGeometry->getDetectorType(det) == theType) {
      out[det] = LA;
    }
  }
}

void SiPhase2OuterTrackerLorentzAngle::printDebug(std::stringstream& ss, const TrackerTopology* /*trackerTopo*/) const {
  const std::unordered_map<unsigned int, float>& detid_la = getLorentzAngles();
  ss << "SiPhase2OuterTrackerLorentzAngleReader:" << std::endl;
  ss << "detid \t Lorentz angle" << std::endl;
  for (const auto& it : detid_la) {
    ss << it.first << "\t" << it.second << std::endl;
  }
}
