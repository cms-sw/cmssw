#include "CondFormats/SiPhase2TrackerObjects/interface/SiPhase2OuterTrackerLorentzAngle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

bool SiPhase2OuterTrackerLorentzAngle::putLorentzAngle(const uint32_t& detid, float value) {
  std::unordered_map<unsigned int, float>::const_iterator id = m_LA.find(detid);
  if (id != m_LA.end()) {
    edm::LogError("SiPhase2OuterTrackerLorentzAngle") << "SiPhase2OuterTrackerLorentzAngle for DetID " << detid
                                                      << " is already stored. Skippig this put" << std::endl;
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
    edm::LogError("SiPhase2OuterTrackerLorentzAngle")
        << "SiPhase2OuterTrackerLorentzAngle for DetID " << detid << " is not stored" << std::endl;
  }
  return 0;
}

void SiPhase2OuterTrackerLorentzAngle::getLorentzAngles_PSP(const TrackerGeometry* trackerGeometry,
                                                            std::unordered_map<unsigned int, float>& out) const {
  for (const auto& [det, LA] : m_LA) {
    if (trackerGeometry->getDetectorType(det) == TrackerGeometry::ModuleType::Ph2PSP) {
      out[det] = LA;
    }
  }
}

void SiPhase2OuterTrackerLorentzAngle::getLorentzAngles_PSS(const TrackerGeometry* trackerGeometry,
                                                            std::unordered_map<unsigned int, float>& out) const {
  for (const auto& [det, LA] : m_LA) {
    if (trackerGeometry->getDetectorType(det) == TrackerGeometry::ModuleType::Ph2PSS) {
      out[det] = LA;
    }
  }
}

void SiPhase2OuterTrackerLorentzAngle::getLorentzAngles_2S(const TrackerGeometry* trackerGeometry,
                                                           std::unordered_map<unsigned int, float>& out) const {
  for (const auto& [det, LA] : m_LA) {
    if (trackerGeometry->getDetectorType(det) == TrackerGeometry::ModuleType::Ph2SS) {
      out[det] = LA;
    }
  }
}

void SiPhase2OuterTrackerLorentzAngle::printDebug(std::stringstream& ss, const TrackerTopology* /*trackerTopo*/) const {
  std::unordered_map<unsigned int, float> detid_la = getLorentzAngles();
  std::unordered_map<unsigned int, float>::const_iterator it;
  size_t count = 0;
  ss << "SiPhase2OuterTrackerLorentzAngleReader:" << std::endl;
  ss << "detid \t Lorentz angle" << std::endl;
  for (it = detid_la.begin(); it != detid_la.end(); ++it) {
    ss << it->first << "\t" << it->second << std::endl;
    ++count;
  }
}
