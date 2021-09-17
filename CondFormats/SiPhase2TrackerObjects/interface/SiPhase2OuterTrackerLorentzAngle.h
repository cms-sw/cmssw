#ifndef CondCore_SiPhase2TrackerObjects_SiPhase2OuterTrackerLorentzAngle_h
#define CondCore_SiPhase2TrackerObjects_SiPhase2OuterTrackerLorentzAngle_h

#include "CondFormats/Serialization/interface/Serializable.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include <vector>
#include <map>
#include <iostream>
#include <cstdint>

/**
 * Stores the Lorentz Angle value for all DetIds.
 * The values are saved internally in a std::unordered_map<detid, lorentzAngle>.
 * It can be filled either by the complete map (putLorentzAngles) or passing
 * a single detIds and lorentzAngles (putLorentzAngle).
 * In the same way getLorentzAngles returns the complete map, while getLorentzAngle
 * the value corresponding to a given DetId.
 * The three methods getLorentzAngles_PSP, getLorentzAngles_PSS and getLorentzAngles_2S
 * are provided to retrieve by reference the Lorentz Angle values map for the P- and S- sensors,
 * in the PS modules separately, as well as the values for the S-senors in the 2S modules.
 * The printDebug method prints LorentzAngles for all detIds.
 */

class SiPhase2OuterTrackerLorentzAngle {
public:
  SiPhase2OuterTrackerLorentzAngle(){};
  ~SiPhase2OuterTrackerLorentzAngle(){};

  inline void putLorentzAngles(std::unordered_map<unsigned int, float>& LA) { m_LA = LA; }
  inline const std::unordered_map<unsigned int, float>& getLorentzAngles() const { return m_LA; }

  void getLorentzAngles_PSP(const TrackerGeometry* geo, std::unordered_map<unsigned int, float>& out) const {
    getLorentzAnglesByModuleType(geo, TrackerGeometry::ModuleType::Ph2PSP, out);
  }

  void getLorentzAngles_PSS(const TrackerGeometry* geo, std::unordered_map<unsigned int, float>& out) const {
    getLorentzAnglesByModuleType(geo, TrackerGeometry::ModuleType::Ph2PSS, out);
  }

  void getLorentzAngles_2S(const TrackerGeometry* geo, std::unordered_map<unsigned int, float>& out) const {
    getLorentzAnglesByModuleType(geo, TrackerGeometry::ModuleType::Ph2SS, out);
  }

  bool putLorentzAngle(const uint32_t&, float);
  float getLorentzAngle(const uint32_t&) const;

  // Prints LorentzAngles for all detIds.
  void printDebug(std::stringstream& ss, const TrackerTopology* trackerTopo) const;

private:
  void getLorentzAnglesByModuleType(const TrackerGeometry* trackerGeometry,
                                    const TrackerGeometry::ModuleType& theType,
                                    std::unordered_map<unsigned int, float>& out) const;

  std::unordered_map<unsigned int, float> m_LA;

  COND_SERIALIZABLE;
};

#endif
