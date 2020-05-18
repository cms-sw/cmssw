#ifndef CondCore_SiPhase2TrackerObjects_SiPhase2OuterTrackerLorentzAngle_h
#define CondCore_SiPhase2TrackerObjects_SiPhase2OuterTrackerLorentzAngle_h

#include "CondFormats/Serialization/interface/Serializable.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include <vector>
#include <map>
#include <iostream>
#include <cstdint>

/**
 * Stores the lorentz angle value for all DetIds.
 * The values are saved internally in a std::map<detid, lorentzAngle>.
 * It can be filled either by the complete map (putLorentzAngles) or passing
 * a single detIds and lorentzAngles (putLorentzAngle).
 * In the same way getLorentzAngles returns the complete map, while getLorentzAngle
 * the value corresponding to a given DetId.
 * The printDebug method prints LorentzAngles for all detIds.
 * The printSummary mehtod uses the SiStripDetSummary class to produce a summary
 * of LorentzAngle values divided by subdetector and layer/disk.
 */

class SiPhase2OuterTrackerLorentzAngle {
public:
  SiPhase2OuterTrackerLorentzAngle(){};
  ~SiPhase2OuterTrackerLorentzAngle(){};

  inline void putLorentsAngles(std::map<unsigned int, float>& LA) { m_LA = LA; }
  inline const std::map<unsigned int, float>& getLorentzAngles() const { return m_LA; }

  bool putLorentzAngle(const uint32_t&, float);
  float getLorentzAngle(const uint32_t&) const;

  // Prints LorentzAngles for all detIds.
  void printDebug(std::stringstream& ss, const TrackerTopology* trackerTopo) const;

  // Prints the mean value of the LorentzAngle divided by subdetector, layer and mono/stereo.
  //void printSummary(std::stringstream& ss, const TrackerTopology* trackerTopo) const;

private:
  std::map<unsigned int, float> m_LA;

  COND_SERIALIZABLE;
};

#endif
