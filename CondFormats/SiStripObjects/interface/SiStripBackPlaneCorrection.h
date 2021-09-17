#ifndef SiStripBackPlaneCorrection_h
#define SiStripBackPlaneCorrection_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include <map>
#include <iostream>
// #include "CondFormats/SiStripObjects/interface/SiStripBaseObject.h"
#include "CondFormats/SiStripObjects/interface/SiStripDetSummary.h"
#include <cstdint>

/**
 * Author: Loic Quertenmont.  THis class is adapted from SiStripLorentzAngle
 * Stores the lorentz angle value for all DetIds. <br>
 * The values are saved internally in a std::map<detid, float backPlaneCorrection>. <br>
 * It can be filled either by the complete map (putBackPlaneCorrections) or passing
 * a single detIds and lorentzAngles (putBackPlaneCorrection). <br>
 * In the same way getBackPlaneCorrections returns the complete map, while getBackPlaneCorrection
 * the value corresponding to a given DetId. <br>
 * The printDebug method prints BackPlaneCorrections for all detIds. <br>
 * The printSummary mehtod uses the SiStripDetSummary class to produce a summary
 * of BackPlaneCorrection values divided by subdetector and layer/disk.
 */

// class SiStripBackPlaneCorrection : public SiStripBaseObject
class SiStripBackPlaneCorrection {
public:
  SiStripBackPlaneCorrection(){};
  ~SiStripBackPlaneCorrection(){};

  inline void putLorentsAngles(std::map<unsigned int, float>& BPC) { m_BPC = BPC; }
  inline const std::map<unsigned int, float>& getBackPlaneCorrections() const { return m_BPC; }

  bool putBackPlaneCorrection(const uint32_t&, float);
  float getBackPlaneCorrection(const uint32_t&) const;

  /// Prints BackPlaneCorrections for all detIds.
  void printDebug(std::stringstream& ss, const TrackerTopology* trackerTopo) const;
  /// Prints the mean value of the BackPlaneCorrection divided by subdetector, layer and mono/stereo.
  void printSummary(std::stringstream& ss, const TrackerTopology* trackerTopo) const;

private:
  std::map<unsigned int, float> m_BPC;

  COND_SERIALIZABLE;
};

#endif
