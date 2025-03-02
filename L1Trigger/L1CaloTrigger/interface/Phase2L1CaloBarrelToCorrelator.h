#ifndef PHASE_2_L1_CALO_BARREL_TO_CORRELATOR
#define PHASE_2_L1_CALO_BARREL_TO_CORRELATOR

#include "DataFormats/L1TCalorimeterPhase2/interface/GCTEmDigiCluster.h"
#include "DataFormats/L1TCalorimeterPhase2/interface/GCTHadDigiCluster.h"

#include "L1Trigger/L1CaloTrigger/interface/Phase2L1CaloEGammaUtils.h"

/*
 * Returns the difference in the azimuth coordinates of phi1 and phi2 (all in degrees not radians), taking the wrap-around at 180 degrees into account
 */
inline float p2eg::deltaPhiInDegrees(float phi1, float phi2, const float c = 180) {
  float r = std::fmod(phi1 - phi2, 2.0 * c);
  if (r < -c) {
    r += 2.0 * c;
  } else if (r > c) {
    r -= 2.0 * c;
  }
  return r;
}

/*
 * For a given phi in degrees (e.g. computed from some difference), return the phi (in degrees) which takes the wrap-around at 180 degrees into account
 */
inline float p2eg::wrappedPhiInDegrees(float phi) { return p2eg::deltaPhiInDegrees(phi, 0); }

#endif