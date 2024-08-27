#ifndef RecoTracker_MkFitCore_interface_DeadRegion_h
#define RecoTracker_MkFitCore_interface_DeadRegion_h

#include <vector>

namespace mkfit {
  struct DeadRegion {
    float phi1, phi2, q1, q2;
    DeadRegion(float a1, float a2, float b1, float b2) : phi1(a1), phi2(a2), q1(b1), q2(b2) {}
  };
  typedef std::vector<DeadRegion> DeadVec;
}  // namespace mkfit

#endif
