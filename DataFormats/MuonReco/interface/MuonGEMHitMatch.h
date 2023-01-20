#ifndef MuonReco_MuonGEMHitMatch_h
#define MuonReco_MuonGEMHitMatch_h

#include <cmath>

namespace reco {
  class MuonGEMHitMatch {
  public:
    float x;            // X position of the matched segment
    unsigned int mask;  // arbitration mask
    int bx;             // bunch crossing
    float y;            // Y position of the matched segment
    float eta;            // eta position of the matched segment
    float phi;            // phi position of the matched segment

    //MuonGEMHitMatch() : x(0), mask(0), bx(0) {}
    MuonGEMHitMatch() : x(0), mask(0), bx(0), y(0), eta(0), phi(0) {}
  };
}  // namespace reco

#endif
