#include <cmath>
#include <limits>
#include "DataFormats/TrackReco/interface/TrackResiduals.h"

namespace reco {

  namespace {

    inline TrackResiduals::StorageType pack(float x, float min, float fact) {
      constexpr float amax = std::numeric_limits<TrackResiduals::StorageType>::max();
      return std::min(std::round(fact * (x - min)), amax);
    }

    inline float unpack(TrackResiduals::StorageType x, float min, float fact) { return fact * x + min; }

    constexpr float pmin = -32;    // overkill
    constexpr float pmult = 1000;  // overkill
    constexpr float pdiv = 1. / pmult;
    constexpr float rmin = -3.2;    // (in cm)
    constexpr float rmult = 10000;  // micron
    constexpr float rdiv = 1. / rmult;

  }  // namespace

  float TrackResiduals::unpack_pull(StorageType x) { return unpack(x, pmin, pdiv); }
  TrackResiduals::StorageType TrackResiduals::pack_pull(float x) { return pack(x, pmin, pmult); }
  float TrackResiduals::unpack_residual(StorageType x) { return unpack(x, rmin, rdiv); }
  TrackResiduals::StorageType TrackResiduals::pack_residual(float x) { return pack(x, rmin, rmult); }

  void TrackResiduals::setResidualXY(int idx, float residualX, float residualY) {
    m_storage[4 * idx] = pack_residual(residualX);
    m_storage[4 * idx + 1] = pack_residual(residualY);
  }
  void TrackResiduals::setPullXY(int idx, float pullX, float pullY) {
    m_storage[4 * idx + 2] = pack_pull(pullX);
    m_storage[4 * idx + 3] = pack_pull(pullY);
  }

}  // namespace reco
