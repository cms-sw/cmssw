#ifndef DataFormats_Math_RectangularEtaPhiRegion_h
#define DataFormats_Math_RectangularEtaPhiRegion_h

#include "DataFormats/Math/interface/normalizedPhi.h"

class RectangularEtaPhiRegion {
public:
  RectangularEtaPhiRegion(float etaLow, float etaHigh, float phiLow, float phiHigh)
      : ceta(0.5f * (etaHigh + etaLow)), deta(0.5f * std::abs(etaHigh - etaLow)) {
    phiHigh = proxim(phiHigh, phiLow);
    constexpr float c1 = 2. * M_PI;
    if (phiHigh < phiLow)
      phiHigh += c1;
    dphi = 0.5f * (phiHigh - phiLow);
    cphi = phiLow + dphi;
  }

  bool inRegion(float eta, float phi) const {
    return std::abs(eta - ceta) < deta && std::abs(proxim(phi, cphi) - cphi) < dphi;
  }

  auto etaLow() const { return ceta - deta; }
  auto etaHigh() const { return ceta + deta; }
  auto phiLow() const { return cphi - dphi; }
  auto phiHigh() const { return cphi + dphi; }

private:
  float ceta;
  float deta;
  float cphi;
  float dphi;
};

#endif
