#include "RecoTracker/LSTGeometry/interface/SensorBinning.h"

namespace lstgeometry {

  bool isInThetaPhiBin(float theta, float phi, unsigned int theta_bin, unsigned int phi_bin) {
    if (theta_bin == 0) {
      if (theta > 3. * kThetaBinRad / 2.)
        return false;
    } else if (theta_bin == kNThetaBins - 1) {
      if (theta < (2 * (kNThetaBins - 1) - 1) * kThetaBinRad / 2.)
        return false;
    } else if (theta < (2 * theta_bin - 1) * kThetaBinRad / 2. ||
               theta > (2 * (theta_bin + 1) + 1) * kThetaBinRad / 2.) {
      return false;
    }

    float pi = std::numbers::pi_v<float>;
    if (phi_bin == 0) {
      if (phi > -pi + kPhiBinWidth && phi < pi - kPhiBinWidth)
        return false;
    } else {
      if (phi < -pi + (phi_bin - 1) * kPhiBinWidth || phi > -pi + (phi_bin + 1) * kPhiBinWidth)
        return false;
    }

    return true;
  }

  std::pair<unsigned int, unsigned int> getThetaPhiBins(float theta, float phi) {
    unsigned int theta_bin = std::clamp(static_cast<unsigned int>(theta / kThetaBinRad), 0u, kNThetaBins - 1);

    float pi = std::numbers::pi_v<float>;
    unsigned int phi_bin =
        std::clamp(static_cast<unsigned int>((phi + pi + kPhiBinWidth / 2.) / kPhiBinWidth), 0u, kNPhiBins - 1);
    if (phi >= pi - kPhiBinWidth / 2)
      phi_bin = 0;  // The 0 bin wraps around

    return std::make_pair(theta_bin, phi_bin);
  }

  std::pair<float, float> getCompatibleEtaRange(Sensor const& sensor, float zmin_bound, float zmax_bound) {
    float minr = sensor.extra->minR;
    float maxr = sensor.extra->maxR;
    float minz = sensor.extra->minZ;
    float maxz = sensor.extra->maxZ;
    float mineta = std::asinh((minz - zmin_bound) / (minz > 0 ? maxr : minr));
    float maxeta = std::asinh((maxz - zmax_bound) / (maxz > 0 ? minr : maxr));

    if (maxeta < mineta)
      std::swap(maxeta, mineta);
    return std::make_pair(mineta, maxeta);
  }

  std::pair<std::pair<float, float>, std::pair<float, float>> getCompatiblePhiRange(Sensor const& sensor,
                                                                                    float ptmin,
                                                                                    float ptmax) {
    float minr = sensor.extra->minR;
    float maxr = sensor.extra->maxR;
    float minphi = sensor.extra->minPhi;
    float maxphi = sensor.extra->maxPhi;
    float A = kC * kB / 2.;
    float pos_q_phi_lo_bound = phi_mpi_pi(A * minr / ptmax + minphi);
    float pos_q_phi_hi_bound = phi_mpi_pi(A * maxr / ptmin + maxphi);
    float neg_q_phi_lo_bound = phi_mpi_pi(-A * maxr / ptmin + minphi);
    float neg_q_phi_hi_bound = phi_mpi_pi(-A * minr / ptmax + maxphi);
    return std::make_pair(std::make_pair(pos_q_phi_lo_bound, pos_q_phi_hi_bound),
                          std::make_pair(neg_q_phi_lo_bound, neg_q_phi_hi_bound));
  }

  BinnedDetIds binDetIds(Sensors const& sensors) {
    BinnedDetIds binned_detids;

    for (auto const& [detid, sensor] : sensors) {
      if (!sensor.extra->lower)
        continue;
      // Only the central bin and its direct neighbors can be touched by the overlapping bin windows.
      auto [centerThetaBin, centerPhiBin] = getThetaPhiBins(sensor.extra->centerTheta, sensor.centerPhi);
      for (int thetaOffset = -1; thetaOffset <= 1; ++thetaOffset) {
        int thetaBin = static_cast<int>(centerThetaBin) + thetaOffset;
        if (thetaBin < 0 || thetaBin >= static_cast<int>(kNThetaBins))
          continue;
        for (int phiOffset = -1; phiOffset <= 1; ++phiOffset) {
          unsigned int phibin = static_cast<unsigned int>(
              (static_cast<int>(centerPhiBin) + static_cast<int>(kNPhiBins) + phiOffset) % static_cast<int>(kNPhiBins));
          unsigned int thetabin = static_cast<unsigned int>(thetaBin);
          if (isInThetaPhiBin(sensor.extra->centerTheta, sensor.centerPhi, thetabin, phibin)) {
            binnedDetIdsAt(binned_detids, sensor.extra->location, sensor.extra->layer, thetabin, phibin)
                .push_back(detid);
          }
        }
      }
    }

    return binned_detids;
  }

}  // namespace lstgeometry
