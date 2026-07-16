#include <algorithm>

#include "RecoTracker/LSTGeometry/interface/Common.h"
#include "RecoTracker/LSTGeometry/interface/PixelMap.h"

namespace lstgeometry {

  PixelMap buildPixelMap(Sensors const& sensors, float pt_cut) {
    // Charge 0 is the union of charge 1 and charge -1
    PixelMap maps;
    maps.reserve(12);

    std::size_t nSuperbin = kPtBounds.size() * kNPhi * kNEta * kNZ;
    constexpr float zBinWidth = 60.f / kNZ;
    constexpr float etaBinScale = kNEta / 5.2f;
    constexpr float inversePhiBinWidth = kNPhi / (2.f * std::numbers::pi_v<float>);
    constexpr float curvatureToPhi = kC * kB / 2.f;

    // Initialize empty lists for the pixel map
    for (unsigned int layer : {1, 2}) {
      for (unsigned int subdet : {SubDet::Barrel, SubDet::Endcap}) {
        for (int charge : {-1, 0, 1}) {
          maps.try_emplace({layer, subdet, charge}, nSuperbin);
        }
      }
    }

    // Loop over the detids and for each detid compute which superbins it is connected to
    for (auto const& [detId, sensor] : sensors) {
      auto layer = sensor.extra->layer;
      if (layer > 2)
        continue;

      // Skip if the module is not PS module and is not lower sensor
      if (sensor.moduleType == ModuleType::Ph2SS || !sensor.extra->lower)
        continue;

      // Phase-2 enum differs from the legacy one used here
      unsigned int subdet = sensor.extra->subdet == SubDetector::P2OTB ? SubDet::Barrel : SubDet::Endcap;
      auto location = sensor.extra->location;
      float minR = sensor.extra->minR;
      float maxR = sensor.extra->maxR;
      float minZ = sensor.extra->minZ;
      float maxZ = sensor.extra->maxZ;
      float minPhi = sensor.extra->minPhi;
      float maxPhi = sensor.extra->maxPhi;
      float invMinR = 1.f / minR;
      float invMaxR = 1.f / maxR;
      float minEtaInvR = minZ > 0 ? invMaxR : invMinR;
      float maxEtaInvR = maxZ > 0 ? invMinR : invMaxR;

      auto& pos_map = maps.at({layer, subdet, 1});
      auto& neg_map = maps.at({layer, subdet, -1});

      // For this module, now compute which super bins they belong to
      // To compute which super bins it belongs to, one needs to provide at least pt and z window to compute compatible eta and phi range
      // So we have a loop in pt and Z
      for (unsigned int ipt = 0; ipt < kPtBounds.size(); ipt++) {
        // The ptmin, ptmax of consideration
        float pt_lo = ipt == 0 ? pt_cut : kPtBounds[ipt - 1];
        float pt_hi = kPtBounds[ipt];

        float invPtLo = 1.f / pt_lo;
        float invPtHi = 1.f / pt_hi;
        float pos_q_phi_lo_bound = phi_mpi_pi(curvatureToPhi * minR * invPtHi + minPhi);
        float pos_q_phi_hi_bound = phi_mpi_pi(curvatureToPhi * maxR * invPtLo + maxPhi);
        float neg_q_phi_lo_bound = phi_mpi_pi(-curvatureToPhi * maxR * invPtLo + minPhi);
        float neg_q_phi_hi_bound = phi_mpi_pi(-curvatureToPhi * minR * invPtHi + maxPhi);

        unsigned int iphimin_pos =
            static_cast<unsigned int>((pos_q_phi_lo_bound + std::numbers::pi_v<float>)*inversePhiBinWidth);
        unsigned int iphimax_pos =
            static_cast<unsigned int>((pos_q_phi_hi_bound + std::numbers::pi_v<float>)*inversePhiBinWidth);
        unsigned int iphimin_neg =
            static_cast<unsigned int>((neg_q_phi_lo_bound + std::numbers::pi_v<float>)*inversePhiBinWidth);
        unsigned int iphimax_neg =
            static_cast<unsigned int>((neg_q_phi_hi_bound + std::numbers::pi_v<float>)*inversePhiBinWidth);

        for (unsigned int iz = 0; iz < kNZ; iz++) {
          // The zmin, zmax of consideration
          float zmin = -30.f + iz * zBinWidth - 0.05f;
          float zmax = -30.f + (iz + 1) * zBinWidth + 0.05f;
          float etamin = std::asinh((minZ - zmin) * minEtaInvR);
          float etamax = std::asinh((maxZ - zmax) * maxEtaInvR);
          if (etamax < etamin)
            std::swap(etamax, etamin);

          etamin -= 0.05;
          etamax += 0.05;

          if (layer == 2 && location == Location::endcap) {
            if (etamax < 2.3)
              continue;
            if (etamin < 2.3)
              etamin = 2.3;
          }

          // Compute the indices of the compatible eta range
          unsigned int ietamin = static_cast<unsigned int>(std::max((etamin + 2.6f) * etaBinScale, 0.0f));
          unsigned int ietamax =
              static_cast<unsigned int>(std::min((etamax + 2.6f) * etaBinScale, static_cast<float>(kNEta - 1)));

          // <= to cover some inefficiencies
          for (unsigned int ieta = ietamin; ieta <= ietamax; ieta++) {
            unsigned int superbinEtaBase = (ipt * kNPhi * kNEta * kNZ) + (ieta * kNPhi * kNZ) + iz;
            // if the range is crossing the -pi v. pi boundary special care is needed
            if (iphimin_pos <= iphimax_pos) {
              for (unsigned int iphi = iphimin_pos; iphi < iphimax_pos; iphi++) {
                unsigned int isuperbin = superbinEtaBase + iphi * kNZ;
                pos_map[isuperbin].push_back(detId);
              }
            } else {
              for (unsigned int iphi = 0; iphi < iphimax_pos; iphi++) {
                unsigned int isuperbin = superbinEtaBase + iphi * kNZ;
                pos_map[isuperbin].push_back(detId);
              }
              for (unsigned int iphi = iphimin_pos; iphi < kNPhi; iphi++) {
                unsigned int isuperbin = superbinEtaBase + iphi * kNZ;
                pos_map[isuperbin].push_back(detId);
              }
            }
            if (iphimin_neg <= iphimax_neg) {
              for (unsigned int iphi = iphimin_neg; iphi < iphimax_neg; iphi++) {
                unsigned int isuperbin = superbinEtaBase + iphi * kNZ;
                neg_map[isuperbin].push_back(detId);
              }
            } else {
              for (unsigned int iphi = 0; iphi < iphimax_neg; iphi++) {
                unsigned int isuperbin = superbinEtaBase + iphi * kNZ;
                neg_map[isuperbin].push_back(detId);
              }
              for (unsigned int iphi = iphimin_neg; iphi < kNPhi; iphi++) {
                unsigned int isuperbin = superbinEtaBase + iphi * kNZ;
                neg_map[isuperbin].push_back(detId);
              }
            }
          }
        }
      }
    }

    for (unsigned int layer : {1, 2}) {
      for (unsigned int subdet : {SubDet::Barrel, SubDet::Endcap}) {
        auto const& pos_map = maps.at({layer, subdet, 1});
        auto const& neg_map = maps.at({layer, subdet, -1});
        auto& zero_map = maps.at({layer, subdet, 0});
        for (std::size_t isuperbin = 0; isuperbin < nSuperbin; ++isuperbin) {
          zero_map[isuperbin].reserve(pos_map[isuperbin].size() + neg_map[isuperbin].size());
          zero_map[isuperbin].insert(zero_map[isuperbin].end(), pos_map[isuperbin].begin(), pos_map[isuperbin].end());
          zero_map[isuperbin].insert(zero_map[isuperbin].end(), neg_map[isuperbin].begin(), neg_map[isuperbin].end());
        }
      }
    }

    for (auto& [key, vec_of_vecs] : maps) {
      for (auto& vec : vec_of_vecs) {
        if (vec.empty())
          continue;
        std::sort(vec.begin(), vec.end());
        vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
      }
    }
    return maps;
  }

}  // namespace lstgeometry
