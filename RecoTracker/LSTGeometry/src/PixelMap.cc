#include "RecoTracker/LSTGeometry/interface/Common.h"
#include "RecoTracker/LSTGeometry/interface/PixelMap.h"

namespace lstgeometry {

  PixelMap buildPixelMap(Sensors const& sensors, float pt_cut) {
    // Charge 0 is the union of charge 1 and charge -1
    PixelMap maps;

    std::size_t nSuperbin = kPtBounds.size() * kNPhi * kNEta * kNZ;

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
      // Phase-2 enum differs from the legacy one used here
      unsigned int subdet = sensor.extra->subdet == SubDetector::P2OTB ? SubDet::Barrel : SubDet::Endcap;
      auto layer = sensor.extra->layer;
      if (layer > 2)
        continue;
      auto location = sensor.extra->location;

      // Skip if the module is not PS module and is not lower sensor
      if (sensor.moduleType == ModuleType::Ph2SS || !sensor.extra->lower)
        continue;

      // For this module, now compute which super bins they belong to
      // To compute which super bins it belongs to, one needs to provide at least pt and z window to compute compatible eta and phi range
      // So we have a loop in pt and Z
      for (unsigned int ipt = 0; ipt < kPtBounds.size(); ipt++) {
        for (unsigned int iz = 0; iz < kNZ; iz++) {
          // The zmin, zmax of consideration
          float zmin = -30 + iz * (60. / kNZ);
          float zmax = -30 + (iz + 1) * (60. / kNZ);

          zmin -= 0.05;
          zmax += 0.05;

          // The ptmin, ptmax of consideration
          float pt_lo = ipt == 0 ? pt_cut : kPtBounds[ipt - 1];
          float pt_hi = kPtBounds[ipt];

          auto [etamin, etamax] = getCompatibleEtaRange(sensor, zmin, zmax);

          etamin -= 0.05;
          etamax += 0.05;

          if (layer == 2 && location == Location::endcap) {
            if (etamax < 2.3)
              continue;
            if (etamin < 2.3)
              etamin = 2.3;
          }

          // Compute the indices of the compatible eta range
          unsigned int ietamin = static_cast<unsigned int>(std::max((etamin + 2.6f) / (5.2f / kNEta), 0.0f));
          unsigned int ietamax =
              static_cast<unsigned int>(std::min((etamax + 2.6f) / (5.2f / kNEta), static_cast<float>(kNEta - 1)));

          auto phi_ranges = getCompatiblePhiRange(sensor, pt_lo, pt_hi);

          unsigned int iphimin_pos = static_cast<unsigned int>((phi_ranges.first.first + std::numbers::pi_v<float>) /
                                                               (2. * std::numbers::pi_v<float> / kNPhi));
          unsigned int iphimax_pos = static_cast<unsigned int>((phi_ranges.first.second + std::numbers::pi_v<float>) /
                                                               (2. * std::numbers::pi_v<float> / kNPhi));
          unsigned int iphimin_neg = static_cast<unsigned int>((phi_ranges.second.first + std::numbers::pi_v<float>) /
                                                               (2. * std::numbers::pi_v<float> / kNPhi));
          unsigned int iphimax_neg = static_cast<unsigned int>((phi_ranges.second.second + std::numbers::pi_v<float>) /
                                                               (2. * std::numbers::pi_v<float> / kNPhi));

          // <= to cover some inefficiencies
          for (unsigned int ieta = ietamin; ieta <= ietamax; ieta++) {
            // if the range is crossing the -pi v. pi boundary special care is needed
            if (iphimin_pos <= iphimax_pos) {
              for (unsigned int iphi = iphimin_pos; iphi < iphimax_pos; iphi++) {
                unsigned int isuperbin = (ipt * kNPhi * kNEta * kNZ) + (ieta * kNPhi * kNZ) + (iphi * kNZ) + iz;
                maps[{layer, subdet, 1}][isuperbin].insert(detId);
                maps[{layer, subdet, 0}][isuperbin].insert(detId);
              }
            } else {
              for (unsigned int iphi = 0; iphi < iphimax_pos; iphi++) {
                unsigned int isuperbin = (ipt * kNPhi * kNEta * kNZ) + (ieta * kNPhi * kNZ) + (iphi * kNZ) + iz;
                maps[{layer, subdet, 1}][isuperbin].insert(detId);
                maps[{layer, subdet, 0}][isuperbin].insert(detId);
              }
              for (unsigned int iphi = iphimin_pos; iphi < kNPhi; iphi++) {
                unsigned int isuperbin = (ipt * kNPhi * kNEta * kNZ) + (ieta * kNPhi * kNZ) + (iphi * kNZ) + iz;
                maps[{layer, subdet, 1}][isuperbin].insert(detId);
                maps[{layer, subdet, 0}][isuperbin].insert(detId);
              }
            }
            if (iphimin_neg <= iphimax_neg) {
              for (unsigned int iphi = iphimin_neg; iphi < iphimax_neg; iphi++) {
                unsigned int isuperbin = (ipt * kNPhi * kNEta * kNZ) + (ieta * kNPhi * kNZ) + (iphi * kNZ) + iz;
                maps[{layer, subdet, -1}][isuperbin].insert(detId);
                maps[{layer, subdet, 0}][isuperbin].insert(detId);
              }
            } else {
              for (unsigned int iphi = 0; iphi < iphimax_neg; iphi++) {
                unsigned int isuperbin = (ipt * kNPhi * kNEta * kNZ) + (ieta * kNPhi * kNZ) + (iphi * kNZ) + iz;
                maps[{layer, subdet, -1}][isuperbin].insert(detId);
                maps[{layer, subdet, 0}][isuperbin].insert(detId);
              }
              for (unsigned int iphi = iphimin_neg; iphi < kNPhi; iphi++) {
                unsigned int isuperbin = (ipt * kNPhi * kNEta * kNZ) + (ieta * kNPhi * kNZ) + (iphi * kNZ) + iz;
                maps[{layer, subdet, -1}][isuperbin].insert(detId);
                maps[{layer, subdet, 0}][isuperbin].insert(detId);
              }
            }
          }
        }
      }
    }

    return maps;
  }

}  // namespace lstgeometry
