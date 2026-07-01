
#ifndef DataFormats_CaloRecHit_CaloClusterSoA_H
#define DataFormats_CaloRecHit_CaloClusterSoA_H

// Authors: Simone Balducci, Felice Pantaleo, Wahid Redjeb, Aurora Perego, Leonardo Beltrame

#include "DataFormats/SoATemplate/interface/SoABlocks.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloID.h"
#include "DataFormats/DetId/interface/DetId.h"
#include <xtd/xtd.h>

namespace reco {

  GENERATE_SOA_LAYOUT(CaloClusterSoAPosition,
                      SOA_COLUMN(float, x),
                      SOA_COLUMN(float, y),
                      SOA_COLUMN(float, z),
                      SOA_COLUMN(int, layer),
                      SOA_COLUMN(int, cells))

  GENERATE_SOA_LAYOUT(CaloClusterSoAEnergy,
                      SOA_COLUMN(float, energy),
                      SOA_COLUMN(float, correctedEnergy),
                      SOA_COLUMN(float, correctedEnergyUncertainty))

  GENERATE_SOA_LAYOUT(CaloClusterSoAIndexes,
                      SOA_COLUMN(CaloID, caloID),
                      SOA_COLUMN(CaloCluster::AlgoID, algoID),
                      SOA_COLUMN(DetId, seedID),
                      SOA_COLUMN(uint32_t, flags))

  GENERATE_SOA_LAYOUT(CaloClusterSoATiming, SOA_COLUMN(float, time), SOA_COLUMN(float, timeError))

  // clang-format off
  GENERATE_SOA_BLOCKS(CaloClusterSoALayout,
                      SOA_BLOCK(position, CaloClusterSoAPosition),
                      SOA_BLOCK(energy, CaloClusterSoAEnergy),
                      SOA_BLOCK(indexes, CaloClusterSoAIndexes),
                      SOA_BLOCK(timing, CaloClusterSoATiming),
                      SOA_CONST_VIEW_METHODS(
                        SOA_HOST_DEVICE auto r(std::integral auto idx) const {
                          const auto x = this->position()[idx].x();
                          const auto y = this->position()[idx].y();
                          return xtd::sqrt(x * x + y * y);
                        }
                        SOA_HOST_DEVICE auto phi(std::integral auto idx) const {
                          const auto x = this->position()[idx].x();
                          const auto y = this->position()[idx].y();
                          return xtd::atan2(y, x);
                        }
                        SOA_HOST_DEVICE auto eta(std::integral auto idx) const {
                          const auto x = this->position()[idx].x();
                          const auto y = this->position()[idx].y();
                          const auto z = this->position()[idx].z();
                          const auto theta = xtd::atan(xtd::sqrt(x * x + y * y) / z);
                          return -xtd::log(xtd::tan(theta / 2));
                        }
                      )
  )
  // clang-format on

  using CaloClusterSoA = CaloClusterSoALayout<>;
  using CaloClusterSoAView = CaloClusterSoA::View;
  using CaloClusterSoAConstView = CaloClusterSoA::ConstView;

}  // namespace reco

#endif
