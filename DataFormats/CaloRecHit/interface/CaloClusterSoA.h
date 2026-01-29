
#pragma once

// Authors: Simone Balducci, Felice Pantaleo, Wahid Redjeb, Aurora Perego, Leonardo Beltrame

#include "DataFormats/SoATemplate/interface/SoABlocks.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloID.h"
#include "DataFormats/DetId/interface/DetId.h"

namespace reco {

  GENERATE_SOA_LAYOUT(
      CaloClusterSoAPosition, SOA_COLUMN(float, x), SOA_COLUMN(float, y), SOA_COLUMN(float, z), SOA_COLUMN(int, cells))

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

  GENERATE_SOA_BLOCKS(CaloClusterSoALayout,
                      SOA_BLOCK(position, CaloClusterSoAPosition),
                      SOA_BLOCK(energy, CaloClusterSoAEnergy),
                      SOA_BLOCK(indexes, CaloClusterSoAIndexes),
                      SOA_BLOCK(timing, CaloClusterSoATiming))

  using CaloClusterSoA = CaloClusterSoALayout<>;
  using CaloClusterSoAView = CaloClusterSoA::View;
  using CaloClusterSoAConstView = CaloClusterSoA::ConstView;

}  // namespace reco
