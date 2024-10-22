#ifndef DataFormats_HGCalReco_interface_HGCalSoAClustersExtra_h
#define DataFormats_HGCalReco_interface_HGCalSoAClustersExtra_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

GENERATE_SOA_LAYOUT(
    HGCalSoAClustersExtraLayout,
    // columns: one value per element
    SOA_COLUMN(float, total_weight),
    SOA_COLUMN(float, total_weight_log),
    SOA_COLUMN(float, maxEnergyValue),
    SOA_COLUMN(int, maxEnergyIndex)  // Index in the RecHitSoA of the rechit with highest energy in each cluster
)

using HGCalSoAClustersExtra = HGCalSoAClustersExtraLayout<>;

#endif
