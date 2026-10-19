#ifndef DataFormats_HGCalReco_interface_HGCalSoAClustersExtra_h
#define DataFormats_HGCalReco_interface_HGCalSoAClustersExtra_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/HGCalReco/interface/HGCalRecoVariations.h"

GENERATE_SOA_LAYOUT(
    HGCalSoAClustersExtraLayout,
    // columns: one value per element
    SOA_COLUMN(float, total_weight),
    SOA_COLUMN(float, total_weight_log),
    SOA_COLUMN(float, maxEnergyValue),
    SOA_COLUMN(int, maxEnergyIndex),  // Index in the RecHitSoA of the rechit with highest energy in each cluster
    // energy-weighted time accumulators
    SOA_COLUMN(float, time_weight),
    SOA_COLUMN(float, time_weighted_sum),
    SOA_EIGEN_COLUMN(hgcalreco::VariationVector, time_weight_variations),
    SOA_EIGEN_COLUMN(hgcalreco::VariationVector, time_weighted_sum_variations))

using HGCalSoAClustersExtra = HGCalSoAClustersExtraLayout<>;

#endif
