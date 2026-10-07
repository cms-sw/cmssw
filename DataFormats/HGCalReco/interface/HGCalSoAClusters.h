#ifndef DataFormats_HGCalReco_interface_HGCalSoAClusters_h
#define DataFormats_HGCalReco_interface_HGCalSoAClusters_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/HGCalReco/interface/HGCalRecoVariations.h"

GENERATE_SOA_LAYOUT(HGCalSoAClustersLayout,
                    // columns: one value per element
                    SOA_COLUMN(float, x),
                    SOA_COLUMN(float, y),
                    SOA_COLUMN(float, z),
                    SOA_COLUMN(float, energy),
                    SOA_COLUMN(int, cells),  // number of hits in the cluster
                    SOA_COLUMN(int, seed),   // This is the index of the seed of each cluster inside the RecHit SoA
                    SOA_COLUMN(float, time),  // sum(w t)/sum(w), w = max(E,0), hits with t > 0; -99 if none
                    // by hgcalreco::Variation, summed from the member rechits (no re-clustering)
                    SOA_EIGEN_COLUMN(hgcalreco::VariationVector, energy_variations),
                    SOA_EIGEN_COLUMN(hgcalreco::VariationVector, time_variations)
)

using HGCalSoAClusters = HGCalSoAClustersLayout<>;

#endif
