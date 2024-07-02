#ifndef DataFormats_HGCalReco_interface_HGCalSoARecHits_h
#define DataFormats_HGCalReco_interface_HGCalSoARecHits_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

// SoA layout with dim1, dim2, weight, sigmaNoise, recHitsIndex layer and cellsCount fields
GENERATE_SOA_LAYOUT(HGCalSoARecHitsLayout,
                    // columns: one value per element
                    SOA_COLUMN(float, dim1),
                    SOA_COLUMN(float, dim2),
                    SOA_COLUMN(float, dim3),
                    SOA_COLUMN(int, layer),
                    SOA_COLUMN(float, weight),
                    SOA_COLUMN(float, sigmaNoise),
                    SOA_COLUMN(unsigned int, recHitIndex),
                    SOA_COLUMN(uint32_t, detid),
                    SOA_COLUMN(float, time),
                    SOA_COLUMN(float, timeError))

using HGCalSoARecHits = HGCalSoARecHitsLayout<>;

#endif  // DataFormats_PortableTestObjects_interface_TestSoA_h
