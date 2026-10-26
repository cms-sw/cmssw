#ifndef DataFormats_HGCalReco_interface_HGCalSoARecHitsExtra_h
#define DataFormats_HGCalReco_interface_HGCalSoARecHitsExtra_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

// Layer clustering of RecHits: the layer cluster id of each rechit,
// whether it is the seed of the layer cluster and the number of layer clusters
GENERATE_SOA_LAYOUT(HGCalSoARecHitsExtraLayout,
                    // columns: one value per element
                    SOA_COLUMN(int, clusterIndex),
                    SOA_COLUMN(uint8_t, isSeed),
                    SOA_SCALAR(unsigned int, numberOfClustersScalar))

using HGCalSoARecHitsExtra = HGCalSoARecHitsExtraLayout<>;

#endif
