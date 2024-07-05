#ifndef DataFormats_HGCalReco_interface_HGCalSoARecHitsExtraHostCollection_h
#define DataFormats_HGCalReco_interface_HGCalSoARecHitsExtraHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/HGCalReco/interface/HGCalSoARecHitsExtra.h"

// SoA with delta, rho, weight, nearestHigher, clusterIndex, layer, isSeed, and cellsCount fields in host memory
using HGCalSoARecHitsExtraHostCollection = PortableHostCollection<HGCalSoARecHitsExtra>;

#endif  // DataFormats_HGCalReco_interface_HGCalSoARecHitsExtraHostCollection_h
