#ifndef DataFormats_HGCalReco_interface_HGCalSoARecHitsHostCollection_h
#define DataFormats_HGCalReco_interface_HGCalSoARecHitsHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/HGCalReco/interface/HGCalSoARecHits.h"

// SoA with x, y, z, id fields in host memory
using HGCalSoARecHitsHostCollection = PortableHostCollection<HGCalSoARecHits>;

#endif  // DataFormats_HGCalReco_interface_HGCalSoARecHitsHostCollection_h
