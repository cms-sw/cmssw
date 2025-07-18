#ifndef DataFormats_HGCalRecHit_interface_HGCalRecHitHost_h
#define DataFormats_HGCalRecHit_interface_HGCalRecHitHost_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/HGCalRecHit/interface/HGCalRecHitSoA.h"

namespace hgcalrechit {

  // SoA with x, y, z, id fields in host memory
  using HGCalRecHitHost = PortableHostCollection<HGCalRecHitSoA>;

}  // namespace hgcalrechit

#endif  // DataFormats_HGCalRecHit_interface_HGCalRecHitHost_h
