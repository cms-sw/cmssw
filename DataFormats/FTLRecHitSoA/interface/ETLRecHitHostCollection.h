#ifndef DataFormats_FTLRecHitSoA_interface_ETLRecHitHostCollection_h
#define DataFormats_FTLRecHitSoA_interface_ETLRecHitHostCollection_h

#include "DataFormats/FTLRecHitSoA/interface/ETLRecHitSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace etlrechit {

  using ETLRecHitHostCollection = PortableHostCollection<ETLRecHitSoA>;

}  // namespace etlrechit

#endif  // DataFormats_FTLRecHitSoA_interface_ETLRecHitHostCollection_h
