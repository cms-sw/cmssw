#ifndef DataFormats_FTLRecHitSoA_interface_ETLBaseRecHitHostCollection_h
#define DataFormats_FTLRecHitSoA_interface_ETLBaseRecHitHostCollection_h

#include "DataFormats/FTLRecHitSoA/interface/ETLBaseRecHitSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace etlrechit {

  using ETLBaseRecHitHostCollection = PortableHostCollection<ETLBaseRecHitSoA>;

}  // namespace etlrechit

#endif  // DataFormats_FTLRecHitSoA_interface_ETLBaseRecHitHostCollection_h
