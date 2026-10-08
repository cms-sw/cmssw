#ifndef DataFormats_FTLRecHitSoA_interface_BTLBaseRecHitHostCollection_h
#define DataFormats_FTLRecHitSoA_interface_BTLBaseRecHitHostCollection_h

#include "DataFormats/FTLRecHitSoA/interface/BTLBaseRecHitSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace btlrechit {

  using BTLBaseRecHitHostCollection = PortableHostCollection<BTLBaseRecHitSoA>;

}  // namespace btlrechit

#endif  // DataFormats_FTLRecHitSoA_interface_BTLBaseRecHitHostCollection_h
