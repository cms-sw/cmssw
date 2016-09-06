#include <boost/cstdint.hpp> 
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCTriggerDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCTriggerHexDetId.h"
#include "DataFormats/ForwardDetId/interface/FastTimeDetId.h"

namespace DataFormats_ForwardDetId {
  struct dictionary {

    //EE specific
    HGCEEDetId anHGCEEDetId;

    //HE specific
    HGCHEDetId anHGCHEDetId;

    //HGCal specific
    HGCalDetId anHGCalDetId;

    //HGCal trigger specific
    HGCTriggerHexDetId anHGCTriggerHexDetId;

    //FastTimer specific
    FastTimeDetId anFastTimeDetId;
  };
}
