#include <boost/cstdint.hpp> 
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/FastTimeDetId.h"

namespace DataFormats_ForwardDetId {
  struct dictionary {

    //EE specific
    HGCEEDetId anHGCEEDetId;

    //HE specific
    HGCHEDetId anHGCHEDetId;

    //HGCal specific
    HGCalDetId anHGCalDetId;

    //FastTimer specific
    FastTimeDetId anFastTimeDetId;
  };
}
