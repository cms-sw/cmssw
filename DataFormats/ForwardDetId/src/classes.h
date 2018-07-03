#include <boost/cstdint.hpp> 
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCTriggerDetId.h"
#include "DataFormats/ForwardDetId/interface/FastTimeDetId.h"
#include "DataFormats/ForwardDetId/interface/MTDDetId.h"
#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"

namespace DataFormats_ForwardDetId {
  struct dictionary {

    //EE specific
    HGCEEDetId anHGCEEDetId;

    //HE specific
    HGCHEDetId anHGCHEDetId;

    //HGCal specific
    HGCalDetId anHGCalDetId;

    //HGCal specific (new format)
    HGCSiliconDetId anHGCSiliconDetid;
    HGCScintillatorDetId anHGCScintillatorDetId;

    //FastTimer specific
    FastTimeDetId anFastTimeDetId;

    //MTD specific 
    MTDDetId anMTDDetId;

    //BTL specific 
    BTLDetId aBTLDetId;

    //ETL specific 
    ETLDetId anETLDetId;
  };
}
