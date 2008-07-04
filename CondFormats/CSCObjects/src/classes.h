#include "CondFormats/CSCObjects/interface/CSCPedestals.h"
namespace{
 std::map< int, std::vector<CSCPedestals::Item> > pedmap;
}
#include "CondFormats/CSCObjects/interface/CSCDBPedestals.h"
namespace{
 std::vector<CSCDBPedestals::Item> pedcontainer;
}
#include "CondFormats/CSCObjects/interface/CSCGains.h"
namespace{
  std::map< int, std::vector<CSCGains::Item> > gmap;
}
#include "CondFormats/CSCObjects/interface/CSCDBGains.h"
namespace{
  std::vector<CSCDBGains::Item> gcontainer;
}
#include "CondFormats/CSCObjects/interface/CSCNoiseMatrix.h"
namespace{
  std::map< int, std::vector< CSCNoiseMatrix::Item> > mmap;
}
#include "CondFormats/CSCObjects/interface/CSCDBNoiseMatrix.h"
namespace{
  std::vector<CSCDBNoiseMatrix::Item> mcontainer;
}
#include "CondFormats/CSCObjects/interface/CSCcrosstalk.h"
namespace{
  std::map< int, std::vector< CSCcrosstalk::Item> > cmap;
}
#include "CondFormats/CSCObjects/interface/CSCDBCrosstalk.h"
namespace{
  std::vector<CSCDBCrosstalk::Item> ccontainer;
}
#include "CondFormats/CSCObjects/interface/CSCBadStrips.h"
namespace{
  std::vector<CSCBadStrips::BadChamber> bschmcontainer;
  std::vector<CSCBadStrips::BadChannel> bschncontainer;
}
#include "CondFormats/CSCObjects/interface/CSCBadWires.h"
namespace{
  std::vector<CSCBadWires::BadChamber> bwchmcontainer;
  std::vector<CSCBadWires::BadChannel> bwchncontainer;
}
#include "CondFormats/CSCObjects/interface/CSCChamberMap.h"
namespace{
  std::map< int, CSCMapItem::MapItem > chmap;
}
#include "CondFormats/CSCObjects/interface/CSCCrateMap.h"
namespace{
  std::map< int, CSCMapItem::MapItem > cratemap;
}
#include "CondFormats/CSCObjects/interface/CSCDDUMap.h"
namespace{
  std::map< int, CSCMapItem::MapItem > ddumap;
}
#include "CondFormats/CSCObjects/interface/CSCChamberIndex.h"
namespace{
  std::map< int, CSCMapItem::MapItem > indexmap;
}

#include "CondFormats/CSCObjects/interface/CSCIdentifier.h"
#include "CondFormats/CSCObjects/interface/CSCReadoutMapping.h"
#include "CondFormats/CSCObjects/interface/CSCTriggerMapping.h"
#include "CondFormats/CSCObjects/interface/CSCL1TPParameters.h"
