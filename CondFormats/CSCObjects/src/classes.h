#include "CondFormats/CSCObjects/interface/CSCPedestals.h"
namespace{
 std::map< int, std::vector<CSCPedestals::Item> > pedmap;
}
#include "CondFormats/CSCObjects/interface/CSCGains.h"
namespace{
  std::map< int, std::vector<CSCGains::Item> > gmap;
}
#include "CondFormats/CSCObjects/interface/CSCDBGains.h"
namespace{
  std::vector<CSCDBGains::Item> gcontainer;
  //std::map< int, std::vector< CSCGains::Item> > gmap;
}

#include "CondFormats/CSCObjects/interface/CSCNoiseMatrix.h"
namespace{
  std::map< int, std::vector< CSCNoiseMatrix::Item> > mmap;
}
#include "CondFormats/CSCObjects/interface/CSCcrosstalk.h"
namespace{
  std::map< int, std::vector< CSCcrosstalk::Item> > cmap;
}
#include "CondFormats/CSCObjects/interface/CSCIdentifier.h"
#include "CondFormats/CSCObjects/interface/CSCReadoutMapping.h"
#include "CondFormats/CSCObjects/interface/CSCTriggerMapping.h"
