#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include <iomanip>

std::ostream& operator<< ( std::ostream& os, const BTLDetId& id ) {
  os << ( MTDDetId& ) id;
  os << " BTL " << std::endl
     << " Side        : " << id.mtdSide() << std::endl
     << " Rod         : " << id.mtdRR() << std::endl
     << " Module      : " << id.btlModule() << std::endl
     << " Crystal type: " << id.btlmodType() << std::endl
     << " Crystal     : " << id.btlCrystal() << std::endl;
  return os;
}
