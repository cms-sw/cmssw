#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include <iomanip>

std::ostream& operator<< ( std::ostream& os, const BTLDetId& id ) {
  os << ( MTDDetId& ) id;
  os << " BTL " << std::endl
     << " Side        : " << id.mtdSide() << std::endl
     << " Rod         : " << id.mtdRR() << std::endl
     << " Module      : " << id.module() << std::endl
     << " Crystal type: " << id.modType() << std::endl
     << " Crystal     : " << id.crystal() << std::endl;
  return os;
}
