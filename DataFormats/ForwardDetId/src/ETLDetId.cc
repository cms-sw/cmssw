#include "DataFormats/ForwardDetId/interface/ETLDetId.h"
#include <iomanip>

std::ostream& operator<< ( std::ostream& os, const ETLDetId& id ) {
  os << ( MTDDetId& ) id;
  os << " ETL " << std::endl
     << " Side        : " << id.mtdSide() << std::endl
     << " Ring        : " << id.mtdRR() << std::endl
     << " Module      : " << id.etlModule() << std::endl
     << " Module type : " << id.etlModType() << std::endl;
  return os;
}
