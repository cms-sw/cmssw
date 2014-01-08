#include "CondFormats/L1TObjects/interface/CaloParams.h"

using namespace l1t;

void CaloParams::print(std::ostream& out) const {
  out << "param:  " << param() << std::endl;
}
