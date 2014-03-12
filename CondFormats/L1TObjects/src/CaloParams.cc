#include "CondFormats/L1TObjects/interface/CaloParams.h"

using namespace l1t;

void CaloParams::print(std::ostream& out) const {

  out << "L1 Calo Parameters" << std::endl;

//   out << this->firmwarePP() << std::endl;
//   out << this->firmwareMP() << std::endl;
  out << this->towerLsbH() << std::endl;
  out << this->towerLsbE() << std::endl;
  out << this->towerNBitsH() << std::endl;
  out << this->towerNBitsE() << std::endl;
  
}
