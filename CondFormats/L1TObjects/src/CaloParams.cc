#include "CondFormats/L1TObjects/interface/CaloParams.h"

using namespace l1t;

void CaloParams::print(std::ostream& out) const {

  out << "L1 Calo Parameters" << std::endl;
  out << "Towers" << std::endl;
  out << " LSB H       : " << this->towerLsbH() << std::endl;
  out << " LSB E       : " << this->towerLsbE() << std::endl;
  out << " LSB Sum     : " << this->towerLsbSum() << std::endl;
  out << " Nbits H     : " << this->towerNBitsH() << std::endl;
  out << " Nbits E     : " << this->towerNBitsE() << std::endl;
  out << " Nbits Sum   : " << this->towerNBitsSum() << std::endl;
  out << " Nbits Ratio : " << this->towerNBitsRatio() << std::endl;
  out << " Mask E      : " << this->towerMaskE() << std::endl;  
  out << " Mask H      : " << this->towerMaskH() << std::endl;  
  out << " Mask Sum    : " << this->towerMaskSum() << std::endl;  
  out << " Mask Ratio  : " << this->towerMaskRatio() << std::endl;  
  out << " Compression : " << this->doTowerCompression() << std::endl;
  out << "EG" << std::endl;
  out << "Tau" << std::endl;
  out << "Jets" << std::endl;
  out << " seed threshold : " << this->jetSeedThreshold() << std::endl;
  out << "Sums" << std::endl;
}
