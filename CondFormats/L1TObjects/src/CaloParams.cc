#include "CondFormats/L1TObjects/interface/CaloParams.h"

using namespace l1t;

int CaloParams::etSumEtaMin(unsigned isum) const {
  if (etSumEtaMin_.size()>isum) return etSumEtaMin_.at(isum);
  else return 0;
}

int CaloParams::etSumEtaMax(unsigned isum) const {
  if (etSumEtaMax_.size()>isum) return etSumEtaMax_.at(isum);
  else return 0;
}

double CaloParams::etSumEtThreshold(unsigned isum) const {
  if (etSumEtThreshold_.size()>isum) return etSumEtThreshold_.at(isum);
  else return 0.;
}

void CaloParams::setEtSumEtaMin(unsigned isum, int eta) {
  if (etSumEtaMin_.size()<=isum) etSumEtaMin_.resize(isum+1);
  etSumEtaMin_.at(isum) = eta;
}

void CaloParams::setEtSumEtaMax(unsigned isum, int eta) {
  if (etSumEtaMax_.size()<=isum) etSumEtaMax_.resize(isum+1);
  etSumEtaMax_.at(isum) = eta;
}

void CaloParams::setEtSumEtThreshold(unsigned isum, double thresh) {
  if (etSumEtThreshold_.size()<=isum) etSumEtThreshold_.resize(isum+1);
  etSumEtThreshold_.at(isum) = thresh;
}

void CaloParams::print(std::ostream& out) const {

  out << "L1 Calo Parameters" << std::endl;
  out << "Towers" << std::endl;
  out << " LSB H            : " << this->towerLsbH() << std::endl;
  out << " LSB E            : " << this->towerLsbE() << std::endl;
  out << " LSB Sum          : " << this->towerLsbSum() << std::endl;
  out << " Nbits H          : " << this->towerNBitsH() << std::endl;
  out << " Nbits E          : " << this->towerNBitsE() << std::endl;
  out << " Nbits Sum        : " << this->towerNBitsSum() << std::endl;
  out << " Nbits Ratio      : " << this->towerNBitsRatio() << std::endl;
  out << " Mask E           : " << this->towerMaskE() << std::endl;  
  out << " Mask H           : " << this->towerMaskH() << std::endl;  
  out << " Mask Sum         : " << this->towerMaskSum() << std::endl;  
  out << " Mask Ratio       : " << this->towerMaskRatio() << std::endl;  
  out << " Encoding         : " << this->doTowerEncoding() << std::endl;

  out << "Regions" << std::endl;
  out << " PUS              : " << this->regionPUSType() << std::endl;
  out << " LSB              : " << this->regionLsb() << std::endl;

  out << "EG" << std::endl;
  out << " LSB              : " << this->egLsb() << std::endl;
  out << " Seed thresh      : " << this->egSeedThreshold() << std::endl;
  out << " Neighbour thresh : " << this->egNeighbourThreshold() << std::endl;
  out << " HCAL thresh      : " << this->egHcalThreshold() << std::endl;
  out << " HCAL max Et      : " << this->egMaxHcalEt() << std::endl;
  out << " Iso PUS type     : " << this->egPUSType() << std::endl;

  out << "Tau" << std::endl;
  out << " Seed thresh      : " << this->tauSeedThreshold() << std::endl;
  out << " Neighbour thresh : " << this->tauNeighbourThreshold() << std::endl;
  out << " Iso PUS type     : " << this->tauPUSType() << std::endl;

  out << "Jets" << std::endl;
  out << " LSB              : " << this->jetLsb() << std::endl;
  out << " Seed thresh      : " << this->jetSeedThreshold() << std::endl;
  out << " Neighbour thresh : " << this->jetNeighbourThreshold() << std::endl;
  out << " PUS type         : " << this->jetPUSType() << std::endl;
  out << " Calibration type : " << this->jetCalibrationType() << std::endl;

  out << "Sums" << std::endl;
  for (unsigned i=0; i<etSumEtaMin_.size(); ++i) {
    out << " EtSum" << i << " eta min     : " << this->etSumEtaMin(i) << std::endl;
    if (etSumEtaMax_.size()>i) out << " EtSum" << i << " eta max     : " << this->etSumEtaMax(i) << std::endl;
    if (etSumEtThreshold_.size()>i) out << " EtSum" << i << " Et thresh     : " << this->etSumEtThreshold(i) << std::endl;
  }

}
