#include "CondFormats/GeometryObjects/interface/HcalParameters.h"

double HcalParameters::getEtaHF(const unsigned int i) const {
  unsigned int k = rTable.size()-i-1;
  double eta = -log(tan(0.5*atan(rTable[k]/gparHF[4])));
  return eta;
}

std::vector<double> HcalParameters::getEtaTableHF() const {

  std::vector<double> etas;
  for (unsigned int i=0; i<rTable.size(); ++i) {
    etas.push_back(getEtaHF(i));
  }
  return etas;
}
