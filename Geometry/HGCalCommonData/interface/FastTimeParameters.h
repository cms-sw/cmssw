#ifndef Geometry_HGCalCommonData_FastTimeParameters_h
#define Geometry_HGCalCommonData_FastTimeParameters_h

#include <cstdint>
#include <string>
#include <vector>
#include "CondFormats/Serialization/interface/Serializable.h"

class FastTimeParameters {
 public:
  FastTimeParameters();
  ~FastTimeParameters();

  int nZBarrel_;
  int nPhiBarrel_;
  int nEtaEndcap_;
  int nPhiEndcap_;
  std::vector<double> geomParBarrel_;
  std::vector<double> geomParEndcap_;

  COND_SERIALIZABLE;
};

#endif
